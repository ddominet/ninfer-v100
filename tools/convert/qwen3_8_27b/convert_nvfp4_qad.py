"""Convert an NVIDIA ModelOpt mixed-precision NVFP4/FP8 Qwen3.8-27B checkpoint
(e.g. ``nvidia/Qwen3.8-27B-NVFP4-QAD`` or ``nvidia/Qwen3.8-27B-NVFP4``) to a
NInfer ``.ninfer`` artifact.

Canonical invocation::

    python3 -m tools.convert.qwen3_8_27b.convert_nvfp4_qad \
      --model /path/to/Qwen3.8-27B-base-bf16 \
      --quantized-model /path/to/Qwen3.8-27B-NVFP4-QAD \
      --dflash2-model /path/to/Qwen3.8-27B-DFlash2 \
      --out out/qwen3_8_27b_nvfp4_qad.ninfer
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import struct
import time
from typing import Iterable, Mapping, Sequence

import torch

from tools.artifact.container import (
    ArtifactIdentity,
    ArtifactObject,
    ArtifactWriter,
)
from tools.artifact.layouts import (
    encode_direct,
    encode_fp8_row_scaled,
    encode_nvfp4,
)
from tools.artifact.numeric import valid_positive_fp32_word
from tools.convert.common.quantize import pick_device
from tools.convert.common.safetensors import ShardReader
from tools.convert.qwen3_6.common import conversion as family_conversion
from tools.convert.qwen3_6.common import inventory as common_inventory
from tools.convert.qwen3_6.common import recipe as family_recipe
from tools.convert.qwen3_6_27b import draft_head
from tools.convert.qwen3_6_27b import recipe as official_recipe

from . import convert as base_convert
from . import dflash2_recipe, fp8_embedding
from . import inventory_nvfp4 as inv
from . import recipe_nvfp4 as nvfp4_recipe
from .dflash2_inventory import DFLASH2_TENSOR_SPECS


RECIPE_ID = "qwen3_8_27b_nvfp4_qad-v1"


@dataclass(frozen=True, slots=True)
class ModelOptLayerPlan:
    gdn_nvfp4_layers: frozenset[int]
    mlp_nvfp4_layers: frozenset[int]
    output_head_nvfp4: bool


def parse_modelopt_quant_config(quant_dir: Path) -> ModelOptLayerPlan:
    """Validate and parse NVIDIA ModelOpt's ``hf_quant_config.json``."""
    cfg_path = quant_dir / "hf_quant_config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(
            f"{cfg_path}: ModelOpt checkpoint must provide hf_quant_config.json"
        )
    data = family_conversion.load_json(cfg_path)
    producer = data.get("producer", {})
    if not isinstance(producer, Mapping) or producer.get("name") != "modelopt":
        raise ValueError(f"{cfg_path}: expected producer.name == 'modelopt'")
    quant = data.get("quantization", {})
    if not isinstance(quant, Mapping):
        raise ValueError(f"{cfg_path}: missing quantization object")
    layers = quant.get("quantized_layers")
    if not isinstance(layers, Mapping) or not layers:
        raise ValueError(f"{cfg_path}: quantization.quantized_layers is empty")

    def layer_algo(key: str) -> str:
        entry = layers.get(key)
        if not isinstance(entry, Mapping):
            raise ValueError(f"{cfg_path}: missing quantized_layers entry '{key}'")
        algo = entry.get("quant_algo")
        if algo not in ("NVFP4", "FP8"):
            raise ValueError(
                f"{cfg_path}: unsupported quant_algo {algo!r} for '{key}'"
            )
        if algo == "NVFP4" and entry.get("group_size") != 16:
            raise ValueError(
                f"{cfg_path}: NVFP4 layer '{key}' must have group_size=16"
            )
        return str(algo)

    gdn_nvfp4: set[int] = set()
    mlp_nvfp4: set[int] = set()
    for layer in range(64):
        prefix = f"model.language_model.layers.{layer}."
        if layer in inv.FULL_ATTENTION_LAYERS:
            for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
                algo = layer_algo(f"{prefix}self_attn.{proj}")
                if algo != "FP8":
                    raise ValueError(
                        f"{cfg_path}: full-attention layer {layer} {proj} expected FP8, got {algo}"
                    )
        else:
            qkv_algo = layer_algo(f"{prefix}linear_attn.in_proj_qkv")
            z_algo = layer_algo(f"{prefix}linear_attn.in_proj_z")
            out_algo = layer_algo(f"{prefix}linear_attn.out_proj")
            if not (qkv_algo == z_algo == out_algo):
                raise ValueError(
                    f"{cfg_path}: GDN layer {layer} mixes formats ({qkv_algo}, {z_algo}, {out_algo})"
                )
            if qkv_algo == "NVFP4":
                gdn_nvfp4.add(layer)

        gate_algo = layer_algo(f"{prefix}mlp.gate_proj")
        up_algo = layer_algo(f"{prefix}mlp.up_proj")
        down_algo = layer_algo(f"{prefix}mlp.down_proj")
        if not (gate_algo == up_algo == down_algo):
            raise ValueError(
                f"{cfg_path}: MLP layer {layer} mixes formats ({gate_algo}, {up_algo}, {down_algo})"
            )
        if gate_algo == "NVFP4":
            mlp_nvfp4.add(layer)

    head_entry = layers.get("lm_head")
    head_nvfp4 = isinstance(head_entry, Mapping) and head_entry.get("quant_algo") == "NVFP4"
    if head_entry is not None and not head_nvfp4 and head_entry.get("quant_algo") != "FP8":
        raise ValueError(
            f"{cfg_path}: unsupported lm_head quant_algo {head_entry.get('quant_algo')!r}"
        )
    return ModelOptLayerPlan(
        gdn_nvfp4_layers=frozenset(gdn_nvfp4),
        mlp_nvfp4_layers=frozenset(mlp_nvfp4),
        output_head_nvfp4=head_nvfp4,
    )


def _positive_scalar_f32(tensor: torch.Tensor, name: str) -> float:
    if tensor.numel() != 1:
        raise ValueError(f"{name}: expected scalar tensor, got shape {tuple(tensor.shape)}")
    value = float(tensor.detach().cpu().to(torch.float32).item())
    if not (value > 0.0 and torch.isfinite(torch.tensor(value))):
        raise ValueError(f"{name}: scale must be finite and positive, got {value!r}")
    return value


def scale_to_divisor_word(scale: float, name: str) -> int:
    """Convert ModelOpt multiplier scale -> NInfer FP32 divisor bits (1.0 / scale)."""
    if not (scale > 0.0):
        raise ValueError(f"{name}: scale must be positive")
    divisor = struct.unpack("<f", struct.pack("<f", 1.0 / scale))[0]
    word = struct.unpack("<I", struct.pack("<f", divisor))[0]
    if not valid_positive_fp32_word(word):
        raise ValueError(f"{name}: inverted divisor {divisor!r} is invalid")
    return word


def fuse_modelopt_nvfp4_parts(
    parts: Sequence[tuple[torch.Tensor, torch.Tensor, float]],
    label: str,
) -> tuple[torch.Tensor, torch.Tensor, bytes]:
    """Concatenate one or more ModelOpt NVFP4 projections along N.

    Each part is ``(packed_u8[N_i, K/2], block_scale_f8[N_i, K/16], weight_scale_2_f32)``.
    If parts share the exact same ``weight_scale_2`` their E4M3 scales pass through
    untouched; otherwise smaller-scale parts are rescaled down by ``s2_i / max(s2) <= 1.0``
    so E4M3 never overflows.
    """
    if not parts:
        raise ValueError(f"{label}: empty NVFP4 part list")
    max_s2 = max(s2 for _, _, s2 in parts)
    packed_list: list[torch.Tensor] = []
    scale_list: list[torch.Tensor] = []
    for packed, scales, s2 in parts:
        if packed.dtype != torch.uint8 or scales.dtype != torch.float8_e4m3fn:
            raise ValueError(f"{label}: invalid ModelOpt NVFP4 part dtypes")
        packed_list.append(packed.contiguous())
        if s2 == max_s2:
            scale_list.append(scales.view(torch.uint8).contiguous())
        else:
            ratio = s2 / max_s2
            rescaled = (scales.to(torch.float32) * ratio).to(torch.float8_e4m3fn)
            scale_list.append(rescaled.view(torch.uint8).contiguous())
    fused_packed = packed_list[0] if len(packed_list) == 1 else torch.cat(packed_list, dim=0)
    fused_scales = scale_list[0] if len(scale_list) == 1 else torch.cat(scale_list, dim=0)
    divisor_word = scale_to_divisor_word(max_s2, label)
    return fused_packed, fused_scales, struct.pack("<I", divisor_word)


def materialize_modelopt_fp8_weight(
    recipe_obj: nvfp4_recipe.Fp8WeightRecipe,
    reader: ShardReader,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize a ModelOpt FP8 weight (broadcasting per-tensor F32 scale to per-row BF16)."""
    code_parts: list[torch.Tensor] = []
    scale_parts: list[torch.Tensor] = []
    cache: dict[nvfp4_recipe.MatrixSource, tuple[torch.Tensor, torch.Tensor]] = {}
    for part in recipe_obj.parts:
        cached = cache.get(part.source)
        if cached is None:
            codes = reader.get(part.source.field("weight"))
            raw_scale = reader.get(part.source.field("weight_scale"))
            if codes.dtype != torch.float8_e4m3fn or tuple(codes.shape) != part.source.shape:
                raise ValueError(
                    f"{part.source.name}: FP8 weight signature mismatch "
                    f"{codes.dtype} {tuple(codes.shape)} vs {part.source.shape}"
                )
            if raw_scale.numel() == 1:
                scale_val = _positive_scalar_f32(raw_scale, part.source.field("weight_scale"))
                row_scales = torch.full(
                    (part.source.shape[0],), scale_val, dtype=torch.bfloat16
                )
            else:
                flat = raw_scale.reshape(-1)
                if tuple(flat.shape) != (part.source.shape[0],):
                    raise ValueError(
                        f"{part.source.name}: unexpected FP8 scale shape {tuple(raw_scale.shape)}"
                    )
                row_scales = flat.to(torch.bfloat16)
            cached = (codes.view(torch.uint8), row_scales)
            cache[part.source] = cached
        code_parts.append(nvfp4_recipe._select_rows(cached[0], part))
        scale_parts.append(nvfp4_recipe._select_rows(cached[1], part))
    fused_codes = code_parts[0].contiguous() if len(code_parts) == 1 else torch.cat(code_parts, dim=0)
    fused_scales = scale_parts[0].contiguous() if len(scale_parts) == 1 else torch.cat(scale_parts, dim=0)
    return fused_codes, fused_scales


def materialize_modelopt_nvfp4_weight(
    recipe_obj: nvfp4_recipe.Nvfp4WeightRecipe,
    reader: ShardReader,
) -> tuple[torch.Tensor, torch.Tensor, bytes]:
    parts: list[tuple[torch.Tensor, torch.Tensor, float]] = []
    cache: dict[nvfp4_recipe.MatrixSource, tuple[torch.Tensor, torch.Tensor, float]] = {}
    for part in recipe_obj.parts:
        cached = cache.get(part.source)
        if cached is None:
            n, k = part.source.shape
            packed = reader.get(part.source.field("weight"))
            scales = reader.get(part.source.field("weight_scale"))
            s2 = _positive_scalar_f32(
                reader.get(part.source.field("weight_scale_2")),
                part.source.field("weight_scale_2"),
            )
            if (
                packed.dtype != torch.uint8
                or tuple(packed.shape) != (n, k // 2)
                or scales.dtype != torch.float8_e4m3fn
                or tuple(scales.shape) != (n, k // 16)
            ):
                raise ValueError(
                    f"{part.source.name}: ModelOpt NVFP4 source signature mismatch"
                )
            cached = (packed, scales, s2)
            cache[part.source] = cached
        p_rows = nvfp4_recipe._select_rows(cached[0], part)
        s_rows = nvfp4_recipe._select_rows(cached[1], part)
        parts.append((p_rows, s_rows, cached[2]))
    return fuse_modelopt_nvfp4_parts(parts, recipe_obj.object_name)


def materialize_modelopt_input_divisor(
    sources: Sequence[nvfp4_recipe.MatrixSource],
    reader: ShardReader,
    label: str,
) -> torch.Tensor:
    scales = [
        _positive_scalar_f32(reader.get(src.field("input_scale")), src.field("input_scale"))
        for src in sources
    ]
    # Shared input activations have identical calibrated input_scale; if slightly distinct across
    # fused sibling projections, pick max(input_scale) (most conservative dynamic range).
    word = scale_to_divisor_word(max(scales), label)
    return torch.frombuffer(bytearray(struct.pack("<I", word)), dtype=torch.float32).reshape(())


def build_qad_plan(
    layer_plan: ModelOptLayerPlan,
    *,
    embedding_format: str = inv.W8,
    include_dflash2: bool = True,
) -> tuple[
    tuple[inv.TensorSpec, ...],
    dict[str, nvfp4_recipe.Fp8WeightRecipe],
    dict[str, nvfp4_recipe.Nvfp4WeightRecipe],
    dict[str, tuple[nvfp4_recipe.MatrixSource, ...]],
    dict[str, family_recipe.TensorRecipe],
]:
    """Build object specs and source mappings for a ModelOpt NVFP4-QAD checkpoint."""
    specs: list[inv.TensorSpec] = [
        inv.tensor_spec("text/token_embedding", (248320, 5120), embedding_format),
    ]
    fp8_weights: dict[str, nvfp4_recipe.Fp8WeightRecipe] = {}
    nvfp4_weights: dict[str, nvfp4_recipe.Nvfp4WeightRecipe] = {}
    input_divisors: dict[str, tuple[nvfp4_recipe.MatrixSource, ...]] = {}
    quant_direct: dict[str, family_recipe.TensorRecipe] = {}

    def src(name: str, n: int, k: int) -> nvfp4_recipe.MatrixSource:
        return nvfp4_recipe.MatrixSource(name, (n, k))

    def all_rows(s: nvfp4_recipe.MatrixSource) -> nvfp4_recipe.MatrixPart:
        return nvfp4_recipe.MatrixPart(s, (nvfp4_recipe.RowRange(0, s.shape[0]),))

    for layer in range(64):
        sp = f"model.language_model.layers.{layer}."
        op = f"text/layers/{layer}/"
        specs.append(inv.tensor_spec(op + "input_norm", (5120,), inv.BF16))
        quant_direct[op + "input_norm"] = family_recipe.TensorRecipe(
            op + "input_norm",
            family_recipe.source(sp + "input_layernorm.weight", (5120,)),
        )
        if layer in inv.FULL_ATTENTION_LAYERS:
            q = src(sp + "self_attn.q_proj", 12288, 5120)
            k = src(sp + "self_attn.k_proj", 1024, 5120)
            v = src(sp + "self_attn.v_proj", 1024, 5120)
            o = src(sp + "self_attn.o_proj", 5120, 6144)
            specs.extend(
                (
                    inv.tensor_spec(op + "attention/query_key_gate_value", (14336, 5120), inv.FP8),
                    inv.tensor_spec(op + "attention/query_norm", (256,), inv.BF16),
                    inv.tensor_spec(op + "attention/key_norm", (256,), inv.BF16),
                    inv.tensor_spec(op + "attention/output", (5120, 6144), inv.FP8),
                )
            )
            fp8_weights[op + "attention/query_key_gate_value"] = nvfp4_recipe.Fp8WeightRecipe(
                op + "attention/query_key_gate_value",
                (14336, 5120),
                (nvfp4_recipe._q_part(q, False), all_rows(k), nvfp4_recipe._q_part(q, True), all_rows(v)),
            )
            fp8_weights[op + "attention/output"] = nvfp4_recipe.Fp8WeightRecipe(
                op + "attention/output", (5120, 6144), (all_rows(o),)
            )
            quant_direct[op + "attention/query_norm"] = family_recipe.TensorRecipe(
                op + "attention/query_norm",
                family_recipe.source(sp + "self_attn.q_norm.weight", (256,)),
            )
            quant_direct[op + "attention/key_norm"] = family_recipe.TensorRecipe(
                op + "attention/key_norm",
                family_recipe.source(sp + "self_attn.k_norm.weight", (256,)),
            )
        else:
            qkv = src(sp + "linear_attn.in_proj_qkv", 10240, 5120)
            z = src(sp + "linear_attn.in_proj_z", 6144, 5120)
            out = src(sp + "linear_attn.out_proj", 5120, 6144)
            gdn_fmt = inv.NVFP4 if layer in layer_plan.gdn_nvfp4_layers else inv.FP8
            specs.extend(
                (
                    inv.tensor_spec(op + "gdn/a_log", (48,), inv.FP32),
                    inv.tensor_spec(op + "gdn/dt_bias", (48,), inv.FP32),
                    inv.tensor_spec(op + "gdn/convolution", (4, 10240), inv.BF16),
                    inv.tensor_spec(op + "gdn/a_projection", (48, 5120), inv.BF16),
                    inv.tensor_spec(op + "gdn/b_projection", (48, 5120), inv.BF16),
                    inv.tensor_spec(op + "gdn/query_key_value_z", (16384, 5120), gdn_fmt),
                )
            )
            if gdn_fmt == inv.NVFP4:
                specs.append(
                    inv.tensor_spec(op + "gdn/input_projection/input_scale_divisor", (), inv.FP32)
                )
                nvfp4_weights[op + "gdn/query_key_value_z"] = nvfp4_recipe.Nvfp4WeightRecipe(
                    op + "gdn/query_key_value_z",
                    (16384, 5120),
                    (all_rows(qkv), all_rows(z)),
                    (qkv, z),
                )
                input_divisors[op + "gdn/input_projection/input_scale_divisor"] = (qkv, z)
            else:
                fp8_weights[op + "gdn/query_key_value_z"] = nvfp4_recipe.Fp8WeightRecipe(
                    op + "gdn/query_key_value_z", (16384, 5120), (all_rows(qkv), all_rows(z))
                )
            specs.extend(
                (
                    inv.tensor_spec(op + "gdn/norm", (128,), inv.BF16),
                    inv.tensor_spec(op + "gdn/output", (5120, 6144), gdn_fmt),
                )
            )
            if gdn_fmt == inv.NVFP4:
                specs.append(
                    inv.tensor_spec(op + "gdn/output_projection/input_scale_divisor", (), inv.FP32)
                )
                nvfp4_weights[op + "gdn/output"] = nvfp4_recipe.Nvfp4WeightRecipe(
                    op + "gdn/output", (5120, 6144), (all_rows(out),), (out,)
                )
                input_divisors[op + "gdn/output_projection/input_scale_divisor"] = (out,)
            else:
                fp8_weights[op + "gdn/output"] = nvfp4_recipe.Fp8WeightRecipe(
                    op + "gdn/output", (5120, 6144), (all_rows(out),)
                )
            convolution = family_recipe.source(sp + "linear_attn.conv1d.weight", (10240, 1, 4))
            quant_direct[op + "gdn/a_log"] = family_recipe.TensorRecipe(
                op + "gdn/a_log",
                family_recipe.Cast(family_recipe.source(sp + "linear_attn.A_log", (48,)), inv.FP32),
            )
            quant_direct[op + "gdn/dt_bias"] = family_recipe.TensorRecipe(
                op + "gdn/dt_bias",
                family_recipe.Cast(
                    family_recipe.source(sp + "linear_attn.dt_bias", (48,)), inv.FP32
                ),
            )
            quant_direct[op + "gdn/convolution"] = family_recipe.TensorRecipe(
                op + "gdn/convolution",
                family_recipe.Transpose(
                    family_recipe.Reshape(family_recipe.Slice(convolution, 1, 0, 1), (10240, 4)),
                    (1, 0),
                ),
            )
            quant_direct[op + "gdn/a_projection"] = family_recipe.TensorRecipe(
                op + "gdn/a_projection",
                family_recipe.source(sp + "linear_attn.in_proj_a.weight", (48, 5120)),
            )
            quant_direct[op + "gdn/b_projection"] = family_recipe.TensorRecipe(
                op + "gdn/b_projection",
                family_recipe.source(sp + "linear_attn.in_proj_b.weight", (48, 5120)),
            )
            quant_direct[op + "gdn/norm"] = family_recipe.TensorRecipe(
                op + "gdn/norm",
                family_recipe.source(sp + "linear_attn.norm.weight", (128,)),
            )

        specs.append(inv.tensor_spec(op + "post_attention_norm", (5120,), inv.BF16))
        quant_direct[op + "post_attention_norm"] = family_recipe.TensorRecipe(
            op + "post_attention_norm",
            family_recipe.source(sp + "post_attention_layernorm.weight", (5120,)),
        )

        gate = src(sp + "mlp.gate_proj", 17408, 5120)
        up = src(sp + "mlp.up_proj", 17408, 5120)
        down = src(sp + "mlp.down_proj", 5120, 17408)
        mlp_fmt = inv.NVFP4 if layer in layer_plan.mlp_nvfp4_layers else inv.FP8
        specs.append(inv.tensor_spec(op + "mlp/gate_up", (34816, 5120), mlp_fmt))
        if mlp_fmt == inv.NVFP4:
            specs.append(
                inv.tensor_spec(op + "mlp/gate_up_projection/input_scale_divisor", (), inv.FP32)
            )
            nvfp4_weights[op + "mlp/gate_up"] = nvfp4_recipe.Nvfp4WeightRecipe(
                op + "mlp/gate_up", (34816, 5120), (all_rows(gate), all_rows(up)), (gate, up)
            )
            input_divisors[op + "mlp/gate_up_projection/input_scale_divisor"] = (gate, up)
        else:
            fp8_weights[op + "mlp/gate_up"] = nvfp4_recipe.Fp8WeightRecipe(
                op + "mlp/gate_up", (34816, 5120), (all_rows(gate), all_rows(up))
            )
        specs.append(inv.tensor_spec(op + "mlp/down", (5120, 17408), mlp_fmt))
        if mlp_fmt == inv.NVFP4:
            specs.append(
                inv.tensor_spec(op + "mlp/down_projection/input_scale_divisor", (), inv.FP32)
            )
            nvfp4_weights[op + "mlp/down"] = nvfp4_recipe.Nvfp4WeightRecipe(
                op + "mlp/down", (5120, 17408), (all_rows(down),), (down,)
            )
            input_divisors[op + "mlp/down_projection/input_scale_divisor"] = (down,)
        else:
            fp8_weights[op + "mlp/down"] = nvfp4_recipe.Fp8WeightRecipe(
                op + "mlp/down", (5120, 17408), (all_rows(down),)
            )

    specs.append(inv.tensor_spec("text/final_norm", (5120,), inv.BF16))
    quant_direct["text/final_norm"] = family_recipe.TensorRecipe(
        "text/final_norm",
        family_recipe.source("model.language_model.norm.weight", (5120,)),
    )

    head = src("lm_head", 248320, 5120)
    head_fmt = inv.NVFP4 if layer_plan.output_head_nvfp4 else inv.FP8
    specs.append(inv.tensor_spec("text/output_head", (248320, 5120), head_fmt))
    if head_fmt == inv.NVFP4:
        specs.append(inv.tensor_spec("text/output_head/input_scale_divisor", (), inv.FP32))
        nvfp4_weights["text/output_head"] = nvfp4_recipe.Nvfp4WeightRecipe(
            "text/output_head", (248320, 5120), (all_rows(head),), (head,)
        )
        input_divisors["text/output_head/input_scale_divisor"] = (head,)
    else:
        fp8_weights["text/output_head"] = nvfp4_recipe.Fp8WeightRecipe(
            "text/output_head", (248320, 5120), (all_rows(head),)
        )

    # Draft head, MTP, Vision, and optional DFlash2 follow the standard Qwen3.8-27B inventory.
    specs.extend(inv._build_draft_head_specs())
    specs.extend(inv._build_mtp_specs())
    specs.extend(common_inventory.build_vision_specs(5120))
    if include_dflash2:
        specs.extend(DFLASH2_TENSOR_SPECS)

    return tuple(specs), fp8_weights, nvfp4_weights, input_divisors, quant_direct


def convert(
    official_dir: str | Path,
    quantized_dir: str | Path,
    out_path: str | Path,
    *,
    dflash2_model_dir: str | Path | None = None,
    embedding_format: str = inv.W8,
    device: str | torch.device = "cuda",
) -> Path:
    """Convert an NVIDIA ModelOpt NVFP4-QAD checkpoint into a NInfer v2 artifact."""
    started = time.perf_counter()
    official = Path(official_dir)
    quantized = Path(quantized_dir)
    output = Path(out_path)
    resolved_device = pick_device(device)

    layer_plan = parse_modelopt_quant_config(quantized)
    tensor_specs, fp8_weights, nvfp4_weights, input_divisors, quant_direct = build_qad_plan(
        layer_plan,
        embedding_format=embedding_format,
        include_dflash2=dflash2_model_dir is not None,
    )
    resources = base_convert.load_resources(official)
    resource_map = {r.name: r.data for r in resources}
    all_specs: tuple[inv.StoredObjectSpec, ...] = tuple(inv.RESOURCE_SPECS) + tensor_specs
    object_plan = family_conversion.build_object_plan(all_specs, resource_map)

    ranking = Path(__file__).resolve().parents[3] / draft_head.DEFAULT_RANKING
    draft = draft_head.compute_shortlist(ranking, official)
    draft_ids = draft_head.materialize_draft_head_token_ids(draft)
    derived = {draft_head.DRAFT_HEAD_TOKEN_IDS_OBJECT: draft_ids}
    official_recipes_by_name = {r.object_name: r for r in official_recipe.TENSOR_RECIPES}

    output.parent.mkdir(parents=True, exist_ok=True)
    dflash2_names = {s.name for s in DFLASH2_TENSOR_SPECS}
    total = len(all_specs)
    index = 0
    with ArtifactWriter(
        output,
        ArtifactIdentity(inv.MODEL_ID, inv.WEIGHTS_ID),
        object_plan.specs,
    ) as writer:
        for rspec in inv.RESOURCE_SPECS:
            index += 1
            writer.write(rspec.name, resource_map[rspec.name])
            print(f"[{index}/{total}] {rspec.name}", flush=True)

        with ShardReader(official) as official_reader, ShardReader(quantized) as quant_reader:
            dflash2_reader = (
                ShardReader.from_file(Path(dflash2_model_dir) / "model.safetensors")
                if dflash2_model_dir is not None
                else None
            )
            try:
                for spec in tensor_specs:
                    index += 1
                    payload: bytes | Iterable[bytes]
                    if spec.name == "text/token_embedding":
                        if embedding_format == inv.FP8:
                            payload = fp8_embedding.iter_reader_payload(
                                quant_reader,
                                "model.language_model.embed_tokens.weight",
                                spec.shape,
                            )
                        else:
                            emb = quant_reader.get("model.language_model.embed_tokens.weight")
                            payload = family_conversion.encode_tensor_payload(
                                emb, spec, resolved_device
                            )
                            del emb
                    elif spec.name in fp8_weights:
                        codes, scales = materialize_modelopt_fp8_weight(
                            fp8_weights[spec.name], quant_reader
                        )
                        payload = encode_fp8_row_scaled(codes, scales, spec.shape)
                    elif spec.name in nvfp4_weights:
                        packed, scales, divisor = materialize_modelopt_nvfp4_weight(
                            nvfp4_weights[spec.name], quant_reader
                        )
                        payload = encode_nvfp4(packed, scales, divisor, spec.shape)
                    elif spec.name in input_divisors:
                        scalar = materialize_modelopt_input_divisor(
                            input_divisors[spec.name], quant_reader, spec.name
                        )
                        payload = encode_direct(scalar, inv.FP32)
                    elif spec.name in quant_direct:
                        tensor = family_recipe.materialize_recipe(
                            quant_direct[spec.name], quant_reader
                        )
                        payload = encode_direct(tensor, spec.format)
                        del tensor
                    elif spec.name in dflash2_names:
                        assert dflash2_reader is not None
                        tensor = dflash2_recipe.materialize_tensor(spec.name, dflash2_reader)
                        payload = family_conversion.encode_tensor_payload(
                            tensor, spec, resolved_device
                        )
                        del tensor
                    else:
                        tensor = family_recipe.materialize_recipe(
                            official_recipes_by_name[spec.name], official_reader, dict(derived)
                        )
                        payload = family_conversion.encode_tensor_payload(
                            tensor, spec, resolved_device
                        )
                        del tensor
                    writer.write(spec.name, payload)
                    del payload
                    print(f"[{index}/{total}] {spec.name}", flush=True)
            finally:
                if dflash2_reader is not None:
                    dflash2_reader.close()

    elapsed = time.perf_counter() - started
    final_bytes = output.stat().st_size
    report_path = Path(str(output) + ".conversion.json")
    report = {
        "recipe_id": RECIPE_ID,
        "identity": {"model_id": inv.MODEL_ID, "weights_id": inv.WEIGHTS_ID},
        "gdn_nvfp4_layers": sorted(layer_plan.gdn_nvfp4_layers),
        "mlp_nvfp4_layers": sorted(layer_plan.mlp_nvfp4_layers),
        "output_head_nvfp4": layer_plan.output_head_nvfp4,
        "embedding_format": embedding_format,
        "final_bytes": final_bytes,
        "elapsed_seconds": elapsed,
    }
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(f"complete: {final_bytes} bytes in {elapsed:.1f}s; report={report_path}", flush=True)
    return report_path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path, help="Base BF16 Qwen3.8-27B dir")
    parser.add_argument(
        "--quantized-model", required=True, type=Path, help="NVIDIA ModelOpt NVFP4-QAD dir"
    )
    parser.add_argument(
        "--dflash2-model", default=None, type=Path, help="Optional DFlash2 companion dir"
    )
    parser.add_argument(
        "--embedding-format",
        choices=("w8", "fp8"),
        default="w8",
        help="Format for text/token_embedding (default: w8 = W8G32_F16S)",
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    emb_fmt = inv.W8 if args.embedding_format == "w8" else inv.FP8
    convert(
        args.model,
        args.quantized_model,
        args.out,
        dflash2_model_dir=args.dflash2_model,
        embedding_format=emb_fmt,
        device=args.device,
    )


if __name__ == "__main__":
    main()
