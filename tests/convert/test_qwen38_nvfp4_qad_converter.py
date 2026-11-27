"""Unit tests for the NVIDIA ModelOpt NVFP4-QAD converter recipe."""

from __future__ import annotations

import json
from pathlib import Path
import struct
import tempfile

import torch

from tools.convert.qwen3_8_27b import convert_nvfp4_qad as qad
from tools.convert.qwen3_8_27b import inventory_nvfp4 as inv


def test_parse_modelopt_config_and_plan() -> None:
    gdn_nvfp4_expected = {
        layer
        for layer in range(6, 57)
        if layer not in inv.FULL_ATTENTION_LAYERS
    }
    assert len(gdn_nvfp4_expected) == 38
    quantized_layers: dict[str, dict[str, object]] = {
        "lm_head": {"quant_algo": "NVFP4", "group_size": 16}
    }
    for layer in range(64):
        prefix = f"model.language_model.layers.{layer}."
        if layer in inv.FULL_ATTENTION_LAYERS:
            for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
                quantized_layers[f"{prefix}self_attn.{proj}"] = {"quant_algo": "FP8"}
        else:
            algo = "NVFP4" if layer in gdn_nvfp4_expected else "FP8"
            entry: dict[str, object] = {"quant_algo": algo}
            if algo == "NVFP4":
                entry["group_size"] = 16
            for proj in ("in_proj_qkv", "in_proj_z", "out_proj"):
                quantized_layers[f"{prefix}linear_attn.{proj}"] = dict(entry)
        for proj in ("gate_proj", "up_proj", "down_proj"):
            quantized_layers[f"{prefix}mlp.{proj}"] = {
                "quant_algo": "NVFP4",
                "group_size": 16,
            }

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "hf_quant_config.json").write_text(
            json.dumps(
                {
                    "producer": {"name": "modelopt", "version": "0.46.0rc1"},
                    "quantization": {
                        "quant_algo": "MIXED_PRECISION",
                        "quantized_layers": quantized_layers,
                    },
                }
            ),
            encoding="utf-8",
        )
        plan = qad.parse_modelopt_quant_config(tmp_path)
        assert plan.gdn_nvfp4_layers == frozenset(gdn_nvfp4_expected)
        assert plan.mlp_nvfp4_layers == frozenset(range(64))
        assert plan.output_head_nvfp4 is True

        specs, fp8_w, nvfp4_w, divs, _ = qad.build_qad_plan(
            plan, embedding_format=inv.W8, include_dflash2=False
        )
        by_name = {s.name: s for s in specs}
        assert by_name["text/token_embedding"].format == inv.W8
        assert by_name["text/output_head"].format == inv.NVFP4
        assert "text/output_head/input_scale_divisor" in divs
        assert by_name["text/layers/0/gdn/query_key_value_z"].format == inv.FP8
        assert by_name["text/layers/6/gdn/query_key_value_z"].format == inv.NVFP4
        assert by_name["text/layers/6/gdn/output"].format == inv.NVFP4
        assert by_name["text/layers/57/gdn/query_key_value_z"].format == inv.FP8
        assert by_name["text/layers/63/mlp/down"].format == inv.NVFP4
        assert len(nvfp4_w) == 1 + 64 * 2 + 38 * 2
        assert len(fp8_w) == 16 * 2 + 10 * 2


def test_scale_inversion_and_fusion_rescaling() -> None:
    # Matches WarlaxZ/Qwen3.8-27B-NVFP4-QAD-NInfer-V100 lm_head weight_scale_2 exactly.
    word = qad.scale_to_divisor_word(0.00012715657, "lm_head")
    val = struct.unpack("<f", struct.pack("<I", word))[0]
    assert abs(val - 7864.3203) < 0.01

    p0 = torch.zeros((128, 32), dtype=torch.uint8)
    p1 = torch.ones((128, 32), dtype=torch.uint8)
    s0 = torch.full((128, 4), 2.0, dtype=torch.float32).to(torch.float8_e4m3fn)
    s1 = torch.full((128, 4), 4.0, dtype=torch.float32).to(torch.float8_e4m3fn)
    fused_p, fused_s, div_bytes = qad.fuse_modelopt_nvfp4_parts(
        [(p0, s0, 0.0002), (p1, s1, 0.0001)], "test_fuse"
    )
    assert fused_p.shape == (256, 32)
    assert fused_s.shape == (256, 4)
    # Second half's scales should be halved (0.0001 / 0.0002 = 0.5 -> 4.0 * 0.5 = 2.0).
    s1_out = fused_s[128:].view(torch.float8_e4m3fn).to(torch.float32)
    assert torch.allclose(s1_out, torch.full_like(s1_out, 2.0))
    div_val = struct.unpack("<f", div_bytes)[0]
    assert abs(div_val - 5000.0) < 0.01


if __name__ == "__main__":
    test_parse_modelopt_config_and_plan()
    test_scale_inversion_and_fusion_rescaling()
    print("ok")
