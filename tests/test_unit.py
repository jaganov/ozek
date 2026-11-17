"""Offline unit smoke for Ozek contract / structured / config / stop helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ozek.config import (
    bundled_example_path,
    load_config,
    merge_preset_with_overrides,
    preset_to_generate_kwargs,
)
from ozek.contract import (
    FINISH_TIMEOUT,
    FINISH_REASONS,
    contract_meta,
    normalize_generate_kwargs,
    openai_finish_reason,
)
from ozek.budget import ContextBudget, ContextBudgetError
from ozek.contract import FINISH_FINAL, openai_finish_reason
from ozek.harmony import (
    final_channel_closed,
    normalize_reasoning_effort,
    parse_harmony,
)
from ozek.profiles import get_profile
from ozek.runtime import BusyError, GenerateResult, GenerateStats, OzekMLXRuntime, _apply_stop
from ozek.structured import extract_json, try_extract_json, validate_against_schema


class TestStop(unittest.TestCase):
    def test_earliest_stop(self) -> None:
        text, hit = _apply_stop("abcSTOP more", ["STOP", "x"])
        self.assertTrue(hit)
        self.assertEqual(text, "abc")

    def test_no_stop(self) -> None:
        text, hit = _apply_stop("hello", ["\n"])
        self.assertFalse(hit)
        self.assertEqual(text, "hello")


class TestStructured(unittest.TestCase):
    def test_extract_object(self) -> None:
        self.assertEqual(extract_json('{"a": 1}'), {"a": 1})

    def test_extract_fenced(self) -> None:
        self.assertEqual(
            extract_json('Sure.\n```json\n{"ok": true}\n```\n'),
            {"ok": True},
        )

    def test_try_fail(self) -> None:
        parsed, err = try_extract_json("not json")
        self.assertIsNone(parsed)
        self.assertIsNotNone(err)

    def test_schema_required_and_types(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "n": {"type": "integer"},
            },
            "required": ["city", "n"],
        }
        self.assertIsNone(validate_against_schema({"city": "Tokyo", "n": 1}, schema))
        self.assertIn("missing", validate_against_schema({"city": "Tokyo"}, schema) or "")
        self.assertIn("type", validate_against_schema({"city": "Tokyo", "n": "x"}, schema) or "")


class TestHarmony(unittest.TestCase):
    SAMPLE = (
        "<|channel|>analysis<|message|>thinking here<|end|>"
        "<|start|>assistant<|channel|>final<|message|>東京"
    )
    SAMPLE_CLOSED = SAMPLE + "<|return|>"

    def test_parse_channels(self) -> None:
        parts = parse_harmony(self.SAMPLE)
        self.assertTrue(parts.has_channels)
        self.assertEqual(parts.thinking, "thinking here")
        self.assertEqual(parts.final, "東京")

    def test_plain_text(self) -> None:
        parts = parse_harmony("just hello")
        self.assertFalse(parts.has_channels)
        self.assertIsNone(parts.final)

    def test_final_channel_closed(self) -> None:
        ok, cut = final_channel_closed(self.SAMPLE)
        self.assertFalse(ok)
        ok, cut = final_channel_closed(self.SAMPLE_CLOSED)
        self.assertTrue(ok)
        self.assertTrue(cut.endswith("<|return|>"))
        self.assertIn("東京", cut)

    def test_reasoning_effort_aliases(self) -> None:
        self.assertEqual(normalize_reasoning_effort("med"), "medium")
        self.assertEqual(normalize_reasoning_effort("off"), "off")
        self.assertEqual(normalize_reasoning_effort("direct"), "off")
        with self.assertRaises(ValueError):
            normalize_reasoning_effort("ultra")

    def test_result_json_fields(self) -> None:
        stats = GenerateStats(
            model="x",
            prompt_tokens=1,
            generation_tokens=2,
            ttft_s=0.1,
            wall_s=0.2,
            prompt_tps=1.0,
            generation_tps=2.0,
            decode_tps_wall=2.0,
            peak_memory_gb=1.0,
            enable_thinking=False,
        )
        parts = parse_harmony(self.SAMPLE)
        r = GenerateResult(
            text=self.SAMPLE,
            stats=stats,
            thinking=parts.thinking,
            final=parts.final,
            final_only=True,
        )
        d = r.as_dict()
        self.assertEqual(d["text"], "東京")
        self.assertEqual(d["final"], "東京")
        self.assertEqual(d["thinking"], "thinking here")
        self.assertEqual(d["raw"], self.SAMPLE)

    def test_openai_maps_final(self) -> None:
        self.assertEqual(openai_finish_reason(FINISH_FINAL), "stop")


class TestContract(unittest.TestCase):
    def test_finish_reasons_include_timeout(self) -> None:
        self.assertIn(FINISH_TIMEOUT, FINISH_REASONS)
        self.assertIn("timeout", contract_meta()["finish_reasons"])

    def test_openai_map(self) -> None:
        self.assertEqual(openai_finish_reason("cancel"), "stop")
        self.assertEqual(openai_finish_reason("timeout"), "length")
        self.assertEqual(openai_finish_reason("eos"), "stop")

    def test_normalize_reliability_keys(self) -> None:
        kw = normalize_generate_kwargs(
            {
                "timeout_s": 12.5,
                "seed": 7,
                "json_retries": 2,
                "response_format": {"type": "json_object"},
            }
        )
        self.assertEqual(kw["timeout_s"], 12.5)
        self.assertEqual(kw["seed"], 7)
        self.assertEqual(kw["json_retries"], 2)
        self.assertEqual(kw["response_format"], "json")


class TestConfig(unittest.TestCase):
    def test_bundled_example_exists(self) -> None:
        p = bundled_example_path()
        self.assertTrue(p.is_file(), f"missing bundled example: {p}")
        cfg = load_config(p)
        self.assertEqual(cfg.default_profile, "qwen25-3b-4bit")
        self.assertIn("summarize", cfg.presets)

    def test_load_yaml_preset(self) -> None:
        raw = """
default_profile: qwen25-3b-4bit
presets:
  demo:
    max_tokens: 32
    temperature: 0.1
    timeout_s: 5
    seed: 1
    json_retries: 1
    response_format: json
"""
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "ozek.yml"
            path.write_text(raw, encoding="utf-8")
            cfg = load_config(path)
            preset = cfg.get_preset("demo")
            merged = merge_preset_with_overrides(preset, overrides={"max_tokens": 16})
            kw = preset_to_generate_kwargs(merged)
            self.assertEqual(kw["max_tokens"], 16)
            self.assertEqual(kw["timeout_s"], 5.0)
            self.assertEqual(kw["seed"], 1)
            self.assertEqual(kw["json_retries"], 1)
            self.assertEqual(kw["response_format"], "json")


class TestBusy(unittest.TestCase):
    def test_busy_without_model(self) -> None:
        rt = OzekMLXRuntime()
        # Force-hold the slot to simulate in-flight work.
        self.assertTrue(rt._slot.acquire(blocking=False))
        try:
            with self.assertRaises(BusyError):
                rt._acquire_slot()
        finally:
            rt._slot.release()


class TestBudget(unittest.TestCase):
    def test_ok(self) -> None:
        ContextBudget(4096).check(100, 128)

    def test_exceeds(self) -> None:
        with self.assertRaises(ContextBudgetError) as ctx:
            ContextBudget(200).check(150, 100)
        self.assertEqual(ctx.exception.code, "context_budget")
        self.assertIn("max_context_tokens", ctx.exception.as_dict())


class TestProfileAliases(unittest.TestCase):
    def test_ab_aliases(self) -> None:
        self.assertEqual(get_profile("4bit").id, "llmjp41-8b-thinking-4bit")
        self.assertEqual(get_profile("8bit").id, "llmjp41-8b-thinking-8bit")
        self.assertEqual(get_profile("llmjp-4bit").quant, "4bit")
        self.assertTrue(
            get_profile("4bit").model.startswith("andeniel/")
            or "/" in get_profile("4bit").model
        )
        self.assertNotIn("/Users/", get_profile("4bit").model)
        self.assertNotIn("/Users/", get_profile("8bit").model)


if __name__ == "__main__":
    unittest.main()
