"""Offline unit smoke for Ozek contract / structured / config / stop helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ozek.config import (
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
from ozek.runtime import BusyError, OzekMLXRuntime, _apply_stop
from ozek.structured import extract_json, try_extract_json


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


if __name__ == "__main__":
    unittest.main()
