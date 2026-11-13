"""Optional live generate smoke (needs Apple Silicon + model download).

Run:
  OZEK_LIVE=1 python -m unittest tests.test_live -v
"""

from __future__ import annotations

import os
import unittest

from ozek.profiles import resolve_model
from ozek.runtime import OzekMLXRuntime

LIVE = os.environ.get("OZEK_LIVE", "").strip() in ("1", "true", "yes")


@unittest.skipUnless(LIVE, "set OZEK_LIVE=1 to run live generate smoke")
class TestLiveGenerate(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rt = OzekMLXRuntime()
        cls.rt.load(resolve_model())

    def test_short_generate(self) -> None:
        r = self.rt.generate("Say hi in one word.", max_tokens=8, temperature=0.0)
        self.assertTrue(r.text.strip())
        self.assertIn(r.stats.finish_reason, ("length", "stop", "eos"))

    def test_json_mode(self) -> None:
        r = self.rt.generate(
            "Return city and country for Tokyo as JSON.",
            max_tokens=64,
            temperature=0.0,
            response_format="json",
            schema={
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "country": {"type": "string"},
                },
            },
            json_retries=1,
        )
        self.assertEqual(r.response_format, "json")
        self.assertIsNone(r.parse_error, r.text)
        self.assertIsInstance(r.parsed, dict)

    def test_timeout_finish_reason(self) -> None:
        r = self.rt.generate(
            "Write a very long essay about rivers.",
            max_tokens=256,
            temperature=0.8,
            timeout_s=0.05,
        )
        self.assertEqual(r.stats.finish_reason, "timeout")


if __name__ == "__main__":
    unittest.main()
