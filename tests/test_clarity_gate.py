import unittest
from unittest.mock import patch

from clarity_gate import assess, detect_domain, enrich


class _DummyModel:
    def predict_proba(self, _: list[str]):
        return [[0.1, 0.9]]


class ClarityGateTests(unittest.TestCase):
    def test_detect_domain(self):
        self.assertEqual(detect_domain("please debug this python function"), "code")
        self.assertEqual(detect_domain("write a short email"), "writing")
        self.assertEqual(detect_domain("help me"), "general")

    @patch("clarity_gate._load", return_value=_DummyModel())
    def test_assess_ambiguous_returns_questions(self, _):
        result = assess("fix it", threshold=0.5)
        self.assertTrue(result.is_ambiguous)
        self.assertEqual(len(result.questions), 2)

    def test_enrich_appends_selected_answers(self):
        text = enrich("fix it", {"code_goal": ["Debug / fix a bug"], "output_length": []})
        self.assertIn("Context", text)
        self.assertIn("Debug / fix a bug", text)


if __name__ == "__main__":
    unittest.main()
