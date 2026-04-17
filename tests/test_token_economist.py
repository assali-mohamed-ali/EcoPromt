import unittest
from unittest.mock import patch

from token_economist import detect_costly_patterns, detect_languages, remove_fillers


class _DummyToken:
    def __init__(self, lemma_, pos_, is_stop=False, text="word"):
        self.lemma_ = lemma_
        self.pos_ = pos_
        self.is_stop = is_stop
        self.text = text


class _DummyNLP:
    def __call__(self, text):
        class _Doc(list):
            @property
            def sents(self):
                return ["This sentence is repeated often.", "This sentence is repeated often."]

        return _Doc(
            [
                _DummyToken("sentence", "NOUN"),
                _DummyToken("repeat", "VERB"),
                _DummyToken("repeat", "VERB"),
            ]
        )


class TokenEconomistTests(unittest.TestCase):
    def test_remove_fillers(self):
        compressed, removed = remove_fillers("please kindly fix this bug")
        self.assertIn("fix this bug", compressed)
        self.assertTrue(removed)

    def test_detect_languages_non_latin_warning(self):
        result = detect_languages("please explain هذا")
        self.assertTrue(result["token_overhead_warning"])

    def test_detect_costly_patterns(self):
        result = detect_costly_patterns("LOOK HERE!!!")
        names = {item["name"] for item in result}
        self.assertIn("Excessive punctuation", names)
        self.assertIn("All-caps words", names)


if __name__ == "__main__":
    unittest.main()
