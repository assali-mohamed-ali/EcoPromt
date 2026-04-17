import unittest

from token_economist import detect_costly_patterns, detect_languages, remove_fillers


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
