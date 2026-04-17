import unittest

from energy_estimator import estimate, get_tier, savings, simple_complexity


class EnergyEstimatorTests(unittest.TestCase):
    def test_get_tier_escalates_heavy_for_high_complexity(self):
        self.assertEqual(get_tier("writing", 4.2), "heavy")

    def test_estimate_returns_expected_shape(self):
        result = estimate(2000, "light")
        self.assertEqual(result["tier"], "light")
        self.assertIn("wh", result)
        self.assertIn("co2_g", result)

    def test_savings_is_non_negative_for_typical_case(self):
        result = savings(orig_tok=1000, comp_tok=300, heavy_tier="heavy", selected_tier="light")
        self.assertGreaterEqual(result["wh_saved"], 0)
        self.assertGreaterEqual(result["pct_saved"], 0)

    def test_simple_complexity_bounds(self):
        score = simple_complexity(4000, "reasoning")
        self.assertLessEqual(score, 5.0)


if __name__ == "__main__":
    unittest.main()
