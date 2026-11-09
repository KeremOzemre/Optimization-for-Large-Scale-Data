"""Check the probe design without tokenization, downloads, or model inference."""

from collections import Counter
import unittest

from context_checks import PROTOCOL_VERSION, prediction_summary, probes


class ContextProbeTests(unittest.TestCase):
    def setUp(self):
        self.probes = list(probes())

    def test_all_factor_combinations_are_present_in_each_scenario(self):
        self.assertEqual(len(self.probes), 80)
        counts = Counter(probe["scenario"] for probe in self.probes)
        self.assertEqual(len(counts), 10)
        self.assertEqual(set(counts.values()), {8})
        for scenario in counts:
            variants = {(probe["color_assignment"], probe["mention_order"], probe["queried_entity"])
                        for probe in self.probes if probe["scenario"] == scenario}
            self.assertEqual(len(variants), 8)
        self.assertEqual(PROTOCOL_VERSION, "color_tracking_v2_balanced")

    def test_answer_matches_queried_entity_and_the_written_prompt(self):
        for probe in self.probes:
            queried = probe["queried_entity"]
            self.assertEqual(probe["queried_object"], probe[f"object_{queried}"])
            self.assertEqual(probe["correct"], probe[f"color_{queried}"])
            first, second = probe["mention_order"].split("_")
            fact = (f"a {probe[f'color_{first}']} {probe[f'object_{first}']} and a "
                    f"{probe[f'color_{second}']} {probe[f'object_{second}']}.")
            self.assertIn(fact, probe["prompt"])
            self.assertTrue(probe["prompt"].endswith(f"color of the {probe['queried_object']} was"))
            self.assertEqual(probe["query_position"], 1 if queried == first else 2)

    def test_simple_color_and_position_shortcuts_only_score_half(self):
        rules = {
            "constant_red": lambda probe: "red",
            "constant_blue": lambda probe: "blue",
            "first_color": lambda probe: probe[f"color_{probe['mention_order'][0]}"],
            "last_color": lambda probe: probe[f"color_{probe['mention_order'][-1]}"],
            "query_position": lambda probe: "red" if probe["query_position"] == 1 else "blue",
        }
        for name, rule in rules.items():
            with self.subTest(rule=name):
                self.assertEqual(sum(rule(probe) == probe["correct"] for probe in self.probes), 40)
                # The balance also holds within every scenario, not just overall.
                for scenario in range(10):
                    selected = [probe for probe in self.probes if probe["scenario"] == scenario]
                    self.assertEqual(sum(rule(probe) == probe["correct"] for probe in selected), 4)

    def test_strict_score_requires_all_eight_variants(self):
        rows = [{**probe, "correct_prediction": 1} for probe in self.probes]
        summary = prediction_summary(rows)
        self.assertEqual(summary, {"accuracy": 1.0, "prompts": 80, "scenarios": 10,
                                   "scenarios_all_variants_correct": 10})
        rows[0]["correct_prediction"] = 0
        self.assertEqual(prediction_summary(rows)["scenarios_all_variants_correct"], 9)
        rows.pop(0)
        self.assertEqual(prediction_summary(rows)["scenarios_all_variants_correct"], 9)


if __name__ == "__main__":
    unittest.main()
