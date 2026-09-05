import copy
from pathlib import Path
import sys
import unittest

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
from navigation.config import load_config
from navigation.demo_bank import DemoBank
from navigation.registration import Registrar
from navigation.visual_score import calibrate, score_registration


class RegistrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cv2.setNumThreads(2)
        cls.cfg = load_config(ROOT / "config/task2.real.yaml")["navigation"]["visual"]
        cls.bank = DemoBank(cls.cfg["bank_path"], cls.cfg)

    def test_all_references_are_calibrated_across_episodes(self):
        self.assertEqual(self.bank.metadata["episodes"], 238)
        self.assertEqual(len(self.bank.metadata["samples"]), 476)
        for i, ref in enumerate(self.bank.references):
            rgb = cv2.cvtColor(ref.gray, cv2.COLOR_GRAY2RGB)
            result = self.bank.evaluate(rgb, i)
            self.assertTrue(result.ready)
            self.assertGreaterEqual(self.bank.metadata["references"][i]["statistics"]["episodes"], 8)

    def test_blank_corrupt_and_extreme_warp_rejected(self):
        with self.assertRaises(ValueError):
            self.bank.evaluate(np.full((540, 960, 3), np.nan))
        self.assertFalse(self.bank.evaluate(np.zeros((540, 960, 3), np.uint8)).ready)
        gray = self.bank.references[0].gray
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        H = cv2.getRotationMatrix2D((480, 270), 65, 1)
        self.assertFalse(self.bank.evaluate(cv2.warpAffine(rgb, H, (960, 540)), 0).ready)

    def test_geometry_score_decreases_but_photometry_survives(self):
        for i, ref in enumerate(self.bank.references):
            rgb = cv2.cvtColor(ref.gray, cv2.COLOR_GRAY2RGB)
            baseline = self.bank.evaluate(rgb, i)
            H = np.float32([[1, 0, 115], [0, 1, 0]])
            shifted = self.bank.evaluate(cv2.warpAffine(rgb, H, (960, 540)), i)
            self.assertLess(shifted.score, baseline.score - 1)
            self.assertFalse(shifted.ready)
            lit = np.clip(rgb.astype(float) * .8 + 15, 0, 255).astype(np.uint8)
            self.assertTrue(self.bank.evaluate(lit, i).ready)

    def test_self_matches_do_not_calibrate_independent_episodes(self):
        ref = self.bank.references[0]
        result = self.bank.registrar.match(ref, ref, 0)
        stats = calibrate([result] * 20, [1] * 20, self.cfg)
        self.assertFalse(stats["calibrated"])
        self.assertFalse(score_registration(result, stats, self.cfg).ready)


if __name__ == "__main__":
    unittest.main()
