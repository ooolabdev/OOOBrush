"""Spatial scale fitting must reject saturated depths and poorly constrained frames."""

import unittest

import numpy as np

from calibrate import fit


class CalibrationTest(unittest.TestCase):
    def setUp(self):
        self.depth = np.full((256, 256), 23456, dtype=np.uint16)
        y, x = np.mgrid[8:256:16, 8:256:16]
        z = self.depth[y, x] / 65535.0 * 4
        self.points = np.stack([x, y, z], axis=-1).reshape(-1, 3)

    def test_recovers_scale_after_excluding_invalid_points(self):
        self.depth[8, 8] = 0
        self.depth[8, 24] = 65535
        self.points[2, 2] = -1
        result = fit(self.depth, self.points, (256, 256))
        self.assertEqual(result["points"], 253)
        self.assertAlmostEqual(result["scale"], 4)
        self.assertEqual(result["confidence"], 1)
        self.assertAlmostEqual(result["check_error"], 0)

    def test_missing_spatial_support_has_zero_weight(self):
        self.assertEqual(fit(self.depth, self.points[:5], (256, 256))["confidence"], 0)

    def test_disagreeing_spatial_folds_have_zero_weight(self):
        x, y = self.points[:, :2].T
        self.points[(x // 64 + y // 64) % 2 == 0, 2] *= 1.5
        self.assertEqual(fit(self.depth, self.points, (256, 256))["confidence"], 0)


if __name__ == "__main__":
    unittest.main()
