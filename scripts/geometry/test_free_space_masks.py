"""Multi-view target sanity check on a pinhole camera track viewing a flat wall."""

import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from colmap import read_pinhole_cameras
from free_space_masks import poses, write_masks


class FreeSpaceMasksTest(unittest.TestCase):
    def test_consensus_veto_and_holdout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            sparse = dataset / "sparse"
            sparse.mkdir(parents=True)
            (dataset / "depths").mkdir()
            w, h = 64, 48
            (sparse / "cameras.bin").write_bytes(
                struct.pack("<QiiQQdddd", 1, 1, 1, w, h, 80, 80, 32, 24)
            )
            names = [f"{i:03d}.jpg" for i in range(31)]
            held = "held.jpg"
            with (sparse / "images.bin").open("wb") as f:
                f.write(struct.pack("<Q", len(names) + 1))
                for i, name in enumerate([*names, held]):
                    f.write(
                        struct.pack("<i7di", i + 1, 1, 0, 0, 0, -i * 0.025, 0, 0, 1)
                    )
                    f.write(name.encode() + b"\0" + struct.pack("<Q", 0))
            cal = {"images/" + n: dict(scale=4.0, confidence=1.0) for n in names}
            for i, n in enumerate(names):
                z = np.full((h, w), 32768, dtype=np.uint16)
                if i == 15:
                    z[:8] = 0
                if i == 20:
                    # Near occluder in the +5 neighbour, left of its optical centre.
                    z[:, 8:28] = 16384
                Image.fromarray(z).save(
                    dataset / "depths" / Path(n).with_suffix(".png")
                )
            # No held-out depth map exists. Preparing the training maps still succeeds.
            write_masks(
                dataset,
                dataset / "free-space",
                names,
                read_pinhole_cameras(sparse / "cameras.bin"),
                poses(sparse / "images.bin"),
                cal,
            )
            with Image.open(dataset / "free-space/015.png") as image:
                mask = np.asarray(image)
            self.assertEqual(mask[6, 10], 32768)  # Consistent wall at x=40.
            self.assertEqual(mask[6, 6], 0)  # Conflict at x=24.
            self.assertEqual(mask[0, 10], 0)  # Invalid source depth.
            self.assertFalse((dataset / "free-space/held.png").exists())


if __name__ == "__main__":
    unittest.main()
