"""Calibrate relative MoGe Z in native COLMAP units using training observations only."""

from pathlib import Path

import numpy as np
from PIL import Image

from colmap import read_sparse_by_view


def fit(zmap, points, camera):
    width, height = camera[:2]
    h, w = zmap.shape
    x, y, z = points.T
    ok = (x >= 0) & (x < width) & (y >= 0) & (y < height) & (z > 0)
    x, y, z = x[ok], y[ok], z[ok]
    d = zmap[(y * h / height).astype(int), (x * w / width).astype(int)] / 65535.0
    ok = (d > 0) & (d < 1) & np.isfinite(z)
    x, y, z, d = x[ok], y[ok], z[ok], d[ok]
    tiles = (x * 4 / width).astype(int) + 4 * (y * 4 / height).astype(int)
    fold = ((x // 64 + y // 64) % 2).astype(bool)
    n, coverage = len(z), len(np.unique(tiles)) / 16
    if min(fold.sum(), (~fold).sum()) < 8:
        return dict(
            scale=1.0, confidence=0.0, points=n, coverage=coverage, check_error=None
        )
    # Score each spatial half using a scale estimated exclusively on the other half.
    ratio = z / d
    errors = np.where(
        fold,
        np.abs(np.median(ratio[~fold]) / ratio - 1),
        np.abs(np.median(ratio[fold]) / ratio - 1),
    )
    error = float(np.median(errors))
    confidence = min(1, n / 80) * coverage * float(np.clip((0.12 - error) / 0.10, 0, 1))
    return dict(
        scale=float(np.median(ratio)),
        confidence=confidence,
        points=n,
        coverage=coverage,
        check_error=error,
    )


def calibrate(dataset, names):
    intrinsics, observations = read_sparse_by_view(dataset, set(names))
    records = {}
    for name in names:
        zmap = np.asarray(
            Image.open(dataset / "depths" / Path(name).with_suffix(".png")),
            dtype=np.float64,
        )
        records[(Path("images") / name).as_posix()] = fit(
            zmap, observations[name], intrinsics[name]
        )
    return records
