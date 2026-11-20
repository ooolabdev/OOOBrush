"""Build Brush geometry targets from undistorted COLMAP images and MoGe-2 camera-Z/normal maps."""

import argparse
import json
from pathlib import Path, PureWindowsPath
from tempfile import TemporaryDirectory

from calibrate import calibrate
from colmap import read_pinhole_cameras
from free_space_masks import poses, write_masks


def prepare(dataset, eval_split_every=None):
    sparse = dataset / "sparse"
    if (sparse / "0").is_dir():
        sparse = sparse / "0"
    cameras = read_pinhole_cameras(sparse / "cameras.bin")
    views = poses(sparse / "images.bin")
    names = sorted(views)
    if eval_split_every is not None:
        if eval_split_every < 2:
            raise ValueError("eval split must leave training views")
        names = [name for i, name in enumerate(names) if i % eval_split_every]
    if not names:
        raise ValueError("empty training set")
    outputs = set()
    for name in names:
        path = PureWindowsPath(name)
        if path.anchor or ".." in path.parts:
            raise ValueError(f"expected a relative COLMAP image path: {name}")
        png = Path(name).with_suffix(".png")
        if png in outputs:
            raise ValueError(f"duplicate geometry sidecar: {png}")
        outputs.add(png)
        for folder, file in (("images", name), ("normals", png), ("depths", png)):
            if not (dataset / folder / file).is_file():
                raise FileNotFoundError(dataset / folder / file)
    manifest = dataset / "depth-calibration.json"
    masks = dataset / "free-space"
    if manifest.exists() or masks.exists():
        raise FileExistsError(
            "prepare a fresh dataset copy without previous calibration or free-space masks"
        )
    calibration = calibrate(dataset, names)
    with TemporaryDirectory(dir=dataset.parent) as tmp:
        stage = Path(tmp)
        counts = write_masks(
            dataset, stage / "free-space", names, cameras, views, calibration
        )
        staged_manifest = stage / manifest.name
        staged_manifest.write_text(
            json.dumps(calibration, indent=2, allow_nan=False), encoding="utf8"
        )
        (stage / "free-space").rename(masks)
        staged_manifest.rename(manifest)
    print(
        f"{len(names)} training views, {sum(counts.values())} confirmed free-space samples"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument(
        "--eval-split-every", type=int, help="Match the Brush COLMAP evaluation split."
    )
    args = parser.parse_args()
    prepare(args.dataset, args.eval_split_every)
