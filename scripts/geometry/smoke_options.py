"""Exercise the geometry CLI contract on a real GPU, without user datasets.

Each successful case crosses the geometry warm-up boundary and writes a PLY.
Input variants contain just the files required by their enabled losses.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image


def make_dataset(root, *, orientation=False, free_space=False, args=None):
    (root / "images").mkdir(parents=True)
    (root / "sparse").mkdir()
    camera = struct.pack("<QiiQQ4d", 1, 1, 1, 50, 50, 50.0, 50.0, 25.0, 25.0)
    (root / "sparse/cameras.bin").write_bytes(camera)
    images = bytearray(struct.pack("<Q", 3))
    yy, xx = np.mgrid[:50, :50]
    for index in range(3):
        filename = f"view-{index}.png"
        images.extend(
            struct.pack("<i7di", index + 1, 1.0, 0.0, 0.0, 0.0,
                        (index - 1) * 0.12, 0.0, 0.0, 1)
        )
        images.extend(filename.encode() + b"\0" + struct.pack("<Q", 0))
        rgb = np.stack(
            [30 + (xx * 4 + index * 8) % 210, 30 + yy * 4, 80 + (xx + yy) % 120],
            axis=-1,
        ).astype(np.uint8)
        Image.fromarray(rgb).save(root / "images" / filename)
        if orientation:
            for folder in ("depths", "normals"):
                (root / folder).mkdir(exist_ok=True)
            Image.fromarray(np.full((50, 50), 32768, dtype=np.uint16)).save(
                root / "depths" / filename
            )
            normal = np.tile(np.array([128, 128, 0], dtype=np.uint8), (50, 50, 1))
            Image.fromarray(normal).save(root / "normals" / filename)
        if free_space:
            (root / "free-space").mkdir(exist_ok=True)
            Image.fromarray(np.full((50, 50), 40000, dtype=np.uint16)).save(
                root / "free-space" / filename
            )
    (root / "sparse/images.bin").write_bytes(images)
    points = bytearray(struct.pack("<Q", 81))
    positions = ((x, y) for x in np.linspace(-0.6, 0.6, 9) for y in np.linspace(-0.6, 0.6, 9))
    for i, (x, y) in enumerate(positions):
        points.extend(
            struct.pack("<QdddBBBdQ", i + 1, float(x), float(y),
                        1.8 + 0.03 * (i % 3), 180, 120, 90, 0.01, 0)
        )
    (root / "sparse/points3D.bin").write_bytes(points)
    if free_space:
        manifest = {f"images/view-{i}.png": {"scale": 4.0, "confidence": 1.0} for i in range(3)}
        (root / "depth-calibration.json").write_text(json.dumps(manifest), encoding="utf-8")
    if args is not None:
        (root / "args.txt").write_text(args + "\n", encoding="utf-8")
    return root


def check_ply(path, expected_sh_degree=None):
    """Validate the complete, uncompressed float PLY emitted by Brush."""
    with path.open("rb") as stream:
        header = []
        while True:
            line = stream.readline()
            assert line, f"Missing end_header: {path}"
            line = line.rstrip(b"\r\n")
            if line == b"end_header":
                break
            header.append(line)
        payload = stream.read()
    assert header[0] == b"ply"
    assert b"format binary_little_endian 1.0" in header
    if expected_sh_degree is not None:
        assert f"comment SH degree: {expected_sh_degree}".encode() in header
    elements = [line for line in header if line.startswith(b"element ")]
    assert len(elements) == 1 and elements[0].startswith(b"element vertex ")
    count = int(elements[0].split()[2])
    properties = [line for line in header if line.startswith(b"property ")]
    assert properties and all(line.startswith(b"property float ") for line in properties)
    assert 0 < count <= 256
    assert len(payload) == count * len(properties) * 4, f"Incorrect PLY payload size: {path}"
    assert np.isfinite(np.frombuffer(payload, dtype="<f4")).all(), f"Non-finite splat: {path}"
    return count


def run_case(brush, output, env, name, dataset, args, expected, success=True, iterations=510):
    result = output / name
    result.mkdir()
    command = [
        str(brush), str(dataset), "--total-train-iters", str(iterations),
        "--max-resolution", "50", "--max-splats", "256", "--refine-every", "1000",
        "--growth-stop-iter", "0", "--seed", "42", "--export-every", str(iterations),
        "--export-path", str(result), "--export-name", "final.ply", *args,
    ]
    started = perf_counter()
    log_path = result / "brush.log"
    if name == "merged-overrides":
        # Preserve compatibility with consumers filtering the original log target.
        env = dict(env, RUST_LOG="brush_cli=info,brush_process::train_stream=info")
    with log_path.open("wb") as stream:
        child = subprocess.run(command, cwd=output, env=env, stdout=stream, stderr=subprocess.STDOUT)
    log = log_path.read_text(encoding="utf-8", errors="replace")
    assert expected in log, f"{name}: expected {expected!r}; see {log_path}"
    record = dict(name=name, elapsed=perf_counter() - started,
                  exit_code=child.returncode, command=command)
    if success:
        assert child.returncode == 0, f"{name}: {log[-3500:]}"
        assert "Loaded dataset with 3 training, 0 eval views" in log
        assert f"iteration={iterations} total={iterations}" in log
        assert "Export succeeded" in log and "Done training!" in log
        if name != "rgb":
            assert "calibrated_depth=0 relative_depth=0 warmup=500 ramp=500" in log
        record.update(success=True, splats=check_ply(
            result / "final.ply", expected_sh_degree=1 if name == "merged-overrides" else None
        ))
    else:
        assert child.returncode != 0 and "Done training!" not in log, log[-3500:]
        assert "Export succeeded" not in log and not (result / "final.ply").exists()
        record.update(success=False, expected_error=True)
    print(f"{name}: PASS ({record['elapsed']:.2f}s)", flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brush", required=True, type=Path, help="Path to the native brush-cli binary")
    parser.add_argument("--output", required=True, type=Path, help="A new directory for fixtures and results")
    parser.add_argument("--backend", help="Optional WGPU_BACKEND override (e.g. vulkan)")
    options = parser.parse_args()
    brush = options.brush.resolve(strict=True)
    if not brush.is_file():
        parser.error("--brush must point to a file")
    output = options.output.resolve()
    # Never overwrite previous regression results or a real training dataset.
    output.mkdir(parents=True, exist_ok=False)
    scratch = output / "tmp"
    scratch.mkdir()
    env = dict(os.environ, TEMP=str(scratch), TMP=str(scratch), TMPDIR=str(scratch),
               RUST_LOG="info", RUST_BACKTRACE="1")
    if options.backend:
        env["WGPU_BACKEND"] = options.backend
    data = output / "datasets"
    results = output / "results"
    results.mkdir()
    rgb = make_dataset(data / "rgb")
    orientation = make_dataset(data / "orientation", orientation=True)
    free = make_dataset(data / "free-space", free_space=True)
    full = make_dataset(data / "full", orientation=True, free_space=True)
    merged = make_dataset(
        data / "merged",
        args="--geometry-supervision on --geometry-orientation true --geometry-free-space true "
             "--geometry-depth-distortion false --geometry-orientation-weight 0.025 --sh-degree 1",
    )
    cases = [
        ("rgb", rgb, [], "Geometry supervision off; using RGB training"),
        ("orientation-only", orientation,
         ["--geometry-orientation", "true", "--geometry-orientation-weight", "0.025"],
         "Geometry supervision: orientation=0.025 free_space=0 depth_distortion=0"),
        ("free-space-only", free,
         ["--geometry-supervision", "on", "--geometry-orientation", "false",
          "--geometry-depth-distortion", "false", "--geometry-free-space-weight", "0.04"],
         "Geometry supervision: orientation=0 free_space=0.04 depth_distortion=0"),
        ("distortion-only", rgb,
         ["--geometry-depth-distortion", "true", "--geometry-depth-distortion-weight", "0.005"],
         "Geometry supervision: orientation=0 free_space=0 depth_distortion=0.005"),
        ("prepared-defaults", full, [],
         "Geometry supervision: orientation=0.05 free_space=0.02 depth_distortion=0.01"),
        ("custom-combination", full,
         ["--geometry-supervision", "on", "--geometry-orientation-weight", "0.025",
          "--geometry-depth-distortion-weight", "0.005"],
         "Geometry supervision: orientation=0.025 free_space=0.02 depth_distortion=0.005"),
        ("merged-overrides", merged,
         ["--geometry-orientation", "false", "--geometry-free-space", "false",
          "--geometry-depth-distortion", "true", "--geometry-depth-distortion-weight", "0.005"],
         "Geometry supervision: orientation=0 free_space=0 depth_distortion=0.005"),
        ("missing-orientation-input", rgb, ["--geometry-orientation", "true"],
         "missing geometry map", False, 1),
    ]
    summary = [run_case(brush, results, env, *case) for case in cases]
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
