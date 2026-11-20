"""Build conservative MoGe camera-Z targets from neighbouring training cameras."""

from functools import lru_cache
from pathlib import Path
import struct

import numpy as np
from PIL import Image
from scipy.ndimage import maximum_filter, minimum_filter
from scipy.spatial.transform import Rotation


def poses(path):
    result = {}
    with path.open("rb") as f:
        for _ in range(struct.unpack("<Q", f.read(8))[0]):
            _, qw, qx, qy, qz, *tail = struct.unpack("<i7di", f.read(64))
            name = bytearray()
            while (b := f.read(1)) != b"\0":
                if not b:
                    raise EOFError()
                name += b
            f.seek(24 * struct.unpack("<Q", f.read(8))[0], 1)
            name = name.decode()
            rot = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            result[name] = (rot, -rot.T @ np.array(tail[:3]), tail[3])
    return result


def write_masks(dataset, out, names, cameras, cams, cal):
    # Mask pixels are the same RGB pixel centres sampled by the training loss.
    @lru_cache(None)
    def grid(camera_id):
        W, H, fx, fy, cx, cy = cameras[camera_id]
        y, x = np.mgrid[0:H:4, 0:W:4] + 0.5
        rays = np.stack([(x - cx) / fx, (y - cy) / fy, np.ones_like(x)], axis=-1)
        return x, y, rays

    @lru_cache(40)
    def load(name):
        camera_id = cams[name][2]
        x, y, _ = grid(camera_id)
        W, H = cameras[camera_id][:2]
        rec = cal["images/" + name]
        with Image.open(dataset / "depths" / Path(name).with_suffix(".png")) as im:
            src = np.asarray(im)
        h, w = src.shape
        raw = src[(y * h / H).astype(int), (x * w / W).astype(int)].astype(np.uint16)
        low = minimum_filter(raw, size=3)
        high = maximum_filter(raw, size=3)
        ok = (low > 0) & (high < 65535) & (high <= low * 1.05)
        ok &= rec["confidence"] >= 0.4
        return raw, raw * (rec["scale"] / 65535.0), ok

    out.mkdir(exist_ok=True)
    counts = {}
    for i, name in enumerate(names):
        raw, z, valid = load(name)
        votes = np.zeros_like(raw, dtype=np.int32)
        conflict = np.zeros_like(valid)
        R, C, camera_id = cams[name]
        x, y, rays = grid(camera_id)
        _, _, fx, fy, cx, cy = cameras[camera_id]
        ray_world = rays @ R
        world = ray_world * z[..., None] + C
        for j, other in enumerate(names):
            if j == i or not valid.any():
                continue
            R2, C2, other_id = cams[other]
            _, _, fx2, fy2, cx2, cy2 = cameras[other_id]
            vec = world - C2
            xyz = vec @ R2.T
            zz = xyz[..., 2]
            uv = xyz[..., :2] / np.maximum(zz[..., None], 1e-9)
            u, v = uv[..., 0] * fx2 + cx2, uv[..., 1] * fy2 + cy2
            # Nearest supervised pixel centre in the neighbour.
            _, other_z, other_valid = load(other)
            h, w = other_z.shape
            xx = np.rint(np.clip((u - 0.5) / 4, -1, w)).astype(int)
            yy = np.rint(np.clip((v - 0.5) / 4, -1, h)).astype(int)
            inside = (zz > 0) & (xx >= 0) & (yy >= 0) & (xx < w) & (yy < h)
            xx, yy = np.clip(xx, 0, w - 1), np.clip(yy, 0, h - 1)
            observed = other_z[yy, xx]
            cos = np.sum(ray_world * vec, axis=-1) / np.maximum(
                np.linalg.norm(ray_world, axis=-1) * np.linalg.norm(vec, axis=-1), 1e-9
            )
            usable = (
                inside
                & other_valid[yy, xx]
                & (cos < np.cos(np.deg2rad(1.5)))
                & (cos > np.cos(np.deg2rad(20)))
            )
            error = np.abs(observed - zz) / np.maximum(zz, 1e-9)
            # Backproject measured Z along the projected ray. Check source-pixel displacement.
            back = (
                (vec * (observed / np.maximum(zz, 1e-9))[..., None]) + (C2 - C)
            ) @ R.T
            back_uv = back[..., :2] / np.maximum(back[..., 2:], 1e-9)
            reproj = np.hypot(
                back_uv[..., 0] * fx + cx - x, back_uv[..., 1] * fy + cy - y
            )
            votes += usable & (error <= 0.05) & (back[..., 2] > 0) & (reproj <= 2.5)
            conflict |= usable & (error > 0.08)
        accepted = valid & (votes >= 2) & ~conflict
        dest = out / Path(name).with_suffix(".png")
        dest.parent.mkdir(exist_ok=True, parents=True)
        Image.fromarray(np.where(accepted, raw, 0).astype(np.uint16)).save(dest)
        counts[name] = int(accepted.sum())
        if (i + 1) % 100 == 0:
            print(f"{i+1}/{len(names)}", flush=True)
    return counts
