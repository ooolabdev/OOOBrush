"""Read calibrated PINHOLE cameras and their COLMAP observations."""

import struct

import numpy as np


def read_pinhole_cameras(path):
    cameras = {}
    with path.open("rb") as f:
        for _ in range(struct.unpack("<Q", f.read(8))[0]):
            camera_id, model, w, h = struct.unpack("<iiQQ", f.read(24))
            if model != 1:
                raise ValueError("undistorted PINHOLE intrinsics required")
            cameras[camera_id] = (w, h, *struct.unpack("<4d", f.read(32)))
    return cameras


def read_sparse_by_view(dataset, held):
    sparse = dataset / "sparse"
    if (sparse / "0").is_dir():
        sparse = sparse / "0"
    cameras = read_pinhole_cameras(sparse / "cameras.bin")
    intrinsics = {}
    points = {}
    with (sparse / "points3D.bin").open("rb") as f:
        for _ in range(struct.unpack("<Q", f.read(8))[0]):
            idx, *xyz = struct.unpack("<Q3d", f.read(32))
            points[idx] = xyz
            f.read(11)
            f.seek(8 * struct.unpack("<Q", f.read(8))[0], 1)
    observed = {}
    with (sparse / "images.bin").open("rb") as f:
        for _ in range(struct.unpack("<Q", f.read(8))[0]):
            _, qw, qx, qy, qz, _, _, tz, camera_id = struct.unpack("<i7di", f.read(64))
            name = bytearray()
            while (b := f.read(1)) != b"\0":
                if not b:
                    raise EOFError()
                name += b
            n = struct.unpack("<Q", f.read(8))[0]
            data = f.read(n * 24)
            name = name.decode()
            if name not in held:
                continue
            intrinsics[name] = cameras[camera_id]
            row_z = [
                2 * (qx * qz - qw * qy),
                2 * (qy * qz + qw * qx),
                1 - 2 * (qx * qx + qy * qy),
            ]
            samples = [
                (x, y, np.dot(row_z, points[idx]) + tz)
                for x, y, idx in struct.iter_unpack("<ddq", data)
                if idx in points
            ]
            observed[name] = np.asarray(samples).reshape(-1, 3)
    if set(intrinsics) != set(held):
        raise ValueError("missing requested COLMAP views")
    return intrinsics, observed
