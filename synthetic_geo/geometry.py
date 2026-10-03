"""Camera geometry for the VEO cube-face images (cameras.json).

Convention (verified: it reproduces four independently found label pixels within 29 px, and is the only one
of 96 candidate axis conventions that does):

    camera_point = diag(1, -1, -1) @ R.T @ (world_point - camera_position)
    u = f * x / z + cx,   v = f * y / z + cy          (z > 0 = in front of the camera)

R is the rotation matrix of cameras.json "rotation_wxyz"; world = the E57 file frame (same frame as the
cabinet positions in cabinet_registry.json).
"""
import json
from pathlib import Path

import numpy as np

FLIP = np.diag([1.0, -1.0, -1.0])
ROOT = Path(__file__).resolve().parent


def rot(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class Camera:
    def __init__(self, entry: dict):
        self.entry = entry
        self.file = entry["file"]
        self.R = rot(entry["rotation_wxyz"])
        self.C = np.array(entry["position"], float)
        self.fx, self.fy = entry["focal_px_x"], entry["focal_px_y"]
        self.cx, self.cy = entry["cx"], entry["cy"]
        self.W, self.H = entry["width"], entry["height"]

    def to_camera(self, P):
        """World point(s), shape (3,) or (N,3) -> camera frame."""
        P = np.asarray(P, float)
        return (FLIP @ (self.R.T @ (P - self.C).T)).T if P.ndim == 2 else FLIP @ (self.R.T @ (P - self.C))

    def project(self, P):
        """-> (u, v, depth) or None if the point is behind the camera."""
        c = self.to_camera(P)
        if c[2] <= 0.05:
            return None
        return (self.fx * c[0] / c[2] + self.cx, self.fy * c[1] / c[2] + self.cy, float(c[2]))

    def ray(self, u, v):
        """Pixel -> (origin, unit direction) in world coordinates."""
        d = self.R @ FLIP @ np.array([(u - self.cx) / self.fx, (v - self.cy) / self.fy, 1.0])
        return self.C, d / np.linalg.norm(d)


def load_cameras(path):
    return [Camera(e) for e in json.load(open(path, encoding="utf-8"))]


def load_registry(path=ROOT / "cabinet_registry.json"):
    return json.load(open(path, encoding="utf-8"))["cabinets"]


# Door-front direction of each cabinet row (ASSUMED from the layout: H01-H05 at x ~ -5.75 face the corridor, i.e. +x;
# the second row at x ~ -2.7 faces -x). Used only to ignore cameras behind the door.
def door_normal(pos):
    return np.array([1.0, 0.0, 0.0]) if pos["x"] < -4.0 else np.array([-1.0, 0.0, 0.0])
