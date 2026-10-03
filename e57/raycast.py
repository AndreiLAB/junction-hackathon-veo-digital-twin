import numpy as np

def pose_matrix(qw, qx, qy, qz, tx, ty, tz):
    w, x, y, z = qw, qx, qy, qz
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]
    ])
    return R, np.array([tx, ty, tz], float)

def project(cam, f, cx, cy, W, H):
    z = cam[:, 2]
    ok = z > 0.3
    u = np.full(len(cam), -1.0)
    v = np.full(len(cam), -1.0)
    u[ok] = f * cam[ok, 0] / z[ok] + cx
    v[ok] = f * cam[ok, 1] / z[ok] + cy
    ok &= (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return u, v, ok

def raycast_2d_to_3d(bbox, scan_xyz, camera_pose, camera_intrinsics, image_width, image_height, flip_convention=(1, 1, 1), inverse_rotation=False):
    """
    Intersects a 2D bounding box with the physical point cloud to find the 3D surface.
    Returns (median_x, median_y, median_z) or None.
    """
    R, C = camera_pose
    f, cx, cy = camera_intrinsics
    Rc2w = R.T if inverse_rotation else R
    flip = np.array(flip_convention, float)
    
    # transform point cloud to camera coordinates
    cam = ((scan_xyz - C) @ Rc2w) * flip
    u, v, ok = project(cam, f, cx, cy, image_width, image_height)
    
    x0, y0, x1, y1 = bbox
    cx_box = (x0 + x1) / 2
    cy_box = (y0 + y1) / 2
    rad = max(4.0, 0.004 * image_width)
    
    # Filter points inside the bounding box and close to the center
    near = ok & (np.abs(u - cx_box) < rad) & (np.abs(v - cy_box) < rad)
    
    if near.sum() == 0:
        return None
        
    z = cam[near, 2]
    # Filter the front surface (remove background points behind the object)
    front = z < z.min() * 1.05 + 0.03
    
    anchor_xyz = np.median(scan_xyz[near][front], axis=0)
    return tuple(anchor_xyz.round(4).tolist())
