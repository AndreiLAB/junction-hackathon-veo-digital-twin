import numpy as np
import math

def quaternion_to_matrix(q):
    w, x, y, z = q
    # Normalize quaternion
    norm = math.sqrt(w*w + x*x + y*y + z*z)
    if norm == 0:
        return np.eye(3)
    w, x, y, z = w/norm, x/norm, y/norm, z/norm

    return np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*w*z,     2*x*z + 2*w*y],
        [    2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z,     2*y*z - 2*w*x],
        [    2*x*z - 2*w*y,     2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])

def calculate_forward_vector(q, forward_axis="-z"):
    mat = quaternion_to_matrix(q)
    
    if forward_axis == "z":
        vec = mat[:, 2]
    elif forward_axis == "-z":
        vec = -mat[:, 2]
    elif forward_axis == "y":
        vec = mat[:, 1]
    elif forward_axis == "-y":
        vec = -mat[:, 1]
    elif forward_axis == "x":
        vec = mat[:, 0]
    elif forward_axis == "-x":
        vec = -mat[:, 0]
    else:
        raise ValueError(f"Unknown forward axis: {forward_axis}")
        
    # Normalize just in case
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec

def vector_to_yaw_pitch_roll(vec):
    """
    Assuming Z is up (common in E57/Matterport, but we will verify).
    If Z is up:
    Yaw is rotation around Z (xy plane angle)
    Pitch is angle from xy plane
    """
    x, y, z = vec
    
    # Calculate yaw (around Z axis) - in degrees
    yaw = math.degrees(math.atan2(y, x))
    if yaw < 0:
        yaw += 360.0
        
    # Calculate pitch (elevation)
    xy_dist = math.sqrt(x*x + y*y)
    pitch = math.degrees(math.atan2(z, xy_dist))
    
    # Roll isn't determinable from just a forward vector without an up vector.
    # But usually roll is 0 for upright cameras.
    roll = 0.0
    
    return yaw, pitch, roll

def extract_euler_from_matrix(mat):
    """Alternative way to get euler angles directly from rotation matrix"""
    # Assuming XYZ or similar rotation order. Will refine during calibration.
    # For now we use the vector approach for camera direction.
    pass
