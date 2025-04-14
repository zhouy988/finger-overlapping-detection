# feature_extraction.py
import numpy as np

def normalize_landmarks(landmarks):
    """
    Normalize landmarks by translating them so that the wrist (landmark[0])
    is at the origin and scaling by the maximum distance.
    """
    wrist = np.array([landmarks[0]['x'], landmarks[0]['y'], landmarks[0]['z']])
    points = np.array([[lm['x'], lm['y'], lm['z']] for lm in landmarks])
    normalized = points - wrist
    scale = np.linalg.norm(normalized, axis=1).max()
    if scale != 0:
        normalized = normalized / scale
    return normalized

def compute_angle(p1, p2, p3):
    """
    Compute the angle at point p2 given three 3D points p1, p2, and p3.
    Returns the angle in radians.
    """
    v1 = p1 - p2
    v2 = p3 - p2
    dot_prod = np.dot(v1, v2)
    norm_prod = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm_prod == 0:
        return 0.0
    # Clip the cosine value to avoid numerical issues.
    angle = np.arccos(np.clip(dot_prod / norm_prod, -1.0, 1.0))
    return angle

def extract_features_from_landmarks(landmarks):
    """
    Given 21 hand landmarks (each a dict with x, y, z), return a feature vector.
    Features include:
      - Angles (e.g. the angle at the index finger’s joint)
      - Pairwise distances between finger tips
      - Projection spread (range) on the xy, xz, and zy planes
    """
    points = normalize_landmarks(landmarks)
    features = []
    
    # Example: Compute an angle for the index finger (using landmarks 5, 6, 7)
    angle_index = compute_angle(points[5], points[6], points[7])
    features.append(angle_index)
    
    # Compute distances between each pair of finger tips (indices: 4, 8, 12, 16, 20)
    tip_indices = [4, 8, 12, 16, 20]
    for i in range(len(tip_indices)):
        for j in range(i+1, len(tip_indices)):
            dist = np.linalg.norm(points[tip_indices[i]] - points[tip_indices[j]])
            features.append(dist)
    
    # Projection features: range (peak-to-peak) for each projection
    xy_proj = points[:, :2]   # x and y
    xz_proj = points[:, [0, 2]] # x and z
    zy_proj = points[:, 1:]    # y and z
    
    features.append(np.ptp(xy_proj[:, 0]))  # range in x on xy-plane
    features.append(np.ptp(xy_proj[:, 1]))  # range in y on xy-plane
    features.append(np.ptp(xz_proj[:, 0]))  # range in x on xz-plane
    features.append(np.ptp(xz_proj[:, 1]))  # range in z on xz-plane
    features.append(np.ptp(zy_proj[:, 0]))  # range in y on zy-plane
    features.append(np.ptp(zy_proj[:, 1]))  # range in z on zy-plane
    
    return np.array(features)
