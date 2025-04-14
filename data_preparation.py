# data_preparation.py
import json
import numpy as np
from feature_extraction import extract_features_from_landmarks

def load_data(json_file_path):
    """
    Loads training data from a JSON file.
    
    The JSON file is expected to contain a list of samples.
    Each sample should have:
      - "landmarks": a list of 21 points (each a dict with keys 'x', 'y', 'z')
      - "instruction": a string (e.g., "ring4>middle3")
      - Optionally, "overlap": a boolean where True indicates overlapping fingers.
    
    If "overlap" is not provided, a simple heuristic is used: if ">" is found
    in the instruction, the sample is labeled as 1 (overlap); otherwise 0.
    """
    features_list = []
    labels_list = []
    
    with open(json_file_path, "r") as f:
        data = json.load(f)
    
    # Check if the file holds a list of samples or a single sample.
    if isinstance(data, dict):
        # If it's a dict, assume a single sample; convert to a list.
        data = [data]

    for sample in data:
        # Ensure the sample has landmark data before processing.
        landmarks = sample.get("landmarks")
        if not landmarks:
            continue
        
        # Extract feature vector from the 21 hand landmarks.
        features = extract_features_from_landmarks(landmarks)
        features_list.append(features)
        
        # Determine label: use "overlap" key if present; otherwise infer from instruction.
        if "overlap" in sample:
            label = 1 if sample["overlap"] else 0
        else:
            # Simple heuristic: if ">" is in the instruction, label as overlap.
            instruction = sample.get("instruction", "")
            label = 1 if ">" in instruction else 0
        labels_list.append(label)
        
    X = np.array(features_list)
    y = np.array(labels_list)
    
    return X, y

if __name__ == '__main__':
    # Update the JSON path to your file
    json_file_path = "overlap_dataset/overlap_data_20250414_092037.json"
    X, y = load_data(json_file_path)
    print("Features shape:", X.shape)
    print("Labels shape:", y.shape)
