import json
import csv

def convert_json_to_csv(json_path, csv_path):
    # Load data from the JSON file.
    with open(json_path, "r") as f:
        data = json.load(f)
    
    # Ensure data is a list (wrap in a list if a single sample is provided)
    if isinstance(data, dict):
        data = [data]

    output_rows = []
    
    # Define which landmarks belong to each finger (1-indexed as described)
    # In Python's 0-indexing, this means:
    # thumb: landmarks[1] to landmarks[4],
    # index: landmarks[5] to landmarks[8], etc.
    finger_mapping = {
        "thumb": range(1, 5),    # indices 1,2,3,4
        "index": range(5, 9),    # indices 5,6,7,8
        "middle": range(9, 13),  # indices 9,10,11,12
        "ring": range(13, 17),   # indices 13,14,15,16
        "pinky": range(17, 21)   # indices 17,18,19,20
    }
    
    for sample in data:
        landmarks = sample.get("landmarks")
        # Check that landmarks exist and there are enough points (should be 21)
        if not landmarks or len(landmarks) < 21:
            continue
        
        row = {}
        # For each finger, extract its landmark coordinates and convert to a string.
        for finger, indices in finger_mapping.items():
            coords = []
            for i in indices:
                lm = landmarks[i]
                # Format each coordinate to 6 decimal places
                coords.append(f"{lm['x']:.6f},{lm['y']:.6f},{lm['z']:.6f}")
            # Join coordinate strings with a semicolon
            row[finger] = ";".join(coords)
        
        # Add top and bottom finger columns
        row["top_finger"] = sample.get("top_finger", "")
        row["bottom_finger"] = sample.get("bottom_finger", "")
        
        output_rows.append(row)
    
    # Define the order of the columns in the CSV
    fieldnames = ["thumb", "index", "middle", "ring", "pinky", "top_finger", "bottom_finger"]
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"CSV file created at {csv_path}")

if __name__ == "__main__":
    json_path = "overlap_dataset/overlap_data_20250414_092037.json"
    csv_path = "overlap_dataset/overlap_data_20250414_092037.csv"
    convert_json_to_csv(json_path, csv_path)
