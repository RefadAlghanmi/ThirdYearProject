import os
import sys
import json
import cv2
print(cv2.__version__)

def convert_json_to_yolo(json_folder, image_folder, output_folder):
    """Convert JSON labels to YOLO format for a given dataset folder."""
    
    if not os.path.exists(json_folder):
        print(f"❌ Error: Folder '{json_folder}' does not exist. Please create it and add JSON label files.")
        return

    os.makedirs(output_folder, exist_ok=True)

    json_files = [f for f in os.listdir(json_folder) if f.endswith(".json")]

    if not json_files:
        print(f"❌ No JSON label files found in '{json_folder}'. Please add some and try again.")
        return

    for json_file in json_files:
        json_path = os.path.join(json_folder, json_file)

        with open(json_path, "r") as f:
            data = json.load(f)

        # 🚨 Use JSON filename to find corresponding image
        image_filename = json_file.replace(".json", ".png")
        image_path = os.path.join(image_folder, image_filename)

        if not os.path.exists(image_path):
            print(f"⚠️ Image {image_filename} not found in {image_folder}. Skipping...")
            continue

        # Get image dimensions
        image = cv2.imread(image_path)
        image_height, image_width, _ = image.shape  # Extract width & height

        print(f"🔹 Processing: {image_filename} (H: {image_height}, W: {image_width})")
        yolo_labels = []

        for shape in data.get("shapes", []):
            shape_type = shape.get("shape_type", None)  # ✅ Safely get shape type
            
            if shape_type and shape_type != "rectangle":
                print(f"⚠️ Skipping non-rectangle shape in {json_file}")
                continue

            label = shape.get("label", "ROI")  # ✅ Use "ROI" if label is missing
            points = shape.get("points", [])

            if len(points) < 2:
                print(f"⚠️ Skipping invalid annotation in {json_file}. Not enough points.")
                continue

            (x1, y1), (x2, y2) = points[:2]

            # Convert to YOLO format (center_x, center_y, width, height) normalized
            x_center = ((x1 + x2) / 2) / image_width
            y_center = ((y1 + y2) / 2) / image_height
            width = abs(x2 - x1) / image_width
            height = abs(y2 - y1) / image_height

            yolo_labels.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        # Save YOLO labels
        txt_filename = os.path.join(output_folder, json_file.replace(".json", ".txt"))
        with open(txt_filename, "w") as f:
            f.write("\n".join(yolo_labels))

        print(f"✅ Converted {json_file} to YOLO format.")

    print("🎉 JSON to YOLO conversion completed!")

def fix_yolo_labels(label_folder):
    """Fix incorrect YOLO class labels by changing class 1 to class 0."""
    for filename in os.listdir(label_folder):
        if filename.endswith(".txt"):
            file_path = os.path.join(label_folder, filename)
            with open(file_path, "r") as f:
                lines = f.readlines()
            
            fixed_lines = []
            for line in lines:
                parts = line.split()
                if parts and parts[0] == "1":
                    parts[0] = "0"
                fixed_lines.append(" ".join(parts) + "\n")
            
            with open(file_path, "w") as f:
                f.writelines(fixed_lines)
    
    print("✅ All labels fixed: Class 1 changed to Class 0.")

def check_missing_files(image_folder, label_folder):
    """Check if there are missing images or labels."""
    image_files = {f.replace(".png", "") for f in os.listdir(image_folder) if f.endswith(".png")}
    label_files = {f.replace(".txt", "") for f in os.listdir(label_folder) if f.endswith(".txt")}
    
    missing_labels = image_files - label_files
    missing_images = label_files - image_files
    
    print(f"Missing Labels: {missing_labels}")
    print(f"Missing Images: {missing_images}")

def delete_cache(cache_file):
    """Delete the YOLO dataset cache file."""
    if os.path.exists(cache_file):
        os.remove(cache_file)
        print("✅ Deleted labels cache.")
    else:
        print("⚠️ No cache file found.")

if __name__ == "__main__":
    # Define dataset directories
    datasets = {
        "train": {
            "json_folder": "train/labels_json",
            "image_folder": "train/images",
            "output_folder": "train/labels"
        },
        "val": {
            "json_folder": "val/labels_json",
            "image_folder": "val/images",
            "output_folder": "val/labels"
        }
    }

    # Run conversion for both train and val datasets
    for dataset, paths in datasets.items():
        print(f"\n🚀 Converting {dataset.upper()} dataset...")
        convert_json_to_yolo(paths["json_folder"], paths["image_folder"], paths["output_folder"])
        fix_yolo_labels(paths["output_folder"])
        check_missing_files(paths["image_folder"], paths["output_folder"])
    
    # Delete cache to avoid training issues
    delete_cache("train/labels.cache")
