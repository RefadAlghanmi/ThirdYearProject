import sys
import os
import sqlite3
import cv2
import json
import numpy as np
import threading
import matplotlib
matplotlib.use('Qt5Agg')  # Ensure compatibility with PyQt5
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QVBoxLayout, QHBoxLayout, QPushButton, QComboBox
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import torch
print("CUDA Available:", torch.cuda.is_available())

# Fix for OpenMP runtime error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


from ultralytics import YOLO
model = YOLO("yolov8n.pt")  # ✅ Correct way to initialize

# Force CPU usage when running detection or training
results = model.predict("heatmap.png", device="cpu")  # ✅ Correct usage
results[0].show()

class HeatmapGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.local_path = os.getcwd()
        self.available_db = self.load_db_files()
        self.selected_db = self.available_db[0] if self.available_db else None
        self.heatmap_path = None
        self.yolo_model = None  # Placeholder for YOLO model
        self.initUI()

    def initUI(self):
        main_layout = QVBoxLayout()
        input_layout = QHBoxLayout()

        self.label_low = QLabel('Enter lower range:')
        self.input_low = QLineEdit()
        input_layout.addWidget(self.label_low)cd.
        input_layout.addWidget(self.input_low)

        self.label_high = QLabel('Enter higher range:')
        self.input_high = QLineEdit()
        input_layout.addWidget(self.label_high)
        input_layout.addWidget(self.input_high)
        main_layout.addLayout(input_layout)

        self.dataset_selector = QComboBox()
        self.dataset_selector.addItems(self.available_db)
        self.dataset_selector.currentIndexChanged.connect(self.change_selected_db)
        main_layout.addWidget(self.dataset_selector)

        self.btn_generate = QPushButton('Generate Heatmap')
        self.btn_generate.clicked.connect(self.generate_heatmap)
        main_layout.addWidget(self.btn_generate)

        self.btn_convert = QPushButton('Convert JSON Labels to YOLO Format')
        self.btn_convert.clicked.connect( self.convert_json_to_yolo)
        main_layout.addWidget(self.btn_convert)

        self.btn_train = QPushButton('Train YOLO Model')
        self.btn_train.clicked.connect(self.train_yolo_model)
        main_layout.addWidget(self.btn_train)

        self.btn_run = QPushButton('Run YOLO Detection')
        self.btn_run.clicked.connect(self.run_yolo_detection)
        main_layout.addWidget(self.btn_run)

        self.image_label = QLabel("Heatmap Output")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 1px solid black;")
        self.image_label.setFixedSize(1000, 600)
        main_layout.addWidget(self.image_label)

        self.setLayout(main_layout)
        self.setWindowTitle('Heatmap Generator with YOLO')
        self.show()

    def load_db_files(self):
        return [file for file in os.listdir(self.local_path) if file.endswith(".db")]

    def change_selected_db(self):
        self.selected_db = self.dataset_selector.currentText()

    def generate_heatmap(self):
        if not self.selected_db:
            print("❌ No database selected.")
            return

        try:
            lower_range = int(self.input_low.text())
            higher_range = int(self.input_high.text())

            conn = sqlite3.connect(self.selected_db)
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM data")
            data = cursor.fetchall()
            conn.close()

            if not data:
                print("❌ No data found in the database.")
                return

            data_matrix = np.array(data).T  # Transpose, as in MATLAB

            if data_matrix.shape[1] <= higher_range:
                print("❌ Selected range exceeds dataset size.")
                return

            subset_data = data_matrix[:, lower_range:higher_range]

            # Extract Real & Imaginary Data
            real_data = subset_data[15:29, :]
            bz_real = real_data[:8, :]
            bx_real = real_data[8:14, :]

            imaginary_data = subset_data[1:15, :]
            bz_imaginary = imaginary_data[:8, :]
            bx_imaginary = imaginary_data[8:14, :]

            # Compute Magnitudes
            bx_magnitude = np.sqrt(bx_real ** 2 + bx_imaginary ** 2)
            bz_magnitude = np.sqrt(bz_real ** 2 + bz_imaginary ** 2)

            # Normalize
            bz_ref = bz_magnitude[:, 1]
            bz_ref = np.where(bz_ref == 0, 1, bz_ref)
            bz_normalized = (bz_magnitude - bz_ref[:, None]) / bz_ref[:, None]

            bx_ref = bx_magnitude[:, 1]
            bx_ref = np.where(bx_ref == 0, 1, bx_ref)
            bx_normalized = (bx_magnitude - bx_ref[:, None]) / bx_ref[:, None]

            # Plot Heatmap
            fig, axes = plt.subplots(2, 1, figsize=(10, 6), dpi = 300)
            
            # Bx Magnitude Heatmap
            im1 = axes[0].imshow(bx_normalized, cmap='jet', aspect='auto', interpolation='bilinear')
            axes[0].set_title("Bx Signal Magnitude Normalized", fontsize=12, fontweight="bold")
            axes[0].set_xlabel("Time", fontsize=10)
            axes[0].set_ylabel("Bx Magnitude", fontsize=10)
            axes[0].grid(True, alpha=0.3)  # Adjust grid visibility
            fig.colorbar(im1, ax=axes[0], shrink=0.7, pad=0.02)

            # Bz Magnitude Heatmap
            im2 = axes[1].imshow(bz_normalized, cmap='jet', aspect='auto', interpolation='bilinear')
            axes[1].set_title("Bz Signal Magnitude Normalized", fontsize=12, fontweight="bold")
            axes[1].set_xlabel("Time", fontsize=10)
            axes[1].set_ylabel("Bz Magnitude", fontsize=10)
            axes[1].grid(True, alpha=0.3)
            fig.colorbar(im2, ax=axes[1], shrink=0.7, pad=0.02)

            # Save the figure with MATLAB-style naming convention
            filename = f"heatmap[{lower_range},{higher_range}]D219355.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            self.image_label.setPixmap(QPixmap(filename).scaled(800, 800, Qt.KeepAspectRatio))
            self.heatmap_path = filename
            print(f"✅ Heatmap saved as {filename}")

        except Exception as e:
            print(f"❌ Error: {e}")

    def convert_json_to_yolo(self, json_folder="labels_json", image_folder="train/images", output_folder="train/labels"):
        """Convert JSON labels to YOLO format, extracting image dimensions."""
        json_folder = "labels_json"
        output_folder = "train/labels"
        image_folder = "train/images"  # Ensure images are here

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

            print(f"image hieght: ", image_height)
            print(f"image width: ", image_width)
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

                yolo_labels.append(f"1 {x_center} {y_center} {width} {height}")

            # Save YOLO labels
            txt_filename = os.path.join(output_folder, json_file.replace(".json", ".txt"))
            with open(txt_filename, "w") as f:
                f.write("\n".join(yolo_labels))

            print(f"✅ Converted {json_file} to YOLO format.")

        print("🎉 JSON to YOLO conversion completed!")
    
    def create_dataset_yaml(self):
        yaml_content = """train: train/images
        val: val/images

        nc: 1
        names:
        - heatmap
        """
        with open("dataset.yaml", "w") as f:
            f.write(yaml_content)
        print("✅ Created dataset.yaml successfully.")

    
    def train_yolo_model(self):
        """Train YOLO model in a separate thread to prevent UI freezing."""

        if not os.path.exists("dataset.yaml"):
            print("❌ dataset.yaml not found! Creating it now...")
            self.create_dataset_yaml()

        current_directory = os.getcwd()

         # Specify the output path for YOLO training results (e.g., saving in the current directory)
        output_path = os.path.join(current_directory, 'yolo_training_output')

        # Create the directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)

        self.thread = QThread()
        self.worker = TrainYOLOWorker(output_path)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def run_yolo_detection(self):
        """Run YOLO detection safely."""
        if YOLO is None:
            print("❌ YOLO library is not installed.")
            return
        if not self.heatmap_path or not os.path.exists(self.heatmap_path):
            print("❌ No heatmap available for detection.")
            return

        self.yolo_model = YOLO("yolov8n.pt")
        results = self.yolo_model(self.heatmap_path, conf=0.1)

        # Check if any detections were made
        if not results[0].boxes:
            print("❌ No detections found.")
            return
        
        print("Boxes Detected:", results[0].boxes)

        
        boxes = results[0].boxes

        result_img = cv2.imread(self.heatmap_path)
        image_height, image_width, _ = result_img.shape  # Get actual image size

        for box in boxes:
            x_center, y_center, width, height = box.xywhn[0].tolist()  # Get YOLO normalized coordinates

            # Convert from normalized (0-1) to actual pixel coordinates
            x_center *= image_width
            y_center *= image_height
            width *= image_width
            height *= image_height

            # Calculate box corners
            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)

            # Ensure bounding box is within image boundaries
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(image_width, x2), min(image_height, y2)
            cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 0, 255), 2)


        
        # Draw the bounding box on the image (Green color, thickness 2)
        #cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        
        result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

        # Get the current working directory to save the file
        result_filename = f"{self.selected_db}_low{self.input_low.text()}_high{self.input_high.text()}_yolo.png"
        cv2.imwrite(result_filename, result_img)
        self.image_label.setPixmap(QPixmap(result_filename).scaled(800, 800, Qt.KeepAspectRatio))
        cv2.imwrite(result_filename, result_img)


        self.image_label.setPixmap(QPixmap(result_filename).scaled(800, 800, Qt.KeepAspectRatio))
        print(f"✅ Detection results saved as {result_filename}")


class TrainYOLOWorker(QThread):
    finished = pyqtSignal()

    def __init__(self, output_path):
        super().__init__()
        self.output_path = os.getcwd()  # Store output path
        
    def run(self):
        try:
            print("🔹 Training YOLO model...")

            # ✅ Initialize YOLO model inside `run()` to prevent attribute errors
            model = YOLO("yolov8n.pt")

            # ✅ Train the model and store the results
            results = model.train(data="dataset.yaml", epochs=180 , imgsz=640)
            
            print("🔹 validating YOLO model...")
            # ✅ Run validation to obtain metrics
            val_results = model.val()

            # ✅ Extract validation metrics safely
            #precision = float(val_results.box.mean_precision) if hasattr(val_results.box, 'mean_precision') else None
            precision = float(val_results.results_dict["metrics/precision(B)"])
            #recall = float(val_results.box.mean_recall) if hasattr(val_results.box, 'mean_recall') else None
            recall = float(val_results.results_dict["metrics/recall(B)"])
            f1_score = float(val_results.box.f1) if hasattr(val_results.box, 'f1') else None
            map50 = float(val_results.box.map50) if hasattr(val_results.box, 'map50') else None
            map5095 = float(val_results.box.map) if hasattr(val_results.box, 'map') else None

            # ✅ Print extracted metrics
            print("✅ YOLO Model Training Completed!")
            print(f"📊 Precision: {precision:.4f}" if precision is not None else "❌ Precision metric missing")
            print(f"📊 Recall: {recall:.4f}" if recall is not None else "❌ Recall metric missing")
            print(f"📊 F1 Score: {f1_score:.4f}" if f1_score is not None else "❌ F1 Score metric missing")
            print(f"📊 mAP@50: {map50:.4f}" if map50 is not None else "❌ mAP@50 metric missing")
            print(f"📊 mAP@50-95: {map5095:.4f}" if map5095 is not None else "❌ mAP@50-95 metric missing")


        except Exception as e:
            print(f"❌ Training failed: {e}")
        finally:
            self.finished.emit()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = HeatmapGUI()
    sys.exit(app.exec_())
