import os
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO


def train_yolo_model():
    """Train YOLO model with increasing epochs and record performance metrics."""
    
    # Check if dataset.yaml exists
    if not os.path.exists("dataset.yaml"):
        print("❌ dataset.yaml not found! Please create it before training.")
        return

    model = YOLO("yolov8n.pt")  # Initialize YOLO model
    
    epochs_range = list(range(100, 201, 10))  # Epochs from 10 to 100 in increments of 10
    precisions, recalls, f1_scores, map50s, map5095s = [], [], [], [], []
    
    for epochs in epochs_range:
        print(f"🔹 Training YOLO model for {epochs} epochs...")
        results = model.train(data="dataset.yaml", epochs=epochs, imgsz=640)
        
        print("🔹 Validating YOLO model...")
        val_results = model.val()
        
        # Extract validation metrics
        precision = float(val_results.results_dict.get("metrics/precision(B)", 0))
        recall = float(val_results.results_dict.get("metrics/recall(B)", 0))
        f1_score = float(val_results.box.f1) if hasattr(val_results.box, 'f1') else None
        map50 = float(val_results.box.map50) if hasattr(val_results.box, 'map50') else None
        map5095 = float(val_results.box.map) if hasattr(val_results.box, 'map') else None

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)
        map50s.append(map50)
        map5095s.append(map5095)
        
        print(f"✅ Completed {epochs} epochs")
        print(f"📊 Precision: {precision:.4f}")
        print(f"📊 Recall: {recall:.4f}")
        print(f"📊 F1 Score: {f1_score:.4f}")
        print(f"📊 mAP@50: {map50:.4f}")
        print(f"📊 mAP@50-95: {map5095:.4f}")
        
    # Plot the performance metrics
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, precisions, label='Precision', marker='o')
    plt.plot(epochs_range, recalls, label='Recall', marker='s')
    plt.plot(epochs_range, f1_scores, label='F1 Score', marker='^')
    plt.plot(epochs_range, map50s, label='mAP@50', marker='d')
    plt.plot(epochs_range, map5095s, label='mAP@50-95', marker='x')
    
    plt.xlabel("Epochs")
    plt.ylabel("Performance Metrics")
    plt.title("YOLO Training Performance Over Increasing Epochs")
    plt.legend()
    plt.grid()
    plt.savefig("yolo_training_performance1.png")  # Save the plot
    plt.show()
    
    print("🎉 Training completed. Results saved as 'yolo_training_performance1.png'")


if __name__ == "__main__":
    train_yolo_model()
