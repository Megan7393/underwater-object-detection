from ultralytics import YOLO

# Load a pre-trained YOLOv8 nano model
model = YOLO("yolov8n.pt")

# Run prediction on the test images folder
results = model.predict(
    source="data/test_images",
    save=True,
    conf=0.25
)

print("Prediction complete.")
print("Results saved in the runs/detect folder.")