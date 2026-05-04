from ultralytics import YOLO
from waste_analysis import analyze_waste

# Load model
model = YOLO("best.pt")

# Run detection
results = model("Screenshot 2026-02-24 115111.jpg")

detected_classes = []

for box in results[0].boxes:
    cls_id = int(box.cls)
    class_name = model.names[cls_id]
    detected_classes.append(class_name)

analysis = analyze_waste(detected_classes)

print("Detected Objects:", detected_classes)
print("Final Status:", analysis["status"])
print("Recyclable Count:", analysis["recyclable"])
print("Non-Recyclable Count:", analysis["non_recyclable"])
print("Contamination %:", analysis["contamination_percent"])