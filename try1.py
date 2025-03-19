from ultralytics import YOLO

model = YOLO("yolo11n.pt")
results= model.val(data="coco.yaml", imgsz=640)
print(results.box.map)  # Print mAP50-95