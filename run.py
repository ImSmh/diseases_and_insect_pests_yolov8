from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("E:\\mycode\\pytorch\\classify\\ultralytics\\yolov8x-cls.pt")
    model.train(data="E:\\datasets\\flowers\\asd", epochs=3, batch=4, imgsz=640, verbose=True)
    model.eval()