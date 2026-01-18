

from models import YOLO


if __name__ == "__main__":
    model = YOLO(r"E:\yolov11\ultralytics-main\ultralytics\cfg\models\yolov11n-dualbranch.yaml")
    results = model.train(
        data=r"E:\yolov11\ultralytics-main\ultralytics\cfg\datasets\data.yaml",
        epochs=300,
        imgsz=640,
        batch=1,
        # cache = False,
        # single_cls = False,  # 是否是单类别检测
        # workers = 0,
        # resume=,
        amp=True
        )