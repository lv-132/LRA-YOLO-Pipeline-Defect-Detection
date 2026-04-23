
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'\.yolov8n-seg.yaml')
    model.train(data=r'\.data.yaml',
                imgsz=640,
                epochs=10,
                single_cls=False,
                batch=16,
                workers=6,
                device='0',
                augment=False,
                optimizer='SGD',
                cos_lr=True,
                lr0=0.01,
                lrf=0.1,
                name='run',
                )

