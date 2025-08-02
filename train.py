import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/11/yolo11n-VHFE-CSTDHead.yaml')
    # model.load('yolo11n.pt') # loading pretrain weights
    model.train(data='../data/SSDD.yaml',
                cache=False,
                imgsz=512,
                epochs=300,
                batch=16,
                close_mosaic=10,
                workers=8,
                # device='0',
                optimizer='SGD',
                # patience=0,
                # resume=True,
                # amp=False,
                # fraction=0.2,
                project='runs/train',
                name='exp',
                )