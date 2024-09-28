import argparse

from ultralytics import YOLO

def main(config_yaml):
    model = YOLO('yolov8m.pt')
    data_format = config_yaml.split('/')[-1].split('.')[0]
    
    model.train(
        data=config_yaml,
        epochs=50,
        batch=8,
        name=f'mice_yolo_{data_format}'
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="YOLOv8 Training Script")
    parser.add_argument('--config', type=str, help="Path to the data config YAML file", default='configs/train/frame.yaml')

    args = parser.parse_args()
    
    main(args.config)
