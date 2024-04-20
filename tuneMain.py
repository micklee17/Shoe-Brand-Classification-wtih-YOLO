from ultralytics import YOLO

"""for hyper-parameter tuning"""

if __name__ == '__main__':
    # Load a model
    model = YOLO('yolov8l-cls.pt')  # load a custom model

    # Validate the model
    model.tune(data='Nike_Adidas_converse_Shoes_image_dataset', imgsz = 256, epochs = 100, patience = 20, dropout = 0.25, iterations = 300)