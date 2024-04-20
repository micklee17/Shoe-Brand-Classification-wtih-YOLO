from ultralytics import YOLO

if __name__ == '__main__':

    model = YOLO('yolov8l-cls.pt')  # load a pretrained model

    # Train the model
    results = model.train(data='Nike_Adidas_converse_Shoes_image_dataset', imgsz = 256, epochs = 1200, patience = 100, dropout = 0.25)