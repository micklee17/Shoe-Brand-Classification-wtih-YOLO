from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    model = YOLO('model.pt')  # load a custom model

    # Validate the model
    metrics = model.val(split = 'test')  # no arguments needed, dataset and settings remembered
    metrics.top1   # top1 accuracy
    #metrics.top5   # top5 accuracy