from ultralytics import YOLO
import torch
import numpy as np

if __name__ == '__main__':
    # Load a model
    model = YOLO('model.pt')
    # Define path to the image file(can be a folder)
    source = 'kyle-austin-7yUfx7A9rMU-unsplash.jpg'

    results = model.predict(source, save = True) #predict & save annotated images

    