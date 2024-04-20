from ultralytics import YOLO
import argparse
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, help='Path to the trained model')
parser.add_argument('--test_data_path', type=str, help='Path to the test data.')
parser.add_argument('--result_path', type=str, help='Path to save the results.')
args = parser.parse_args()

# Load model
model = YOLO(args.model_path)

# Load images and predict
predicted_results = []

for filename in os.listdir(args.test_data_path):
    source = os.path.join(args.test_data_path, filename)
    results = model(source)
    for result in results:
        predicted_results.append(f"{filename}\t\t\t{result.names[result.probs.top1]}\t\t\t{round(result.probs.top1conf.item(), 2)}")

# Save results to txt
np.savetxt(args.result_path, predicted_results, fmt='%s')