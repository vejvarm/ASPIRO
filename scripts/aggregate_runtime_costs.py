import json
import csv
from pathlib import Path

source_root = Path("/media/freya/kubuntu-data/datasets/d2t/csqa-d2t/EMNLP2023_rebuttal/experiments/webnlg/ASDOT/")

# Locate all runtime_seconds.json files
json_files = list(source_root.glob("**/runtime_seconds.json"))

# Dictionary to store subfolder names and their respective 'mean' values
results = {}

for json_file in json_files:
    # Extract the subfolder name
    subfolder_name = json_file.parent.parent.name

    # Extract the 'mean' value from the JSON file
    with json_file.open('r') as f:
        data = json.load(f)
        mean_value = data.get('mean', None)  # Default to None if 'mean' key is not found

    results[subfolder_name] = mean_value

# Save the results to a CSV file
csv_file_path = source_root / "results.csv"
with csv_file_path.open('w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(["Subfolder Name", "Mean Value"])  # CSV header
    for subfolder, mean_value in results.items():
        csvwriter.writerow([subfolder, mean_value])

print(f"Results saved to: {csv_file_path}")
