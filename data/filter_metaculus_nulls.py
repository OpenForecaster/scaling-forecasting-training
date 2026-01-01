#!/usr/bin/env python3
"""
Metaculus Null Resolution Filter

Purpose:
    Filters out Metaculus questions with null resolutions from the dataset.
    Ensures only resolved questions are included in training/evaluation data.

Input:
    - data/metaculus_resolved.json: Raw Metaculus questions with possible null resolutions

Output:
    - data/metaculus_resolved_filtered.json: Filtered questions with valid resolutions only

Usage:
    python filter_metaculus_nulls.py
"""

import json

# Load the JSON data from the specified file
input_file_path = 'data/metaculus_resolved.json'
output_file_path = 'data/metaculus_resolved_filtered.json'

# Read the JSON data from the input file
with open(input_file_path, 'r') as input_file:
    data = json.load(input_file)

# Filter out items where the 'resolution' field is None
filtered_data = [item for item in data if item['resolution'] is not None]

# Write the filtered data to the output file
with open(output_file_path, 'w') as output_file:
    json.dump(filtered_data, output_file, indent=2)

print(f"{len(filtered_data)} questions saved to {output_file_path}")