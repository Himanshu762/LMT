#!/usr/bin/env python3
"""
Main entry point for the Transit ETA Prediction System V2.
This script is called by the Docker container to make predictions.
Accepts JSON input file with format: {'1':'path1.parquet', '2':'path2.parquet'}
"""

import sys
import os
import json
from predict_v2 import predict_eta

def main():
    """Main function to run predictions."""
    # Determine input JSON path
    # Priority: 1. Command line arg, 2. /app/data/input.json, 3. ./input.json
    if len(sys.argv) >= 2:
        input_json_path = sys.argv[1]
    elif os.path.exists("/app/data/input.json"):
        input_json_path = "/app/data/input.json"
    elif os.path.exists("input.json"):
        input_json_path = "input.json"
    else:
        print("Error: No input JSON file found", file=sys.stderr)
        print("Searched locations:", file=sys.stderr)
        print("  - Command line argument", file=sys.stderr)
        print("  - /app/data/input.json", file=sys.stderr)
        print("  - ./input.json", file=sys.stderr)
        sys.exit(1)

    # Determine artifacts directory
    if len(sys.argv) >= 3:
        artifacts_dir = sys.argv[2]
    elif os.path.exists("/app/artifacts_v2"):
        artifacts_dir = "/app/artifacts_v2"
    elif os.path.exists("artifacts_v2"):
        artifacts_dir = "artifacts_v2"
    elif os.path.exists("artifacts_v2_test"):  # Fallback for local testing
        artifacts_dir = "artifacts_v2_test"
    else:
        print("Error: Artifacts directory not found", file=sys.stderr)
        print("Searched: /app/artifacts_v2, ./artifacts_v2, ./artifacts_v2_test", file=sys.stderr)
        sys.exit(1)

    # Verify input JSON exists
    if not os.path.exists(input_json_path):
        print(f"Error: Input JSON file not found: {input_json_path}", file=sys.stderr)
        sys.exit(1)

    # Verify artifacts directory exists
    if not os.path.exists(artifacts_dir):
        print(f"Error: Artifacts directory not found: {artifacts_dir}")
        sys.exit(1)

    # Load input JSON
    try:
        with open(input_json_path, 'r') as f:
            input_data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {e}", file=sys.stderr)
        sys.exit(1)

    # Get base directory for resolving relative paths (directory of input.json)
    input_base_dir = os.path.dirname(os.path.abspath(input_json_path))

    # Aggregate predictions from all input files
    all_predictions = {}

    for file_id, test_file_path in input_data.items():
        # Resolve relative paths relative to input.json location
        if not os.path.isabs(test_file_path):
            # If path is relative, resolve it relative to input.json's directory
            resolved_path = os.path.join(input_base_dir, test_file_path)
        else:
            resolved_path = test_file_path

        # Verify test file exists
        if not os.path.exists(resolved_path):
            print(f"Warning: Test file not found: {resolved_path}", file=sys.stderr)
            continue

        # Run prediction for this file
        try:
            result_json = predict_eta(resolved_path, artifacts_dir)
            result = json.loads(result_json)

            # Merge predictions (format: {route_id: {stop_id: timestamp}})
            for route_id, stop_predictions in result.items():
                if route_id not in all_predictions:
                    all_predictions[route_id] = {}
                all_predictions[route_id].update(stop_predictions)

        except Exception as e:
            print(f"Prediction failed for {resolved_path}: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            continue

    # Output final aggregated predictions to /app/output.json
    output_path = "/app/output.json"
    try:
        with open(output_path, 'w') as f:
            json.dump(all_predictions, f, indent=2)
    except Exception as e:
        print(f"Error writing output file: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
