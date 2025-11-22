#!/usr/bin/env python3
"""
Convert eta_model_v2.joblib to eta_model_v2.json
Run this script once to convert the existing model to JSON format.
"""

import os
import joblib

def convert_model():
    """Convert joblib model to JSON format."""
    artifacts_dir = "artifacts_v2"

    joblib_path = os.path.join(artifacts_dir, "eta_model_v2.joblib")
    json_path = os.path.join(artifacts_dir, "eta_model_v2.json")

    if not os.path.exists(joblib_path):
        print(f"Error: {joblib_path} not found")
        return

    print(f"Loading model from {joblib_path}...")
    model = joblib.load(joblib_path)

    print(f"Saving model to {json_path}...")
    model.save_model(json_path)

    print("Conversion complete!")
    print(f"JSON model saved to: {json_path}")
    print(f"\nYou can now delete {joblib_path} if desired.")

if __name__ == "__main__":
    convert_model()
