"""
Transit ETA Prediction - Vectorized Inference Pipeline (V2)
GPS-based prediction with vectorized operations.
"""

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')

import pandas as pd
import numpy as np
import json
import sys
import os
import xgboost as xgb
from utils import (
    haversine_vectorized,
    clean_parquet_data,
    match_gps_to_stops_vectorized
)


def predict_eta(test_file_path, artifacts_dir="artifacts_v2"):
    """
    Main prediction function using GPS-based approach.

    Args:
        test_file_path: Path to test parquet file with GPS data
        artifacts_dir: Directory containing trained model and artifacts

    Returns:
        JSON string with predictions
    """
    # --- 1. Load Artifacts ---
    print("Loading artifacts...", file=sys.stderr)
    model = xgb.XGBRegressor()
    model.load_model(os.path.join(artifacts_dir, "eta_model_v2.json"))
    master_route_info = pd.read_csv(os.path.join(artifacts_dir, "master_route_info.csv"))
    historical_stats = pd.read_csv(os.path.join(artifacts_dir, "historical_stats.csv"))
    ideal_stats = pd.read_csv(os.path.join(artifacts_dir, "ideal_time_stats.csv"))

    # --- 2. Load and Clean Test Sequence ---
    print(f"Loading test sequence from {test_file_path}...", file=sys.stderr)
    test_df = pd.read_parquet(test_file_path)
    test_df = clean_parquet_data(test_df)

    if test_df.empty:
        return json.dumps({"error": "No valid data in test file"})

    # Get trip information
    route_id = test_df['route_id'].iloc[0]
    trip_id = test_df['trip_id'].iloc[0]

    print(f"Processing trip_id: {trip_id}, route_id: {route_id}", file=sys.stderr)

    # --- 3. Get Route Information ---
    route_stops_df = master_route_info[master_route_info['route_id'] == route_id][[
        'stop_id', 'stop_lat', 'stop_lon', 'stop_sequence'
    ]].drop_duplicates().sort_values('stop_sequence')

    if route_stops_df.empty:
        print(f"Warning: No route information found for route_id {route_id}", file=sys.stderr)
        return json.dumps({str(int(route_id)): {}})

    # --- 4. Localize Bus Position ---
    print("Localizing bus position...", file=sys.stderr)

    # Find which stop the bus is currently at/near
    last_location = test_df.iloc[-1]
    last_location_df = pd.DataFrame([{
        'latitude': last_location['latitude'],
        'longitude': last_location['longitude']
    }])

    nearest_stop_ids = match_gps_to_stops_vectorized(last_location_df, route_stops_df)
    last_stop_id = nearest_stop_ids[0]
    last_stop_info = route_stops_df[route_stops_df['stop_id'] == last_stop_id].iloc[0]
    last_stop_sequence = last_stop_info['stop_sequence']
    last_timestamp = test_df['timestamp'].max()

    print(f"Last stop sequence: {last_stop_sequence}", file=sys.stderr)

    # --- 5. Get Remaining Stops ---
    remaining_stops = route_stops_df[route_stops_df['stop_sequence'] > last_stop_sequence]

    if remaining_stops.empty:
        print("Bus has reached the final stop.", file=sys.stderr)
        route_id_str = str(int(route_id))
        return json.dumps({route_id_str: {}})

    # --- 6. Iterative Prediction (Vectorized where possible) ---
    print(f"Predicting ETA for {len(remaining_stops)} remaining stops...", file=sys.stderr)

    predictions = []
    current_time = last_timestamp
    current_stop_id = last_stop_info['stop_id']
    current_lateness = 0  # Start with no lateness assumption

    for idx, next_stop_row in remaining_stops.iterrows():
        next_stop_id = next_stop_row['stop_id']

        # Extract temporal features
        hour_of_day = current_time.hour
        day_of_week = current_time.dayofweek
        is_weekend = 1 if day_of_week >= 5 else 0

        # Get current and next stop coordinates
        current_stop_coords = route_stops_df[route_stops_df['stop_id'] == current_stop_id].iloc[0]
        next_stop_coords = next_stop_row

        # Calculate segment distance (vectorized)
        segment_distance_km = haversine_vectorized(
            current_stop_coords['stop_lat'],
            current_stop_coords['stop_lon'],
            next_stop_coords['stop_lat'],
            next_stop_coords['stop_lon']
        )

        # Get historical features
        hist_feats = historical_stats[
            (historical_stats['stop_id_A'] == current_stop_id) &
            (historical_stats['stop_id_B'] == next_stop_id) &
            (historical_stats['is_weekend'] == is_weekend)
        ]

        # Get ideal time
        ideal_time_segment = ideal_stats[
            (ideal_stats['stop_id_A'] == current_stop_id) &
            (ideal_stats['stop_id_B'] == next_stop_id)
        ]

        # Handle missing historical data with fallbacks
        if hist_feats.empty or ideal_time_segment.empty:
            # Fallback: estimate based on distance (40 km/h average speed)
            predicted_time = (segment_distance_km / 40) * 3600  # seconds
            avg_time_s = predicted_time
            median_time_s = predicted_time
            ideal_time_s = predicted_time * 0.8  # Ideal is 80% of average
        else:
            avg_time_s = hist_feats['avg_time_s'].iloc[0]
            median_time_s = hist_feats['median_time_s'].iloc[0]
            ideal_time_s = ideal_time_segment['ideal_time_s'].iloc[0]

            # Create feature vector
            feature_vector = pd.DataFrame([{
                'hour_of_day': hour_of_day,
                'day_of_week': day_of_week,
                'is_weekend': is_weekend,
                'segment_distance_km': segment_distance_km,
                'avg_time_s': avg_time_s,
                'median_time_s': median_time_s,
                'ideal_time_s': ideal_time_s,
                'lateness_seconds': current_lateness
            }])

            # Predict
            predicted_time = model.predict(feature_vector)[0]

        # Update current time
        current_time += pd.Timedelta(seconds=predicted_time)

        # Update lateness for next iteration
        current_lateness += (predicted_time - ideal_time_s)

        # Calculate ETA in seconds from the last known timestamp
        eta_seconds = int(round((current_time - last_timestamp).total_seconds()))

        predictions.append({
            "stop_id": int(next_stop_id),
            "stop_sequence": next_stop_row['stop_sequence'],
            "eta": eta_seconds
        })

        # Move to next segment
        current_stop_id = next_stop_id

    # --- 7. Sort predictions by stop_sequence ---
    predictions.sort(key=lambda x: x['stop_sequence'])

    # --- 8. Enforce non-decreasing timestamps with 2-second minimum gap ---
    previous_time = last_timestamp
    for pred in predictions:
        arrival_time = last_timestamp + pd.Timedelta(seconds=pred['eta'])
        # Ensure timestamp is at least previous_time + 2 seconds
        min_allowed_time = previous_time + pd.Timedelta(seconds=2)
        if arrival_time < min_allowed_time:
            arrival_time = min_allowed_time
        pred['arrival_time'] = arrival_time
        previous_time = arrival_time

    # --- 9. Format Output (Submission Format) ---
    # Convert route_id to string as per submission guidelines
    route_id_str = str(int(route_id))

    # Create stop_id -> timestamp mapping
    stop_predictions = {}
    for pred in predictions:
        # Format as ISO-8601 UTC string: "YYYY-MM-DDTHH:MM:SSZ"
        timestamp_str = pred['arrival_time'].strftime('%Y-%m-%dT%H:%M:%SZ')
        stop_predictions[str(pred['stop_id'])] = timestamp_str

    # Final output format: {route_id: {stop_id: timestamp}}
    output = {route_id_str: stop_predictions}

    return json.dumps(output, indent=2)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_v2.py <test_file_path> [artifacts_dir]", file=sys.stderr)
        sys.exit(1)

    test_file_path = sys.argv[1]
    artifacts_dir = sys.argv[2] if len(sys.argv) > 2 else "artifacts_v2"

    try:
        result = predict_eta(test_file_path, artifacts_dir)
        print(result)  # JSON output to stdout for standalone usage
    except Exception as e:
        print(f"Error during prediction: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
