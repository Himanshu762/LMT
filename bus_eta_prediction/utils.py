"""
Utility functions for transit ETA prediction.
All functions are vectorized for performance.
"""

import numpy as np
import pandas as pd

def haversine_vectorized(lat1, lon1, lat2, lon2):
    """
    Vectorized Haversine distance calculation.

    Supports multiple modes:
    1. Element-wise: (N,) × (N,) → (N,) - distances between paired points
    2. Broadcasting: (N,) × (M,) → (N, M) - all pairwise distances
    3. Scalar: float × float → float - single distance

    Args:
        lat1, lon1: Latitude/longitude of first point(s) in degrees
        lat2, lon2: Latitude/longitude of second point(s) in degrees

    Returns:
        Distance(s) in kilometers
    """
    # Earth radius in kilometers
    R = 6371.0

    # Convert to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c


def match_gps_to_stops_vectorized(gps_df, route_stops_df):
    """
    Match GPS points to nearest stops on the route using vectorized operations.

    This is memory-efficient for large datasets by processing in chunks if needed.

    Args:
        gps_df: DataFrame with columns [latitude, longitude]
        route_stops_df: DataFrame with columns [stop_id, stop_lat, stop_lon, stop_sequence]

    Returns:
        Series of stop_ids (same length as gps_df)
    """
    # Extract coordinates as numpy arrays for speed
    gps_lats = gps_df['latitude'].values
    gps_lons = gps_df['longitude'].values

    stop_lats = route_stops_df['stop_lat'].values
    stop_lons = route_stops_df['stop_lon'].values
    stop_ids = route_stops_df['stop_id'].values

    # Calculate distance matrix: (N_gps, N_stops)
    # Using broadcasting: (N, 1) - (1, M) = (N, M)
    lat1 = gps_lats[:, np.newaxis]  # (N_gps, 1)
    lon1 = gps_lons[:, np.newaxis]  # (N_gps, 1)
    lat2 = stop_lats[np.newaxis, :]  # (1, N_stops)
    lon2 = stop_lons[np.newaxis, :]  # (1, N_stops)

    # Compute all distances at once
    distances = haversine_vectorized(lat1, lon1, lat2, lon2)  # (N_gps, N_stops)

    # Find nearest stop for each GPS point
    nearest_stop_indices = np.argmin(distances, axis=1)  # (N_gps,)
    nearest_stop_ids = stop_ids[nearest_stop_indices]

    return nearest_stop_ids


def detect_stop_arrivals_vectorized(trip_df, route_stops_df):
    """
    Detect when a bus arrives at each stop using vectorized operations.

    Args:
        trip_df: DataFrame for ONE trip with columns [timestamp, latitude, longitude, route_id]
                 Must be sorted by timestamp
        route_stops_df: DataFrame with route's stops [stop_id, stop_lat, stop_lon, stop_sequence]

    Returns:
        DataFrame with columns [stop_id, arrival_time, stop_sequence]
    """
    # Match each GPS ping to nearest stop
    trip_df = trip_df.copy()
    trip_df['nearest_stop_id'] = match_gps_to_stops_vectorized(trip_df, route_stops_df)

    # Detect stop changes (transitions from one stop to another)
    trip_df['stop_changed'] = trip_df['nearest_stop_id'] != trip_df['nearest_stop_id'].shift(1)

    # Group by stop_id and get first timestamp (arrival time)
    # This handles cases where bus stays at stop for multiple GPS pings
    arrivals = trip_df.groupby('nearest_stop_id', as_index=False).agg({
        'timestamp': 'first'
    }).rename(columns={'nearest_stop_id': 'stop_id', 'timestamp': 'arrival_time'})

    # Merge with route info to get stop sequence and validate
    arrivals = arrivals.merge(
        route_stops_df[['stop_id', 'stop_sequence']],
        on='stop_id',
        how='inner'
    )

    # Sort by sequence to ensure chronological order
    arrivals = arrivals.sort_values('stop_sequence').reset_index(drop=True)

    return arrivals


def create_segments_vectorized(arrivals_df):
    """
    Create stop-to-stop segments from arrival times using vectorized operations.

    Args:
        arrivals_df: DataFrame with [stop_id, arrival_time, stop_sequence]
                     Must be sorted by stop_sequence

    Returns:
        DataFrame with [stop_id_A, stop_id_B, departure_time_A, arrival_time_B, travel_time_seconds]
    """
    if len(arrivals_df) < 2:
        return pd.DataFrame(columns=['stop_id_A', 'stop_id_B', 'departure_time_A', 'arrival_time_B', 'travel_time_seconds'])

    # Create segments using shift
    segments = pd.DataFrame({
        'stop_id_A': arrivals_df['stop_id'].values[:-1],
        'stop_id_B': arrivals_df['stop_id'].values[1:],
        'departure_time_A': arrivals_df['arrival_time'].values[:-1],
        'arrival_time_B': arrivals_df['arrival_time'].values[1:],
        'stop_sequence_A': arrivals_df['stop_sequence'].values[:-1],
        'stop_sequence_B': arrivals_df['stop_sequence'].values[1:]
    })

    # Calculate travel time (vectorized)
    segments['travel_time_seconds'] = (
        segments['arrival_time_B'] - segments['departure_time_A']
    ).dt.total_seconds()

    # Filter reasonable travel times (10s to 2 hours)
    segments = segments[
        (segments['travel_time_seconds'] >= 10) &
        (segments['travel_time_seconds'] <= 7200)
    ]

    return segments


def clean_parquet_data(df):
    """
    Clean a parquet dataframe using vectorized string operations.

    Args:
        df: Raw parquet DataFrame

    Returns:
        Cleaned DataFrame
    """
    # Drop rows with missing critical data
    df = df.dropna(subset=['trip_id', 'route_id', 'vehicle_timestamp', 'latitude', 'longitude']).copy()

    # Vectorized string cleaning - remove quotes and whitespace
    df['schedule_relationship'] = df['schedule_relationship'].astype(str).str.strip().str.replace('"', '', regex=False)
    df['trip_id'] = df['trip_id'].astype(str).str.strip().str.replace('"', '', regex=False)
    df['route_id'] = df['route_id'].astype(str).str.strip().str.replace('"', '', regex=False)
    df['vehicle_timestamp'] = df['vehicle_timestamp'].astype(str).str.strip().str.replace('"', '', regex=False)

    # Filter for SCHEDULED trips
    df = df[df['schedule_relationship'] == 'SCHEDULED'].copy()

    # Convert route_id to integer
    df['route_id'] = pd.to_numeric(df['route_id'], errors='coerce')
    df = df.dropna(subset=['route_id'])
    df['route_id'] = df['route_id'].astype(int)

    # Convert timestamp (vectorized) - now that quotes are removed
    df['vehicle_timestamp'] = pd.to_numeric(df['vehicle_timestamp'], errors='coerce')
    df = df.dropna(subset=['vehicle_timestamp'])
    df['timestamp'] = pd.to_datetime(df['vehicle_timestamp'], unit='s', errors='coerce')
    df = df.dropna(subset=['timestamp'])

    # Sort by trip and time for efficient processing
    df = df.sort_values(['trip_id', 'timestamp']).reset_index(drop=True)

    return df[['trip_id', 'route_id', 'timestamp', 'latitude', 'longitude']]
