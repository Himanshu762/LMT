# Transit ETA Prediction System V2

This project is a solution for the [Bengaluru Last Mile Challenge 2025](https://ieee-dataport.org//competitions/bengaluru-last-mile-challenge-2025).

## Overview

The Transit ETA Prediction System V2 is designed to predict the Estimated Time of Arrival (ETA) for buses based on real-time GPS data. It utilizes a vectorized inference pipeline powered by an XGBoost model to provide accurate and efficient predictions.

## Features

-   **GPS-Based Prediction**: Uses real-time GPS coordinates to localize buses and predict arrival times at subsequent stops.
-   **Vectorized Inference**: Optimized for performance using vectorized operations with Pandas and NumPy.
-   **XGBoost Model**: Employs a trained XGBoost regressor for robust ETA estimation.
-   **Dockerized**: Fully containerized for easy deployment and reproducibility.

## Project Structure

-   `bus_eta_prediction/`: Contains the source code for the prediction system.
    -   `main.py`: The main entry point for the application. Handles input parsing and orchestration.
    -   `predict_v2.py`: Core prediction logic using the XGBoost model.
    -   `utils.py`: Utility functions for data cleaning and geospatial calculations.
    -   `Dockerfile`: Configuration for building the Docker image.
    -   `requirements.txt`: Python dependencies.
    -   `artifacts_v2/`: Directory containing trained models and route information (ensure this is populated).

## Usage

### Prerequisites

-   Docker installed on your machine.
-   Python 3.x (if running locally without Docker).

### Running with Docker

1.  **Build the Docker image:**

    ```bash
    cd bus_eta_prediction
    docker build -t bus-eta-prediction .
    ```

2.  **Run the container:**

    You need to mount the directory containing your input data and artifacts.

    ```bash
    docker run -v /path/to/data:/app/data -v /path/to/artifacts:/app/artifacts_v2 bus-eta-prediction
    ```

    *Note: The application expects an `input.json` file to specify the test data.*

### Running Locally

1.  **Install dependencies:**

    ```bash
    cd bus_eta_prediction
    pip install -r requirements.txt
    ```

2.  **Run the prediction script:**

    ```bash
    python main.py <path_to_input.json> <path_to_artifacts_dir>
    ```

    Or run the prediction module directly for a single file:

    ```bash
    python predict_v2.py <path_to_test_file.parquet> <path_to_artifacts_dir>
    ```

## Input Format

The `input.json` file should map identifiers to paths of Parquet files containing GPS data:

```json
{
  "1": "path/to/trip1.parquet",
  "2": "path/to/trip2.parquet"
}
```

## Output

The system generates an `output.json` file containing the predicted arrival timestamps for each stop in the route.

```json
{
  "route_id": {
    "stop_id": "YYYY-MM-DDTHH:MM:SSZ",
    ...
  }
}
```
