# IoT Predictive Maintenance System

A Spring Boot backend for real-time predictive maintenance using machine learning models.

## Overview

This application provides a complete predictive maintenance solution that:

1. Loads pre-trained ML models for anomaly detection, failure prediction, health index estimation, and part risk prediction
2. Fetches real-time sensor data from Supabase via REST API
3. Preprocesses the data (normalization and sequence creation)
4. Runs inference using the ML models
5. Stores prediction results back in Supabase

## Technologies

- **Backend**: Java 21, Spring Boot 3.2.3
- **ML Libraries**: DeepLearning4j (LSTM models), Weka (Random Forest models)
- **Database**: Supabase (PostgreSQL)
- **API Connectivity**: Spring RestTemplate for Supabase REST API
- **Build Tool**: Maven

## Project Structure

```
ml-backend/
├── src/
│   ├── main/
│   │   ├── java/
│   │   │   └── com/example/
│   │   │       ├── mlbackend/
│   │   │       │   ├── config/               # Configuration classes
│   │   │       │   ├── controller/           # REST controllers
│   │   │       │   ├── model/                # Data models
│   │   │       │   ├── service/              # Business logic
│   │   │       │   └── MlBackendApplication.java
│   │   │       ├── AutoencoderTrainer.java   # Training script for autoencoder
│   │   │       ├── FailurePredictionTrainer.java # Training script for failure model
│   │   │       ├── HealthIndexTrainer.java   # Training script for health index model
│   │   │       └── PartRiskTrainer.java      # Training script for part risk model
│   │   └── resources/
│   │       ├── model/                    # ML model files
│   │       ├── data/                     # Training data (not used at runtime)
│   │       └── application.properties    # Application configuration
└── pom.xml                               # Maven dependencies
```

## Key Files Description

### Main Application

- **MlBackendApplication.java**: The Spring Boot application entry point

### Controllers

- **PredictionController.java**: REST controller for manually triggering the prediction pipeline

### Services

- **ModelLoader.java**: Loads all ML models at startup and provides fallback implementations
- **DataPreprocessor.java**: Normalizes data and creates sequences for LSTM models
- **SupabaseApiService.java**: Handles data fetching from Supabase via REST API
- **InferenceService.java**: Runs predictions through all ML models
- **ResultStorageService.java**: Stores prediction results in Supabase
- **DataFetcherService.java**: High-level service for fetching sensor data
- **SchedulerService.java**: Schedules the prediction pipeline to run at regular intervals

### Models

- **SensorData.java**: Represents a single sensor data reading
- **PredictionResult.java**: Represents the combined output of all ML models

### Configuration

- **application.properties**: Contains configuration settings including:
  - Supabase connection details
  - Scheduler settings
  - Logging configuration

### Resource Files

- **autoencoder.model**: Deep learning model for anomaly detection (DL4J)
- **rf_failure.model**: Random Forest model for failure prediction (Weka)
- **rf_health_index.model**: Random Forest model for health index estimation (Weka)
- **part_risk.model**: DL4J model for predicting which part is at risk of failure
- **part_risk_normalizer.bin**: Normalizer for the part risk model input data
- **threshold.bin**: Threshold value for anomaly detection

## ML Models

The system uses four different ML models:

1. **Autoencoder (DL4J)**: Detects anomalies by comparing reconstruction error against a threshold
2. **Random Forest Classifier (Weka)**: Predicts probability of failure
3. **Random Forest Regressor (Weka)**: Estimates equipment health index (0-100)
4. **Deep Learning Network (DL4J)**: Predicts which part is at risk of failure (compressor, condenser, evaporator, expansion_valve, fan_motor, or none)

For detailed information about the ML models, see [MLDescription.md](MLDescription.md).

## Running the Application

For detailed instructions on running the application, see [RUNNING.md](RUNNING.md).

## API Endpoints

- `POST /api/predictions/run-pipeline`: Manually trigger the prediction pipeline

## Prediction Output

Each prediction includes:

- `anomaly` (boolean): Indicates if current data shows abnormal patterns
- `failure_prob` (0.0-1.0): Probability of imminent failure
- `health_index` (0-100): Equipment health score (higher is better)
- `part_at_risk` (string): Identifies which component is most likely to fail (compressor, condenser, evaporator, expansion_valve, fan_motor, or none)

Example output:

```
Stored prediction result: anomaly=true, failure_prob=0.0, health_index=46.77, part_at_risk=compressor
```

## Extending the Application

To extend or modify the application:

- **Add New ML Models**: Update the `ModelLoader` class to load additional models
- **Change Scheduling Frequency**: Modify the `SCHEDULE_RATE` environment variable
- **Add New APIs**: Create new controllers in the `controller` package
- **Add More Features**: Modify the `SensorData` class to include additional sensor metrics
