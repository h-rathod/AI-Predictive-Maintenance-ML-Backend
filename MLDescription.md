# Machine Learning Models Description

This document provides detailed information about the machine learning models used in the IoT Predictive Maintenance System.

## Overview

The system uses four different machine learning models to provide a comprehensive predictive maintenance solution:

1. **Anomaly Detection Model (Autoencoder)**
2. **Failure Prediction Model (Random Forest)**
3. **Health Index Estimation Model (Random Forest)**
4. **Remaining Useful Life (RUL) Prediction Model (LSTM)**

Each model serves a specific purpose in the predictive maintenance pipeline and together they provide a comprehensive view of equipment health and potential failure conditions.

## 1. Anomaly Detection Model

### Technology: Deep Learning Autoencoder (DL4J)

The anomaly detection model is implemented as a deep learning autoencoder using the DeepLearning4J library. An autoencoder is a type of neural network that learns to encode input data into a lower-dimensional representation and then decode it back to the original dimensions.

### How It Works

1. **Input**: The model takes a single sensor data reading with 11 features (normalized).
2. **Architecture**: The model consists of:
   - LSTM encoder layers (11 → 32 → 16 → 16)
   - LSTM decoder layers (16 → 32 → 11)
3. **Training**: During training, the model learns to reconstruct normal operating conditions.
4. **Inference**: For anomaly detection:
   - The model attempts to reconstruct the input
   - The Mean Squared Error (MSE) between the input and reconstruction is calculated
   - If MSE > threshold, the data point is flagged as an anomaly

### Model Files

- **autoencoder.model**: The serialized DeepLearning4J LSTM autoencoder model (~3.7KB)
- **threshold.bin**: A binary file containing the MSE threshold value (84B)

### Training Process

The autoencoder is trained using the `AutoencoderTrainer.java` class, which:

1. Loads a training dataset of normal operational data
2. Normalizes the data using mean and standard deviation
3. Configures the LSTM autoencoder network architecture
4. Trains the model using backpropagation with MSE loss function
5. Establishes an anomaly threshold by evaluating reconstruction error on a validation set
6. Saves both the trained model and the threshold value

```java
// Key training parameters
int numEpochs = 100;
double learningRate = 0.01;
double regularization = 0.0001;
```

## 2. Failure Prediction Model

### Technology: Random Forest Classifier (Weka)

The failure prediction model is implemented as a Random Forest classifier using the Weka machine learning library. Random Forest is an ensemble learning method that constructs multiple decision trees during training.

### How It Works

1. **Input**: The model takes a single sensor data reading with 11 features (normalized).
2. **Architecture**: An ensemble of decision trees (typically 100-500 trees).
3. **Training**: During training, the model learns patterns associated with equipment failures.
4. **Inference**: The model outputs a probability between 0 and 1 representing the likelihood of failure.

### Model Files

- **rf_failure.model**: The serialized Weka Random Forest classifier (~21MB)

### Training Process

The failure prediction model is trained using the `FailurePredictionTrainer.java` class, which:

1. Loads a labeled dataset with examples of both normal operation and failure conditions
2. Processes the data into a Weka Instances object with appropriate attributes
3. Configures a Random Forest classifier with optimized hyperparameters
4. Trains the model using the labeled data
5. Evaluates the model using cross-validation
6. Saves the trained model to disk

```java
// Key training parameters
RandomForest rf = new RandomForest();
rf.setNumIterations(200);  // 200 trees in the forest
rf.setMaxDepth(10);        // Maximum depth of each tree
rf.setSeed(42);            // Random seed for reproducibility
```

## 3. Health Index Estimation Model

### Technology: Random Forest Regressor (Weka)

The health index estimation model is implemented as a Random Forest regressor using the Weka machine learning library. Unlike the failure prediction model which is a classifier, this model performs regression to estimate a continuous health value.

### How It Works

1. **Input**: The model takes a single sensor data reading with 11 features (normalized).
2. **Architecture**: An ensemble of regression trees (typically 100-500 trees).
3. **Training**: During training, the model learns to correlate sensor patterns with health indices.
4. **Inference**: The model outputs a numerical value representing the health index (typically 0-100).

### Model Files

- **rf_health_index.model**: The serialized Weka Random Forest regressor (~89MB)

### Training Process

The health index model is trained using the `HealthIndexTrainer.java` class, which:

1. Loads a dataset with examples mapping sensor readings to health index values
2. Processes the data into a Weka Instances object with numeric class attribute
3. Configures a Random Forest for regression with optimized hyperparameters
4. Trains the model using the labeled data
5. Evaluates the model using metrics like Mean Absolute Error (MAE)
6. Saves the trained model to disk

```java
// Key training parameters
RandomForest rf = new RandomForest();
rf.setNumIterations(300);  // 300 trees in the forest
rf.setMaxDepth(15);        // Deeper trees for more precise regression
rf.setSeed(42);            // Random seed for reproducibility
```

## 4. Remaining Useful Life (RUL) Prediction Model

### Technology: LSTM Neural Network (DL4J)

The RUL prediction model is implemented as a Long Short-Term Memory (LSTM) neural network using the DeepLearning4J library. LSTM networks are specifically designed to handle sequential data and can capture temporal dependencies.

### How It Works

1. **Input**: The model takes a sequence of 11 sensor data readings (normalized).
2. **Architecture**: The model consists of:
   - LSTM layers (11 → 64 → 32)
   - Dense layer (32 → 16)
   - Output layer (16 → 1)
3. **Training**: During training, the model learns to predict RUL from sequences of sensor readings.
4. **Inference**: The model outputs a single value representing the estimated remaining useful life in appropriate units (hours, days, cycles, etc.).

### Model Files

- **rul.model**: The serialized DeepLearning4J LSTM network for RUL prediction (~31KB)

### Training Process

The RUL model is trained using the `RulTrainer.java` class, which:

1. Loads a dataset with sequential sensor readings and their corresponding RUL values
2. Normalizes the data using global mean and standard deviation
3. Creates sequences of specified length (11 time steps)
4. Configures the LSTM network architecture
5. Trains the model using backpropagation with MSE loss function
6. Evaluates the model on a test set
7. Saves the trained model to disk

```java
// Key training parameters
int numEpochs = 200;
double learningRate = 0.005;
int batchSize = 32;
int timeSeriesLength = 11;  // Critical - must match at inference time
```

## Additional Model Files

- **mean.bin** and **std.bin**: Binary files containing the mean and standard deviation values used for normalizing the input data
- **failure_header.model** and **health_index_header.model**: Weka Instances header files that define the attributes and structure expected by the Weka models

## Data Preprocessing

All models require preprocessing of raw sensor data:

1. **Normalization**: Raw sensor values are normalized using pre-calculated mean and standard deviation values:

   ```
   normalized_value = (raw_value - mean) / std
   ```

2. **Sequence Creation**: For LSTM models (Autoencoder and RUL), sequences of consecutive readings are created to capture temporal patterns.

## Model Integration

The four models work together to provide a comprehensive view of equipment health:

1. The **Anomaly Detection** model identifies unusual behavior
2. The **Failure Prediction** model estimates failure probability
3. The **Health Index** model provides a continuous measure of equipment condition
4. The **RUL** model estimates remaining operational time

Together, these models enable maintenance teams to make data-driven decisions about when and how to perform maintenance activities.

## Important Implementation Note

The RUL model expects exactly 11 data points in the sequence. When running the pipeline, ensure that 11 consecutive sensor readings are fetched (as configured in `DataFetcherService.java`) to avoid sequence length mismatch errors.

## Model Performance

When properly trained on representative data, these models can achieve:

- **Anomaly Detection**: 90-95% accuracy in detecting abnormal conditions
- **Failure Prediction**: 85-90% accuracy in predicting failures
- **Health Index**: Strong correlation with actual equipment degradation
- **RUL Prediction**: Mean Absolute Error of less than 10% of the typical equipment lifetime

## Fallback Mechanisms

The system includes fallback mechanisms to ensure robustness:

- If the RUL model encounters a sequence length mismatch, it returns a conservative default value (500.0)
- If any model fails to load, a simple fallback model is created to provide basic functionality
- If a model throws an exception during inference, graceful error handling ensures the application continues running

## Future Improvements

Potential improvements to the ML models include:

1. **Transfer Learning**: Fine-tuning models with equipment-specific data
2. **Multivariate Models**: Incorporating additional data sources like maintenance history
3. **Ensemble Approaches**: Combining multiple model types for each prediction task
4. **Explainable AI**: Adding interpretability to model predictions
5. **Online Learning**: Updating models in real-time based on new data
