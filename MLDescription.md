# Machine Learning Models Description

This document provides detailed information about the machine learning models used in the IoT Predictive Maintenance System.

## Overview

The system uses four different machine learning models to provide a comprehensive predictive maintenance solution:

1. **Anomaly Detection Model (Autoencoder)**
2. **Failure Prediction Model (Random Forest)**
3. **Health Index Estimation Model (Random Forest)**
4. **Part Risk Prediction Model (Neural Network)**

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

## 4. Part Risk Prediction Model

### Technology: Neural Network (DL4J)

The Part Risk prediction model is implemented as a multi-class classification neural network using the DeepLearning4J library. This model identifies which specific refrigerator component is most likely to fail based on sensor data patterns.

### How It Works

1. **Input**: The model takes 12 numeric sensor features (normalized).
2. **Architecture**: The model consists of:
   - Input Layer: 12 units (one per feature)
   - Hidden Layer: 32 units with ReLU activation
   - Output Layer: 6 units with softmax activation for multi-class classification
3. **Training**: During training, the model learns to classify which part is at risk based on sensor patterns.
4. **Inference**: The model outputs the most likely part at risk along with condition severity and the associated sensor column.

### Target Classes (Parts at Risk)

- Evaporator Coil
- Freezer Compartment
- Fridge Section
- Fresh Food Section
- Compressor
- Refrigerant System

### Condition Labels

- **Critical**: Immediate action needed (e.g., Compressor vibration > 0.7 mm/s)
- **Warning**: Potential issue (e.g., Compressor vibration > 0.5 mm/s but ≤ 0.7 mm/s)
- **Good**: No issues detected

### Input Features

The model uses 12 numeric columns from `part_risk_data.csv`:

- `temperature_evaporator` (°C, Evaporator Coil)
- `temperature_internal` (°C, Freezer Compartment)
- `ambient_temperature` (°C, Fridge Section)
- `humidity_internal` (%, Fresh Food Section)
- `pressure_refrigerant` (kPa, Refrigerant System)
- `current_compressor` (A, Compressor)
- `vibration_level` (mm/s, Compressor)
- `gas_leak_level` (ppm, Refrigerant System)
- `compressor_cycle_time` (seconds)
- `energy_consumption` (kWh)
- `temperature_gradient` (°C/h)
- `pressure_trend` (kPa/h)

### Model Files

- **part_risk.model**: The serialized DeepLearning4J neural network for part risk prediction
- **part_risk_normalizer.bin**: Normalizer for scaling input features to [0, 1]

### Training Process

The Part Risk model is trained using the `PartRiskTrainer.java` class, which:

1. Loads the `part_risk_data.csv` dataset (10,000 rows) with labeled examples
2. Applies `NormalizerMinMaxScaler` to scale features to [0, 1]
3. Splits the data into training (80%) and testing (20%) sets
4. Configures the neural network architecture
5. Trains the model using multi-class cross-entropy loss and Adam optimizer
6. Evaluates the model on the test set (expected accuracy ~93%)
7. Saves both the trained model and normalizer to disk

```java
// Key training parameters
int numEpochs = 10;
double learningRate = 0.001;
int batchSize = 32;
int hiddenLayerSize = 32;
```

## Additional Model Files

- **mean.bin** and **std.bin**: Binary files containing the mean and standard deviation values used for normalizing the input data
- **failure_header.model** and **health_index_header.model**: Weka Instances header files that define the attributes and structure expected by the Weka models
- **part_risk_normalizer.bin**: Normalizer for the part risk model input data

## Data Preprocessing

All models require preprocessing of raw sensor data:

1. **Normalization**: Raw sensor values are normalized using appropriate techniques:
   
   For Autoencoder:
   ```
   normalized_value = (raw_value - mean) / std
   ```
   
   For Part Risk model:
   ```
   normalized_value = (raw_value - min) / (max - min)  // MinMaxScaler
   ```

2. **Sequence Creation**: For the LSTM-based Autoencoder model, sequences of consecutive readings are created to capture temporal patterns.

## Model Integration

The four models work together to provide a comprehensive view of equipment health:

1. The **Anomaly Detection** model identifies unusual behavior
2. The **Failure Prediction** model estimates failure probability
3. The **Health Index** model provides a continuous measure of equipment condition
4. The **Part Risk** model identifies which specific component is most likely to fail

Together, these models enable maintenance teams to make data-driven decisions about when and how to perform maintenance activities, with actionable insights about which specific parts require attention.

## Important Implementation Note

The Part Risk model provides more actionable insights than the previous RUL model by:

1. Identifying the specific refrigerator part at risk (e.g., Compressor instead of a general RUL)
2. Assessing the severity (Critical, Warning, Good) for prioritization
3. Linking the prediction to a specific sensor column for targeted maintenance

This makes the system's outputs more interpretable and actionable for end-users.

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
