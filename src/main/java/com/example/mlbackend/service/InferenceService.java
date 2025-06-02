package com.example.mlbackend.service;

import com.example.mlbackend.model.PredictionResult;
import com.example.mlbackend.model.SensorData;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.springframework.stereotype.Service;
import weka.classifiers.Classifier;
import weka.core.Instance;

import java.util.List;

@Slf4j
@Service
@RequiredArgsConstructor
public class InferenceService {

    private final ModelLoader modelLoader;
    private final DataPreprocessor dataPreprocessor;

    /**
     * Run all predictions on sensor data and return results
     */
    public PredictionResult runInference(List<SensorData> sensorDataList) {
        try {
            if (sensorDataList.isEmpty()) {
                throw new IllegalArgumentException("Sensor data list cannot be empty");
            }

            // Get the latest sensor data for Weka models
            SensorData latestData = sensorDataList.get(sensorDataList.size() - 1);
            
            // Create a sequence for LSTM models (ascending order) 
            INDArray sequence = dataPreprocessor.createSequence(sensorDataList);
            
            // 1. Anomaly Detection with Autoencoder
            boolean isAnomaly = detectAnomaly(sequence);
            
            // 2. Failure Prediction with Random Forest
            double failureProbability = predictFailureProbability(latestData);
            
            // 3. Health Index Prediction with Random Forest
            double healthIndex = predictHealthIndex(latestData);
            
            // 4. RUL Prediction with LSTM
            double remainingUsefulLife = predictRUL(sequence);
            
            // 5. Part Risk Prediction with DL4J
            String[] partRiskInfo = predictPartAtRisk(latestData);
            String partAtRisk = partRiskInfo[0];
            String condition = partRiskInfo[1];
            
            // Build and return the prediction result
            return PredictionResult.builder()
                    .deviceId(latestData.getDeviceId())
                    .timestamp(latestData.getTimestamp())
                    .isAnomaly(isAnomaly)
                    .failureProbability(failureProbability)
                    .healthIndex(healthIndex)
                    .remainingUsefulLife(remainingUsefulLife)
                    .partAtRisk(partAtRisk)
                    .condition(condition)
                    .build();
        } catch (Exception e) {
            log.error("Error during inference: {}", e.getMessage(), e);
            throw new RuntimeException("Failed to run inference", e);
        }
    }

    /**
     * Detect anomalies using the autoencoder model
     */
    private boolean detectAnomaly(INDArray sequence) {
        try {
            // Get the autoencoder model and threshold
            double threshold = modelLoader.getThreshold();
            
            // Get the latest data point (last in sequence) and reshape to 2D for autoencoder
            long seqLength = sequence.size(1);
            INDArray latestFeatures = sequence.get(
                    NDArrayIndex.point(0), 
                    NDArrayIndex.point(seqLength - 1),
                    NDArrayIndex.all());
            
            // Reshape from [1, 11] to [1, 11] (removing the extra dimension)
            INDArray reshapedInput = latestFeatures.reshape(1, 11);
            
            log.debug("Autoencoder input shape: {}", reshapedInput.shape());
            
            // Get the reconstruction
            INDArray output = modelLoader.getAutoencoderModel().output(reshapedInput);
            
            // Calculate Mean Squared Error (MSE)
            INDArray mseArray = reshapedInput.sub(output).mul(reshapedInput.sub(output)).sum(1).div(11);
            double mse = mseArray.getDouble(0);
            
            log.debug("Anomaly MSE: {}, Threshold: {}", mse, threshold);
            
            // If MSE > threshold, it's an anomaly
            return mse > threshold;
        } catch (Exception e) {
            log.error("Error in anomaly detection: {}", e.getMessage(), e);
            throw new RuntimeException("Anomaly detection failed", e);
        }
    }

    /**
     * Predict failure probability using Random Forest model
     */
    private double predictFailureProbability(SensorData sensorData) {
        try {
            // Create a Weka instance for failure prediction
            Instance instance = dataPreprocessor.createInstance(sensorData, true);
            
            // Get the failure model
            Classifier failureModel = modelLoader.getFailureModel();
            
            // Get class probability distribution (class 1 = failure)
            double[] distribution = failureModel.distributionForInstance(instance);
            
            // Return probability of class 1 (failure)
            return distribution[1];
        } catch (Exception e) {
            log.error("Error in failure prediction: {}", e.getMessage(), e);
            throw new RuntimeException("Failure prediction failed", e);
        }
    }

    /**
     * Predict health index using Random Forest regression model
     */
    private double predictHealthIndex(SensorData sensorData) {
        try {
            // Create a Weka instance for health index prediction
            Instance instance = dataPreprocessor.createInstance(sensorData, false);
            
            // Get the health index model
            Classifier healthIndexModel = modelLoader.getHealthIndexModel();
            
            // Classify instance (regression value)
            return healthIndexModel.classifyInstance(instance);
        } catch (Exception e) {
            log.error("Error in health index prediction: {}", e.getMessage(), e);
            throw new RuntimeException("Health index prediction failed", e);
        }
    }

    /**
     * Predict Remaining Useful Life (RUL) using LSTM model
     */
    private double predictRUL(INDArray sequence) {
        try {
            // Check if sequence has the correct dimensions for the RUL model
            long seqLength = sequence.size(1);
            if (seqLength != 11) {
                log.warn("RUL model expects sequence length 11, but got {}. Using fallback value.", seqLength);
                // Return a conservative default value (e.g., 50% of typical RUL)
                return 500.0;
            }
            
            // Get the RUL model
            INDArray output = modelLoader.getRulModel().output(sequence);
            
            // Get the RUL value (single scalar output)
            return output.getDouble(0, 0);
        } catch (Exception e) {
            log.error("Error in RUL prediction: {}", e.getMessage(), e);
            log.warn("Using fallback RUL value due to model error");
            // Return a reasonable fallback value
            return 500.0;
        }
    }
    
    /**
     * Predict which part is at risk using the Part Risk DL4J model
     * @param sensorData The latest sensor data
     * @return String array with [partAtRisk, condition]
     */
    private String[] predictPartAtRisk(SensorData sensorData) {
        try {
            // Get the part risk model
            if (modelLoader.getPartRiskModel() == null) {
                log.warn("Part risk model not available. Using fallback values.");
                return new String[] {"unknown", "normal"};
            }
            
            // Create input features array from sensor data
            // We need all 11 features from SensorData plus a derived feature for the part risk model
            double[] features = new double[12];
            features[0] = sensorData.getFeature1();  // evaporator_coil_temperature
            features[1] = sensorData.getFeature2();  // fridge_temperature
            features[2] = sensorData.getFeature3();  // freezer_temperature
            features[3] = sensorData.getFeature4();  // air_temperature
            features[4] = sensorData.getFeature5();  // humidity
            features[5] = sensorData.getFeature6();  // compressor_vibration_x
            features[6] = sensorData.getFeature7();  // compressor_vibration_y
            features[7] = sensorData.getFeature8();  // compressor_vibration_z
            features[8] = sensorData.getFeature9();  // compressor_current
            features[9] = sensorData.getFeature10(); // input_voltage
            features[10] = sensorData.getFeature11(); // gas_leakage_level
            // Add a derived feature (e.g., average of vibration values) as the 12th feature
            features[11] = (features[5] + features[6] + features[7]) / 3.0; // Average vibration
            
            // Convert to INDArray
            INDArray input = Nd4j.create(features).reshape(1, 12);
            
            // Apply normalizer if available
            if (modelLoader.getPartRiskNormalizer() != null) {
                // Create a dataset with the input
                org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler normalizer = 
                    modelLoader.getPartRiskNormalizer();
                
                // We need to create a fake dataset with our input for normalization
                org.nd4j.linalg.dataset.DataSet dummyDataSet = 
                    new org.nd4j.linalg.dataset.DataSet(input, Nd4j.create(1, 6));
                
                // Apply normalization to features only
                normalizer.transform(dummyDataSet);
                
                // Get the normalized input
                input = dummyDataSet.getFeatures();
            }
            
            // Run prediction
            INDArray output = modelLoader.getPartRiskModel().output(input);
            
            // Get the predicted class (index of max value)
            int predictedClass = Nd4j.argMax(output, 1).getInt(0);
            
            // Map class index to part name
            String partAtRisk;
            String condition;
            
            switch (predictedClass) {
                case 0:
                    partAtRisk = "compressor";
                    condition = "warning";
                    break;
                case 1:
                    partAtRisk = "condenser";
                    condition = "warning";
                    break;
                case 2:
                    partAtRisk = "evaporator";
                    condition = "warning";
                    break;
                case 3:
                    partAtRisk = "expansion_valve";
                    condition = "warning";
                    break;
                case 4:
                    partAtRisk = "fan_motor";
                    condition = "warning";
                    break;
                case 5:
                    partAtRisk = "none";
                    condition = "normal";
                    break;
                default:
                    partAtRisk = "unknown";
                    condition = "normal";
                    break;
            }
            
            // Get the confidence score for the predicted class
            double confidence = output.getDouble(0, predictedClass);
            log.debug("Part risk prediction: {} with confidence {}", partAtRisk, confidence);
            
            return new String[] {partAtRisk, condition};
        } catch (Exception e) {
            log.error("Error in part risk prediction: {}", e.getMessage(), e);
            log.warn("Using fallback part risk values due to model error");
            // Return reasonable fallback values
            return new String[] {"unknown", "normal"};
        }
    }
} 