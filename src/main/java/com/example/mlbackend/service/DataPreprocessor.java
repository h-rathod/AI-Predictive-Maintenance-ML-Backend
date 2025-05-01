package com.example.mlbackend.service;

import com.example.mlbackend.model.SensorData;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.springframework.stereotype.Service;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.util.List;

/**
 * Service for preprocessing sensor data before ML model inference.
 * 
 * This service provides crucial data transformation functions including:
 * 1. Data normalization using pre-computed mean and standard deviation values
 * 2. Sequence creation for LSTM models (autoencoder and RUL prediction)
 * 3. Weka instance creation for Random Forest models (failure and health index)
 * 
 * Proper data preprocessing is essential for accurate model predictions.
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class DataPreprocessor {

    private final ModelLoader modelLoader;

    /**
     * Creates a normalized sequence of sensor data for LSTM models.
     * 
     * @param sensorDataList List of sensor data points in chronological order
     * @return INDArray with shape [1, sequence_length, features]
     */
    public INDArray createSequence(List<SensorData> sensorDataList) {
        int sequenceLength = sensorDataList.size();
        int featureCount = 11;
        
        // Create a 3D array: [batch_size=1, sequence_length, features=11]
        INDArray sequence = Nd4j.zeros(1, sequenceLength, featureCount);
        
        for (int i = 0; i < sequenceLength; i++) {
            SensorData data = sensorDataList.get(i);
            double[] features = data.getFeatureArray();
            double[] normalizedFeatures = normalizeFeatures(features);
            
            // Add normalized features to the sequence
            for (int j = 0; j < featureCount; j++) {
                sequence.putScalar(new int[] {0, i, j}, normalizedFeatures[j]);
            }
        }
        
        return sequence;
    }

    /**
     * Creates a Weka Instance for the Random Forest models.
     * 
     * @param sensorData Single sensor data reading
     * @param isFailureModel Whether to use the failure model header (true) or health index header (false)
     * @return Weka Instance ready for model prediction
     */
    public Instance createInstance(SensorData sensorData, boolean isFailureModel) {
        // Get the appropriate header
        Instances header = isFailureModel ? 
                modelLoader.getFailureHeader() : modelLoader.getHealthIndexHeader();
        
        // Create a new instance
        Instance instance = new DenseInstance(12);  // 11 features + 1 class
        instance.setDataset(header);
        
        // Normalize and set feature values
        double[] features = sensorData.getFeatureArray();
        double[] normalizedFeatures = normalizeFeatures(features);
        
        for (int i = 0; i < 11; i++) {
            instance.setValue(i, normalizedFeatures[i]);
        }
        
        // Class will be predicted, so we just set a dummy value
        instance.setMissing(11);
        
        return instance;
    }

    /**
     * Normalizes feature values using pre-computed mean and standard deviation.
     * 
     * @param features Raw feature values
     * @return Normalized feature values
     */
    private double[] normalizeFeatures(double[] features) {
        double[] normalizedFeatures = new double[features.length];
        double[] mean = modelLoader.getMean();
        double[] std = modelLoader.getStd();
        
        for (int i = 0; i < features.length; i++) {
            normalizedFeatures[i] = (features[i] - mean[i]) / std[i];
        }
        
        return normalizedFeatures;
    }
} 