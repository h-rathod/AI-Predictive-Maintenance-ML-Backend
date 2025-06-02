package com.example;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.dataset.api.preprocessor.serializer.NormalizerSerializer;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * PartRiskTrainer - Trains a lightweight DL4J model to predict which refrigerator part is at risk
 * based on sensor data. The model uses 12 numeric features to classify into 6 parts (multi-class classification).
 */
public class PartRiskTrainer {

    // Constants
    private static final int NUM_FEATURES = 12;
    private static final int NUM_CLASSES = 6;
    private static final int BATCH_SIZE = 32;
    private static final int NUM_EPOCHS = 10;
    private static final double LEARNING_RATE = 0.001;
    private static final int HIDDEN_LAYER_SIZE = 32;
    private static final int RANDOM_SEED = 123;
    private static final double TRAIN_RATIO = 0.8; // 80% for training, 20% for testing
    
    // Feature indices in the CSV file (skipping timestamp at index 0)
    private static final int[] FEATURE_INDICES = {
            1,  // temperature_evaporator
            2,  // temperature_internal
            3,  // ambient_temperature
            4,  // humidity_internal
            5,  // pressure_refrigerant
            6,  // current_compressor
            7,  // vibration_level
            8,  // gas_leak_level
            10, // compressor_cycle_time
            11, // energy_consumption
            12, // temperature_gradient
            13  // pressure_trend
    };
    
    // Label index in the CSV file
    private static final int LABEL_INDEX = 14; // part_at_risk
    
    // Class labels for the parts at risk
    private static final List<String> CLASS_LABELS = Arrays.asList(
            "Evaporator Coil", "Freezer Compartment", "Fridge Section", 
            "Fresh Food Section", "Compressor", "Refrigerant System");
    
    // Map to convert class labels to indices
    private static final Map<String, Integer> LABEL_TO_INDEX = new HashMap<>();
    
    static {
        for (int i = 0; i < CLASS_LABELS.size(); i++) {
            LABEL_TO_INDEX.put(CLASS_LABELS.get(i), i);
        }
    }

    public static void main(String[] args) {
        try {
            System.out.println("Starting PartRiskTrainer...");
            
            // Create model directory if it doesn't exist
            Path modelDir = Paths.get("model");
            if (!Files.exists(modelDir)) {
                Files.createDirectories(modelDir);
            }

            // Load data from CSV file manually
            String dataFilePath = "data/part_risk_data.csv";
            System.out.println("Loading data from " + dataFilePath);
            
            List<double[]> features = new ArrayList<>();
            List<Integer> labels = new ArrayList<>();
            
            try (BufferedReader br = new BufferedReader(new FileReader(dataFilePath))) {
                // Skip header line
                br.readLine();
                
                String line;
                int lineCount = 0;
                
                // Process each line
                while ((line = br.readLine()) != null) {
                    lineCount++;
                    try {
                        // Split by comma but handle quoted values
                        String[] values = splitCSVLine(line);
                        
                        // Extract features (12 numeric values)
                        double[] featureArray = new double[NUM_FEATURES];
                        for (int i = 0; i < NUM_FEATURES; i++) {
                            featureArray[i] = Double.parseDouble(values[FEATURE_INDICES[i]]);
                        }
                        features.add(featureArray);
                        
                        // Extract label (part_at_risk)
                        String partAtRisk = values[LABEL_INDEX];
                        if (!LABEL_TO_INDEX.containsKey(partAtRisk)) {
                            System.out.println("Warning: Unknown label '" + partAtRisk + "' at line " + lineCount);
                            continue;
                        }
                        labels.add(LABEL_TO_INDEX.get(partAtRisk));
                    } catch (Exception e) {
                        System.out.println("Warning: Error processing line " + lineCount + ": " + e.getMessage());
                    }
                }
            }
            
            System.out.println("Loaded " + features.size() + " data points");
            
            // Convert to INDArrays
            int numSamples = features.size();
            INDArray featuresArray = Nd4j.create(numSamples, NUM_FEATURES);
            INDArray labelsArray = Nd4j.create(numSamples, NUM_CLASSES);
            
            for (int i = 0; i < numSamples; i++) {
                // Set features
                for (int j = 0; j < NUM_FEATURES; j++) {
                    featuresArray.putScalar(new int[]{i, j}, features.get(i)[j]);
                }
                
                // Set one-hot encoded label
                labelsArray.putScalar(new int[]{i, labels.get(i)}, 1.0);
            }
            
            // Create DataSet
            DataSet allData = new DataSet(featuresArray, labelsArray);
            
            // Shuffle the data (important for training)
            allData.shuffle(RANDOM_SEED);
            
            // Normalize features using min-max scaling
            NormalizerMinMaxScaler normalizer = new NormalizerMinMaxScaler();
            normalizer.fit(allData);
            normalizer.transform(allData);
            
            // Split data into training and test sets
            SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(TRAIN_RATIO);
            DataSet trainingData = testAndTrain.getTrain();
            DataSet testData = testAndTrain.getTest();
            
            System.out.println("Training data size: " + trainingData.numExamples());
            System.out.println("Test data size: " + testData.numExamples());
            
            // Configure neural network
            MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                    .seed(RANDOM_SEED)
                    .weightInit(WeightInit.XAVIER)
                    .updater(new Adam(LEARNING_RATE))
                    .l2(1e-4)
                    .list()
                    .layer(0, new DenseLayer.Builder()
                            .nIn(NUM_FEATURES)
                            .nOut(HIDDEN_LAYER_SIZE)
                            .activation(Activation.RELU)
                            .build())
                    .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                            .nIn(HIDDEN_LAYER_SIZE)
                            .nOut(NUM_CLASSES)
                            .activation(Activation.SOFTMAX)
                            .build())
                    .build();
            
            // Initialize model
            MultiLayerNetwork model = new MultiLayerNetwork(config);
            model.init();
            model.setListeners(new ScoreIterationListener(100));
            
            // Train model
            System.out.println("Training model...");
            for (int i = 0; i < NUM_EPOCHS; i++) {
                model.fit(trainingData);
                System.out.println("Completed epoch " + (i+1) + "/" + NUM_EPOCHS);
            }
            
            // Evaluate model
            System.out.println("Evaluating model...");
            Evaluation eval = new Evaluation(NUM_CLASSES);
            INDArray output = model.output(testData.getFeatures());
            eval.eval(testData.getLabels(), output);
            
            System.out.println("Training complete");
            System.out.println(eval.stats());
            System.out.println("Accuracy: " + eval.accuracy());
            
            // Save model and normalizer
            File modelFile = new File("model/part_risk.model");
            ModelSerializer.writeModel(model, modelFile, true);
            
            // Save normalizer
            File normalizerFile = new File("model/part_risk_normalizer.bin");
            NormalizerSerializer.getDefault().write(normalizer, normalizerFile);
            
            System.out.println("Saved part_risk.model to " + modelFile.getAbsolutePath());
            System.out.println("Saved part_risk_normalizer.bin to " + normalizerFile.getAbsolutePath());
            
        } catch (Exception e) {
            System.err.println("Error training model: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * Split a CSV line handling quoted values
     * @param line The CSV line to split
     * @return Array of values from the CSV line
     */
    private static String[] splitCSVLine(String line) {
        List<String> result = new ArrayList<>();
        StringBuilder currentValue = new StringBuilder();
        boolean inQuotes = false;
        
        for (int i = 0; i < line.length(); i++) {
            char c = line.charAt(i);
            
            if (c == '"') {
                // Toggle the inQuotes flag
                inQuotes = !inQuotes;
            } else if (c == ',' && !inQuotes) {
                // End of value
                result.add(currentValue.toString().trim());
                currentValue = new StringBuilder();
            } else {
                // Add character to current value
                currentValue.append(c);
            }
        }
        
        // Add the last value
        result.add(currentValue.toString().trim());
        
        return result.toArray(new String[0]);
    }
    

}
