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
 * PartRiskTrainerSimple - A simplified version of PartRiskTrainer that uses a direct approach
 * to load and process the CSV data, avoiding issues with timestamp parsing.
 */
public class PartRiskTrainerSimple {

    // Constants
    private static final int NUM_FEATURES = 12;
    private static final int NUM_CLASSES = 6;
    private static final int BATCH_SIZE = 32;
    private static final int NUM_EPOCHS = 10;
    private static final double LEARNING_RATE = 0.001;
    private static final int HIDDEN_LAYER_SIZE = 32;
    private static final int RANDOM_SEED = 123;
    private static final double TRAIN_RATIO = 0.8; // 80% for training, 20% for testing

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
            System.out.println("Starting PartRiskTrainerSimple...");
            
            // Create model directory if it doesn't exist
            Path modelDir = Paths.get("model");
            if (!Files.exists(modelDir)) {
                Files.createDirectories(modelDir);
            }

            // Load data from CSV file
            String dataFilePath = "data/part_risk_data.csv";
            System.out.println("Loading data from " + dataFilePath);
            
            // Read all lines from the CSV file
            List<String> lines = Files.readAllLines(Paths.get(dataFilePath));
            System.out.println("Read " + lines.size() + " lines from CSV file");
            
            // Skip header line
            lines = lines.subList(1, lines.size());
            
            // Prepare data containers
            List<double[]> featuresList = new ArrayList<>();
            List<Integer> labelsList = new ArrayList<>();
            
            // Process each line
            for (int i = 0; i < lines.size(); i++) {
                try {
                    String line = lines.get(i);
                    
                    // Split the line by comma, but handle quoted values
                    List<String> values = parseCSVLine(line);
                    
                    // Extract the 12 feature values (skipping timestamp at index 0)
                    double[] features = new double[NUM_FEATURES];
                    features[0] = Double.parseDouble(values.get(1));  // temperature_evaporator
                    features[1] = Double.parseDouble(values.get(2));  // temperature_internal
                    features[2] = Double.parseDouble(values.get(3));  // ambient_temperature
                    features[3] = Double.parseDouble(values.get(4));  // humidity_internal
                    features[4] = Double.parseDouble(values.get(5));  // pressure_refrigerant
                    features[5] = Double.parseDouble(values.get(6));  // current_compressor
                    features[6] = Double.parseDouble(values.get(7));  // vibration_level
                    features[7] = Double.parseDouble(values.get(8));  // gas_leak_level
                    features[8] = Double.parseDouble(values.get(10)); // compressor_cycle_time
                    features[9] = Double.parseDouble(values.get(11)); // energy_consumption
                    features[10] = Double.parseDouble(values.get(12)); // temperature_gradient
                    features[11] = Double.parseDouble(values.get(13)); // pressure_trend
                    
                    // Extract the label (part_at_risk at index 14)
                    String partAtRisk = values.get(14);
                    
                    // Skip if label is not recognized
                    if (!LABEL_TO_INDEX.containsKey(partAtRisk)) {
                        System.out.println("Warning: Unknown label '" + partAtRisk + "' at line " + (i + 1));
                        continue;
                    }
                    
                    // Add to our data lists
                    featuresList.add(features);
                    labelsList.add(LABEL_TO_INDEX.get(partAtRisk));
                    
                } catch (Exception e) {
                    System.out.println("Warning: Error processing line " + (i + 1) + ": " + e.getMessage());
                }
            }
            
            System.out.println("Successfully loaded " + featuresList.size() + " data points");
            
            // Convert lists to INDArrays for DL4J
            int numSamples = featuresList.size();
            INDArray featuresArray = Nd4j.create(numSamples, NUM_FEATURES);
            INDArray labelsArray = Nd4j.create(numSamples, NUM_CLASSES);
            
            for (int i = 0; i < numSamples; i++) {
                // Set features
                for (int j = 0; j < NUM_FEATURES; j++) {
                    featuresArray.putScalar(new int[]{i, j}, featuresList.get(i)[j]);
                }
                
                // Set one-hot encoded label
                labelsArray.putScalar(new int[]{i, labelsList.get(i)}, 1.0);
            }
            
            // Create DataSet
            DataSet allData = new DataSet(featuresArray, labelsArray);
            
            // Shuffle the data
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
     * Parse a CSV line handling quoted values
     * @param line The CSV line to parse
     * @return List of values from the CSV line
     */
    private static List<String> parseCSVLine(String line) {
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
        
        return result;
    }
}
