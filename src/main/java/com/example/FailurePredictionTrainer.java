package com.example;

import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.core.SerializationHelper;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public class FailurePredictionTrainer {
    public static void main(String[] args) {
        try {
            System.out.println("Starting FailurePredictionTrainer...");

            // Load failure_prediction_data.csv
            System.out.println("Loading failure_prediction_data.csv...");
            CSVLoader loader = new CSVLoader();
            loader.setSource(new File("data/failure_prediction_data.csv"));
            Instances data = loader.getDataSet();
            System.out.println("Loaded " + data.numInstances() + " instances.");

            // Set class attribute (failure)
            System.out.println("Setting class attribute...");
            data.setClassIndex(data.numAttributes() - 1); // Last column: failure

            // Train Random Forest
            System.out.println("Training Random Forest...");
            RandomForest rf = new RandomForest();
            rf.setNumIterations(100); // Number of trees
            try {
                rf.buildClassifier(data);
                System.out.println("Random Forest training complete.");
            } catch (Exception e) {
                System.err.println("Error during Random Forest training: " + e.getMessage());
                e.printStackTrace();
                return;
            }

            // Save model
            System.out.println("Saving model...");
            Path modelDir = Paths.get("model");
            if (!Files.exists(modelDir)) {
                Files.createDirectories(modelDir);
                System.out.println("Created model directory.");
            }
            try {
                SerializationHelper.write("model/rf_failure.model", rf);
                System.out.println("Saved rf_failure.model to the 'model' folder");
            } catch (Exception e) {
                System.err.println("Error saving model: " + e.getMessage());
                e.printStackTrace();
            }
        } catch (Exception e) {
            System.err.println("Error in FailurePredictionTrainer: " + e.getMessage());
            e.printStackTrace();
        }
    }
}