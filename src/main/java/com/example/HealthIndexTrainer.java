package com.example;

import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.core.SerializationHelper;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public class HealthIndexTrainer {
    public static void main(String[] args) {
        try {
            System.out.println("Starting HealthIndexTrainer...");

            // Step 1: Load health_index_data.csv
            System.out.println("Loading health_index_data.csv...");
            CSVLoader loader = new CSVLoader();
            loader.setSource(new File("data/health_index_data.csv"));
            Instances data = loader.getDataSet();
            System.out.println("Loaded " + data.numInstances() + " instances.");

            // Step 2: Set class attribute (health_index)
            System.out.println("Setting class attribute...");
            data.setClassIndex(data.numAttributes() - 1); // Last column: health_index
            System.out.println("Health index attribute type: " + (data.attribute(data.numAttributes() - 1).isNumeric() ? "Numeric" : "Non-numeric"));

            // Step 3: Train Random Forest
            System.out.println("Training Random Forest...");
            RandomForest rf = new RandomForest();
            rf.setNumIterations(100); // Number of trees
            rf.buildClassifier(data);
            System.out.println("Random Forest training complete.");

            // Step 4: Save model
            System.out.println("Saving model...");
            Path modelDir = Paths.get("model");
            if (!Files.exists(modelDir)) {
                Files.createDirectories(modelDir);
                System.out.println("Created model directory.");
            }
            SerializationHelper.write("model/rf_health_index.model", rf);
            System.out.println("Saved rf_health_index.model to the 'model' folder");
        } catch (Exception e) {
            System.err.println("Error in HealthIndexTrainer: " + e.getMessage());
            e.printStackTrace();
        }
    }
}