package com.example;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import com.opencsv.CSVReader;
import com.opencsv.exceptions.CsvValidationException;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

public class AutoencoderTrainer {
    public static void main(String[] args) {
        try {
            System.out.println("Starting AutoencoderTrainer...");

            // Step 1: Load normal_data.csv
            System.out.println("Loading normal_data.csv...");
            List<double[]> data = new ArrayList<>();
            try (CSVReader reader = new CSVReader(new FileReader("data/normal_data.csv"))) {
                reader.readNext(); // Skip header
                String[] line;
                int rowNum = 1;
                while ((line = reader.readNext()) != null) {
                    rowNum++;
                    // Validate row has 11 columns
                    if (line.length != 11) {
                        System.err.println("Warning: Skipping malformed row " + rowNum + " with " + line.length + " columns");
                        continue;
                    }
                    try {
                        double[] features = new double[11];
                        for (int i = 0; i < 11; i++) {
                            features[i] = Double.parseDouble(line[i]);
                        }
                        data.add(features);
                    } catch (NumberFormatException e) {
                        System.err.println("Warning: Skipping row " + rowNum + " due to invalid number format: " + e.getMessage());
                    }
                }
            }
            System.out.println("Loaded " + data.size() + " valid data points.");

            // Check if data is empty
            if (data.isEmpty()) {
                throw new IllegalStateException("No valid data loaded from normal_data.csv");
            }

            // Step 2: Preprocess data (normalize)
            System.out.println("Normalizing data...");
            double[] minValues = new double[]{-5.0, -30.0, -10.0, 15.0, 20.0, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0};
            double[] maxValues = new double[]{0.0, -15.0, 5.0, 35.0, 80.0, 10.0, 10.0, 10.0, 20.0, 240.0, 0.1};
            INDArray features = Nd4j.create(data.size(), 11);
            for (int i = 0; i < data.size(); i++) {
                for (int j = 0; j < 11; j++) {
                    features.putScalar(i, j, (data.get(i)[j] - minValues[j]) / (maxValues[j] - minValues[j]));
                }
            }

            // Step 3: Create DataSet
            System.out.println("Creating DataSet...");
            DataSet dataSet = new DataSet(features, features); // Autoencoder: input = output

            // Step 4: Define autoencoder model
            System.out.println("Defining autoencoder model...");
            MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                    .seed(123)
                    .weightInit(WeightInit.XAVIER)
                    .updater(new Adam(0.001))
                    .list()
                    .layer(new DenseLayer.Builder()
                            .nIn(11)
                            .nOut(8)
                            .activation(Activation.RELU)
                            .build())
                    .layer(new DenseLayer.Builder()
                            .nIn(8)
                            .nOut(4)
                            .activation(Activation.RELU)
                            .build())
                    .layer(new DenseLayer.Builder()
                            .nIn(4)
                            .nOut(8)
                            .activation(Activation.RELU)
                            .build())
                    .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                            .nIn(8)
                            .nOut(11)
                            .activation(Activation.IDENTITY)
                            .build())
                    .build();

            MultiLayerNetwork model = new MultiLayerNetwork(config);
            model.init();
            System.out.println("Model initialized.");

            // Step 5: Train the model
            System.out.println("Starting training...");
            for (int epoch = 0; epoch < 50; epoch++) {
                model.fit(dataSet);
                System.out.println("Epoch " + epoch + " complete");
            }

            // Step 6: Compute reconstruction error threshold
            System.out.println("Computing reconstruction error threshold...");
            INDArray output = model.output(features);
            INDArray errors = features.sub(output).norm2(1); // L2 norm of errors
            double meanError = errors.meanNumber().doubleValue();
            double stdError = errors.stdNumber().doubleValue();
            double threshold = meanError + 2 * stdError; // 2 standard deviations
            System.out.println("Threshold: " + threshold);

            // Step 7: Save model and threshold
            System.out.println("Saving model and threshold...");
            Path modelDir = Paths.get("model");
            if (!Files.exists(modelDir)) {
                Files.createDirectories(modelDir);
                System.out.println("Created model directory.");
            }
            model.save(new File("model/autoencoder.model"));
            try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream("model/threshold.bin"))) {
                oos.writeObject(threshold);
            }
            System.out.println("Saved autoencoder.model and threshold.bin to the 'model' folder");
        } catch (IOException | CsvValidationException e) {
            System.err.println("Error in AutoencoderTrainer: " + e.getMessage());
            e.printStackTrace();
        }
    }
}