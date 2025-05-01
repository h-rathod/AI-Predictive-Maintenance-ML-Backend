package com.example;

import com.opencsv.CSVReader;
import com.opencsv.exceptions.CsvValidationException;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

public class RulTrainer {
    public static class CustomListDataSetIterator implements DataSetIterator {
        private List<DataSet> dataSets;
        private int cursor = 0;
        private int batchSize;

        public CustomListDataSetIterator(List<DataSet> dataSets, int batchSize) {
            this.dataSets = dataSets;
            this.batchSize = batchSize;
        }

        @Override
        public DataSet next(int num) {
            if (cursor >= dataSets.size()) {
                throw new IllegalStateException("No more data available");
            }
            int end = Math.min(cursor + num, dataSets.size());
            List<DataSet> batch = dataSets.subList(cursor, end);
            cursor = end;

            // Prepare 3D features: [batchSize, numFeatures=11, sequenceLength=10]
            int batchSize = batch.size();
            int sequenceLength = 10;
            int numFeatures = 11;
            INDArray featuresBatch = Nd4j.create(batchSize, numFeatures, sequenceLength);
            INDArray labelsBatch = Nd4j.create(batchSize, 1, sequenceLength); // [batchSize, 1, 10]

            for (int i = 0; i < batchSize; i++) {
                INDArray features = batch.get(i).getFeatures(); // Shape: [10, 11]
                // Transpose to [11, 10] for RNN input
                INDArray featuresTransposed = features.permute(1, 0);
                // Assign to the i-th slice of featuresBatch
                featuresBatch.tensorAlongDimension(i, 1, 2).assign(featuresTransposed);

                INDArray label = batch.get(i).getLabels(); // Shape: [1]
                // Set RUL at the last time step
                labelsBatch.putScalar(new int[]{i, 0, sequenceLength - 1}, label.getDouble(0));
            }

            return new DataSet(featuresBatch, labelsBatch);
        }

        @Override
        public int inputColumns() {
            return (int) dataSets.get(0).getFeatures().shape()[1];
        }

        @Override
        public int totalOutcomes() {
            return 1; // Single RUL value
        }

        @Override
        public boolean hasNext() {
            return cursor < dataSets.size();
        }

        @Override
        public DataSet next() {
            return next(batchSize);
        }

        @Override
        public void reset() {
            cursor = 0;
        }

        @Override
        public int batch() {
            return batchSize;
        }

        @Override
        public List<String> getLabels() {
            return null; // Not used for regression tasks
        }

        @Override
        public DataSetPreProcessor getPreProcessor() {
            return null; // No pre-processor is used
        }

        @Override
        public void setPreProcessor(DataSetPreProcessor preProcessor) {
            // No-op: Pre-processor not used
        }

        // Other required methods
        @Override
        public boolean resetSupported() {
            return true;
        }

        @Override
        public boolean asyncSupported() {
            return false;
        }
    }

    public static void main(String[] args) {
        try {
            System.out.println("Starting RulTrainer...");

            // Step 1: Load rul_data.csv
            System.out.println("Loading rul_data.csv...");
            List<double[]> data = new ArrayList<>();
            List<String[]> rawRows = new ArrayList<>();
            try (CSVReader reader = new CSVReader(new FileReader("data/rul_data.csv"))) {
                String[] header = reader.readNext(); // Read header
                if (header == null || header.length != 12) {
                    throw new IllegalStateException("Invalid header in rul_data.csv: " + (header == null ? "null" : String.join(",", header)));
                }
                String[] line;
                int rowNum = 1;
                while ((line = reader.readNext()) != null) {
                    rowNum++;
                    rawRows.add(line); // Store for logging
                    if (line.length != 12) {
                        System.err.println("Warning: Skipping malformed row " + rowNum + " with " + line.length + " columns: " + String.join(",", line));
                        continue;
                    }
                    try {
                        double[] features = new double[12];
                        for (int i = 0; i < 12; i++) {
                            features[i] = Double.parseDouble(line[i]);
                        }
                        data.add(features);
                    } catch (NumberFormatException e) {
                        System.err.println("Warning: Skipping row " + rowNum + " due to invalid number format: " + e.getMessage() + " in row: " + String.join(",", line));
                    }
                }
            } catch (IOException | CsvValidationException e) {
                System.err.println("Error loading rul_data.csv: " + e.getMessage());
                e.printStackTrace();
                return;
            }

            // Log first and last 5 rows
            System.out.println("Loaded " + data.size() + " data points.");
            if (rawRows.size() > 0) {
                System.out.println("First 5 rows of rul_data.csv:");
                for (int i = 0; i < Math.min(5, rawRows.size()); i++) {
                    System.out.println("Row " + (i + 2) + ": " + String.join(",", rawRows.get(i)));
                }
                System.out.println("Last 5 rows of rul_data.csv:");
                for (int i = Math.max(0, rawRows.size() - 5); i < rawRows.size(); i++) {
                    System.out.println("Row " + (i + 2) + ": " + String.join(",", rawRows.get(i)));
                }
            }

            // Validate row count
            if (data.size() != 144000) {
                System.err.println("Error: Expected 144000 data rows, but loaded " + data.size());
            }

            if (data.isEmpty()) {
                throw new IllegalStateException("No valid data loaded from rul_data.csv");
            }

            // Step 2: Normalize data
            System.out.println("Normalizing data...");
            double[] minValues = new double[]{-5.0, -30.0, -10.0, 15.0, 20.0, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0};
            double[] maxValues = new double[]{0.0, -15.0, 5.0, 35.0, 80.0, 10.0, 10.0, 10.0, 20.0, 240.0, 0.1, 1000.0};
            List<INDArray> sequences = new ArrayList<>();
            List<INDArray> labels = new ArrayList<>();
            int sequenceLength = 10;

            for (int i = sequenceLength - 1; i < data.size(); i++) {
                INDArray sequence = Nd4j.create(sequenceLength, 11);
                for (int j = 0; j < sequenceLength; j++) {
                    double[] row = data.get(i - sequenceLength + 1 + j);
                    for (int k = 0; k < 11; k++) {
                        double normalized = (row[k] - minValues[k]) / (maxValues[k] - minValues[k]);
                        sequence.putScalar(j, k, normalized);
                    }
                }
                double rul = data.get(i)[11]; // Last column is RUL
                double normalizedRul = (rul - minValues[11]) / (maxValues[11] - minValues[11]);
                INDArray label = Nd4j.create(1).putScalar(0, normalizedRul);
                sequences.add(sequence);
                labels.add(label);
            }

            List<DataSet> dataSets = new ArrayList<>();
            for (int i = 0; i < sequences.size(); i++) {
                dataSets.add(new DataSet(sequences.get(i), labels.get(i)));
            }
            System.out.println("Created " + dataSets.size() + " sequences. Expected: " + (data.size() - sequenceLength + 1));

            // Step 3: Create DataSetIterator
            System.out.println("Creating DataSetIterator...");
            int batchSize = 8;
            CustomListDataSetIterator iterator = new CustomListDataSetIterator(dataSets, batchSize);

            // Step 4: Define LSTM model
            System.out.println("Defining LSTM model...");
            org.deeplearning4j.nn.conf.MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                    .seed(123)
                    .weightInit(WeightInit.XAVIER)
                    .updater(new Adam(0.001))
                    .trainingWorkspaceMode(WorkspaceMode.ENABLED)
                    .inferenceWorkspaceMode(WorkspaceMode.ENABLED)
                    .list()
                    .layer(new LSTM.Builder()
                            .nIn(11)
                            .nOut(20) // Reduced from 50
                            .activation(Activation.TANH)
                            .build())
                    .layer(new RnnOutputLayer.Builder(LossFunctions.LossFunction.MSE)
                            .nIn(20)
                            .nOut(1)
                            .activation(Activation.IDENTITY)
                            .build())
                    .build();

            MultiLayerNetwork model = new MultiLayerNetwork(config);
            model.init();
            System.out.println("Model initialized.");

            // Step 5: Train the model
            System.out.println("Starting training...");
            try {
                for (int epoch = 0; epoch < 5; epoch++) { // Reduced from 10
                    iterator.reset();
                    while (iterator.hasNext()) {
                        model.fit(iterator.next());
                    }
                    System.out.println("Epoch " + epoch + " complete");
                }
            } catch (Exception e) {
                System.err.println("Error during training: " + e.getMessage());
                e.printStackTrace();
                return;
            }

            // Step 6: Save model
            System.out.println("Saving model...");
            Path modelDir = Paths.get("model");
            if (!Files.exists(modelDir)) {
                Files.createDirectories(modelDir);
                System.out.println("Created model directory.");
            }
            try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream("model/rul.model"))) {
                oos.writeObject(model);
                System.out.println("Saved rul.model to the 'model' folder");
            } catch (Exception e) {
                System.err.println("Error saving model: " + e.getMessage());
                e.printStackTrace();
            }
        } catch (Exception e) {
            System.err.println("Error in RulTrainer: " + e.getMessage());
            e.printStackTrace();
        }
    }
}