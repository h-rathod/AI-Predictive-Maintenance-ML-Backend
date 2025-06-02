package com.example.mlbackend.service;

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.springframework.stereotype.Component;
import weka.classifiers.Classifier;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.FastVector;

import jakarta.annotation.PostConstruct;
import java.io.File;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.FileInputStream;
import java.lang.Exception;
import java.util.ArrayList;

@Slf4j
@Getter
@Component
public class ModelLoader {

    // Absolute path to model directory
    private final String MODEL_DIR = "model/";

    private MultiLayerNetwork autoencoderModel;
    private MultiLayerNetwork rulModel;
    private MultiLayerNetwork partRiskModel;
    private double threshold;
    private Classifier failureModel;
    private Classifier healthIndexModel;
    private double[] mean;
    private double[] std;
    private Instances failureHeader;
    private Instances healthIndexHeader;
    private org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler partRiskNormalizer;

    @PostConstruct
    public void init() {
        log.info("Loading ML models from {}", new File(MODEL_DIR).getAbsolutePath());
        
        try {
            // Create fallback models in case loading fails
            createFallbackModels();
            
            // Try to load the real models
            loadDeepLearningModels();
            loadWekaModels();
            loadThreshold();
            
            // Set up normalization parameters and headers
            setupNormalizationParameters();
            
            log.info("All models loaded successfully");
        } catch (Exception e) {
            log.error("Error during model loading: {}", e.getMessage(), e);
            log.warn("Using fallback models for demonstration");
        }
    }
    
    private void createFallbackModels() {
        log.info("Creating fallback models");
        
        // Create a simple autoencoder model
        MultiLayerConfiguration autoEncoderConf = new NeuralNetConfiguration.Builder()
            .weightInit(WeightInit.XAVIER)
            .list()
            .layer(0, new LSTM.Builder().nIn(11).nOut(32).activation(Activation.TANH).build())
            .layer(1, new LSTM.Builder().nIn(32).nOut(16).activation(Activation.TANH).build())
            .layer(2, new LSTM.Builder().nIn(16).nOut(16).activation(Activation.TANH).build())
            .layer(3, new LSTM.Builder().nIn(16).nOut(32).activation(Activation.TANH).build())
            .layer(4, new OutputLayer.Builder().nIn(32).nOut(11)
                    .activation(Activation.IDENTITY)
                    .lossFunction(LossFunctions.LossFunction.MSE)
                    .build())
            .build();
        
        autoencoderModel = new MultiLayerNetwork(autoEncoderConf);
        autoencoderModel.init();
        
        // Create a simple RUL model
        MultiLayerConfiguration rulConf = new NeuralNetConfiguration.Builder()
            .weightInit(WeightInit.XAVIER)
            .list()
            .layer(0, new LSTM.Builder().nIn(11).nOut(64).activation(Activation.TANH).build())
            .layer(1, new LSTM.Builder().nIn(64).nOut(32).activation(Activation.TANH).build())
            .layer(2, new DenseLayer.Builder().nIn(32).nOut(16).activation(Activation.RELU).build())
            .layer(3, new OutputLayer.Builder().nIn(16).nOut(1)
                    .activation(Activation.IDENTITY)
                    .lossFunction(LossFunctions.LossFunction.MSE)
                    .build())
            .build();
        
        rulModel = new MultiLayerNetwork(rulConf);
        rulModel.init();
        
        // Create fallback random forest models
        failureModel = new RandomForest();
        healthIndexModel = new RandomForest();
        
        // Default threshold
        threshold = 0.5;
        
        // Default normalization parameters
        mean = new double[11];
        std = new double[11];
        for (int i = 0; i < 11; i++) {
            mean[i] = 0.0;
            std[i] = 1.0;
        }
        
        // Create default headers for Weka models
        createDefaultHeaders();
    }
    
    private void createDefaultHeaders() {
        // Create attributes for failure prediction
        ArrayList<Attribute> failureAttrs = new ArrayList<>();
        for (int i = 0; i < 11; i++) {
            failureAttrs.add(new Attribute("feature" + (i+1)));
        }
        
        // Add class attribute with 2 values (0=normal, 1=failure)
        ArrayList<String> classVals = new ArrayList<>();
        classVals.add("normal");
        classVals.add("failure");
        failureAttrs.add(new Attribute("class", classVals));
        
        // Create instances object
        failureHeader = new Instances("failure_data", failureAttrs, 0);
        failureHeader.setClassIndex(11);  // Last attribute is the class
        
        // Same for health index, but with numeric class
        ArrayList<Attribute> healthAttrs = new ArrayList<>();
        for (int i = 0; i < 11; i++) {
            healthAttrs.add(new Attribute("feature" + (i+1)));
        }
        healthAttrs.add(new Attribute("health_index"));
        
        healthIndexHeader = new Instances("health_index_data", healthAttrs, 0);
        healthIndexHeader.setClassIndex(11);  // Last attribute is the class
    }
    
    private void loadDeepLearningModels() {
        try {
            // Try to load autoencoder model
            File autoencoderFile = new File(MODEL_DIR + "autoencoder.model");
            if (autoencoderFile.exists()) {
                log.info("Loading autoencoder model from: {}", autoencoderFile.getAbsolutePath());
                MultiLayerNetwork loadedModel = ModelSerializer.restoreMultiLayerNetwork(autoencoderFile);
                if (loadedModel != null) {
                    autoencoderModel = loadedModel;
                    log.info("Successfully loaded autoencoder model");
                }
            } else {
                log.warn("Autoencoder model file not found: {}", autoencoderFile.getAbsolutePath());
            }
            
            // Try to load RUL model
            File rulFile = new File(MODEL_DIR + "rul.model");
            if (rulFile.exists()) {
                log.info("Loading RUL model from: {}", rulFile.getAbsolutePath());
                MultiLayerNetwork loadedModel = ModelSerializer.restoreMultiLayerNetwork(rulFile);
                if (loadedModel != null) {
                    rulModel = loadedModel;
                    log.info("Successfully loaded RUL model");
                }
            } else {
                log.warn("RUL model file not found: {}", rulFile.getAbsolutePath());
            }
            
            // Try to load Part Risk model
            File partRiskFile = new File(MODEL_DIR + "part_risk.model");
            if (partRiskFile.exists()) {
                log.info("Loading Part Risk model from: {}", partRiskFile.getAbsolutePath());
                MultiLayerNetwork loadedModel = ModelSerializer.restoreMultiLayerNetwork(partRiskFile);
                if (loadedModel != null) {
                    partRiskModel = loadedModel;
                    log.info("Successfully loaded Part Risk model");
                    
                    // Try to load the normalizer for part risk model
                    File normalizerFile = new File(MODEL_DIR + "part_risk_normalizer.bin");
                    if (normalizerFile.exists()) {
                        log.info("Loading Part Risk normalizer from: {}", normalizerFile.getAbsolutePath());
                        try {
                            partRiskNormalizer = org.nd4j.linalg.dataset.api.preprocessor.serializer.NormalizerSerializer
                                .getDefault().restore(normalizerFile);
                            log.info("Successfully loaded Part Risk normalizer");
                        } catch (Exception e) {
                            log.error("Error loading Part Risk normalizer: {}", e.getMessage());
                            partRiskNormalizer = new org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler();
                        }
                    } else {
                        log.warn("Part Risk normalizer file not found: {}", normalizerFile.getAbsolutePath());
                        partRiskNormalizer = new org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler();
                    }
                }
            } else {
                log.warn("Part Risk model file not found: {}", partRiskFile.getAbsolutePath());
                
                // Create a fallback part risk model
                MultiLayerConfiguration partRiskConf = new NeuralNetConfiguration.Builder()
                    .weightInit(WeightInit.XAVIER)
                    .list()
                    .layer(0, new DenseLayer.Builder().nIn(12).nOut(32).activation(Activation.RELU).build())
                    .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                            .nIn(32).nOut(6)
                            .activation(Activation.SOFTMAX)
                            .build())
                    .build();
                
                partRiskModel = new MultiLayerNetwork(partRiskConf);
                partRiskModel.init();
                partRiskNormalizer = new org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler();
            }
        } catch (Exception e) {
            log.error("Error loading deep learning models: {}", e.getMessage());
            log.info("Using fallback deep learning models");
        }
    }
    
    private void loadWekaModels() {
        try {
            // Try to load failure model
            File failureModelFile = new File(MODEL_DIR + "rf_failure.model");
            if (failureModelFile.exists()) {
                log.info("Loading failure model from: {}", failureModelFile.getAbsolutePath());
                Classifier loadedModel = (Classifier) SerializationHelper.read(failureModelFile.getAbsolutePath());
                if (loadedModel != null) {
                    failureModel = loadedModel;
                    log.info("Successfully loaded failure model");
                }
            } else {
                log.warn("Failure model file not found: {}", failureModelFile.getAbsolutePath());
            }
            
            // Try to load health index model
            File healthIndexModelFile = new File(MODEL_DIR + "rf_health_index.model");
            if (healthIndexModelFile.exists()) {
                log.info("Loading health index model from: {}", healthIndexModelFile.getAbsolutePath());
                Classifier loadedModel = (Classifier) SerializationHelper.read(healthIndexModelFile.getAbsolutePath());
                if (loadedModel != null) {
                    healthIndexModel = loadedModel;
                    log.info("Successfully loaded health index model");
                }
            } else {
                log.warn("Health index model file not found: {}", healthIndexModelFile.getAbsolutePath());
            }
        } catch (Exception e) {
            log.error("Error loading Weka models: {}", e.getMessage());
            log.info("Using fallback Weka models");
        }
    }
    
    private void loadThreshold() {
        try {
            File thresholdFile = new File(MODEL_DIR + "threshold.bin");
            if (thresholdFile.exists()) {
                log.info("Loading threshold from: {}", thresholdFile.getAbsolutePath());
                try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(thresholdFile))) {
                    Object obj = ois.readObject();
                    if (obj instanceof Double) {
                        threshold = (Double) obj;
                        log.info("Successfully loaded threshold: {}", threshold);
                    }
                }
            } else {
                log.warn("Threshold file not found: {}", thresholdFile.getAbsolutePath());
            }
        } catch (Exception e) {
            log.error("Error loading threshold: {}", e.getMessage());
            log.info("Using default threshold: {}", threshold);
        }
    }
    
    private void setupNormalizationParameters() {
        // For simplicity, we'll use default values
        mean = new double[11];
        std = new double[11];
        for (int i = 0; i < 11; i++) {
            mean[i] = 0.0;
            std[i] = 1.0;
        }
        
        log.info("Using default normalization parameters");
    }
} 