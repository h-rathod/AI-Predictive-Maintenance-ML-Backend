package com.example.mlbackend.service;

import com.example.mlbackend.model.SensorData;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;

/**
 * Service responsible for fetching sensor data and orchestrating the prediction pipeline.
 * This service coordinates the data flow between different components:
 * 1. Fetches data from Supabase via SupabaseApiService
 * 2. Passes data to InferenceService for ML predictions
 * 3. Saves results via ResultStorageService
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class DataFetcherService {

    private final SupabaseApiService supabaseApiService;
    private final InferenceService inferenceService;
    private final ResultStorageService resultStorageService;

    /**
     * Fetch the latest sensor data from Supabase
     * @param limit number of records to fetch
     * @return list of sensor data
     */
    public List<SensorData> fetchLatestSensorData(int limit) {
        try {
            // Use our REST API service instead of JDBC
            List<SensorData> sensorDataList = supabaseApiService.fetchSensorData(limit);
            
            log.debug("Fetched {} sensor data records", sensorDataList.size());
            return sensorDataList;
        } catch (Exception e) {
            log.error("Error fetching sensor data: {}", e.getMessage(), e);
            return new ArrayList<>();
        }
    }

    /**
     * Run the complete pipeline: fetch data, preprocess, run inference, store results
     */
    public void runPipeline() {
        try {
            log.info("Starting data pipeline execution");
            
            // 1. Fetch the latest sensor data (11 records for sequence - matches RUL model expectations)
            List<SensorData> sensorDataList = fetchLatestSensorData(11);
            
            if (sensorDataList.isEmpty()) {
                log.warn("No sensor data available, skipping inference");
                return;
            }
            
            // 2. Run inference on the data
            var predictionResult = inferenceService.runInference(sensorDataList);
            
            // 3. Store the results
            resultStorageService.storePrediction(predictionResult);
            
            log.info("Pipeline execution completed successfully");
        } catch (Exception e) {
            log.error("Error executing pipeline: {}", e.getMessage(), e);
        }
    }
} 