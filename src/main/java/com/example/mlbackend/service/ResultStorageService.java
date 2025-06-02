package com.example.mlbackend.service;

import com.example.mlbackend.model.PredictionResult;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.time.format.DateTimeFormatter;
import java.util.HashMap;
import java.util.Map;

@Slf4j
@Service
@RequiredArgsConstructor
public class ResultStorageService {

    private final RestTemplate restTemplate;
    
    @Value("${supabase.url}")
    private String supabaseUrl;
    
    @Value("${supabase.key}")
    private String supabaseKey;
    
    private static final DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");

    /**
     * Store prediction result in the database
     * @param result prediction result
     */
    public void storePrediction(PredictionResult result) {
        try {
            log.debug("Storing prediction for device: {}", result.getDeviceId());
            
            // Create HTTP headers with Supabase authentication
            HttpHeaders headers = new HttpHeaders();
            headers.set("apikey", supabaseKey);
            headers.set("Authorization", "Bearer " + supabaseKey);
            headers.setContentType(MediaType.APPLICATION_JSON);
            
            // Create request body
            Map<String, Object> body = new HashMap<>();
            body.put("timestamp", formatter.format(result.getTimestamp()));
            body.put("device_id", result.getDeviceId());
            body.put("is_anomaly", result.isAnomaly());
            body.put("failure_prob", result.getFailureProbability());
            body.put("health_index", result.getHealthIndex());
            
            // Store part risk information in the RUL field for backward compatibility
            // Format: "RUL: X, Part at risk: Y"
            String partRiskInfo = "";
            if (result.getPartAtRisk() != null && !result.getPartAtRisk().equals("none") && !result.getPartAtRisk().equals("unknown")) {
                partRiskInfo = String.format("%s (Part at risk: %s)", 
                        result.getRemainingUsefulLife(), 
                        result.getPartAtRisk());
                body.put("rul", partRiskInfo);
            } else {
                body.put("rul", result.getRemainingUsefulLife());
            }
            
            // Create HTTP entity with headers and body
            HttpEntity<Map<String, Object>> entity = new HttpEntity<>(body, headers);
            
            // Make REST API call to Supabase
            String url = supabaseUrl + "/rest/v1/predictions";
            ResponseEntity<Map> response = restTemplate.postForEntity(url, entity, Map.class);
            
            log.info("Stored prediction result: anomaly={}, failure_prob={}, health_index={}, rul={}, part_at_risk={}", 
                    result.isAnomaly(), 
                    result.getFailureProbability(), 
                    result.getHealthIndex(), 
                    result.getRemainingUsefulLife(),
                    result.getPartAtRisk());
            
            log.debug("Storage response status: {}", response.getStatusCode());
        } catch (Exception e) {
            log.error("Error storing prediction result: {}", e.getMessage(), e);
            throw new RuntimeException("Failed to store prediction result", e);
        }
    }
} 