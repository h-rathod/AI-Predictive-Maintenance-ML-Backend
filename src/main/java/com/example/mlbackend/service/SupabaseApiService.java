package com.example.mlbackend.service;

import com.example.mlbackend.model.SensorData;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.ParameterizedTypeReference;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpMethod;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.time.format.DateTimeParseException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

@Slf4j
@Service
@RequiredArgsConstructor
public class SupabaseApiService {

    private final RestTemplate restTemplate;
    private final ObjectMapper objectMapper;
    
    @Value("${supabase.url}")
    private String supabaseUrl;
    
    @Value("${supabase.key}")
    private String supabaseKey;
    
    private static final DateTimeFormatter STANDARD_FORMATTER = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");
    private static final DateTimeFormatter ISO_FORMATTER = DateTimeFormatter.ofPattern("yyyy-MM-dd'T'HH:mm:ss");
    
    /**
     * Fetch sensor data from Supabase using REST API
     * @param limit number of records to fetch
     * @return list of sensor data
     */
    public List<SensorData> fetchSensorData(int limit) {
        try {
            // Create HTTP headers with Supabase authentication
            HttpHeaders headers = new HttpHeaders();
            headers.set("apikey", supabaseKey);
            headers.set("Authorization", "Bearer " + supabaseKey);
            
            // Create HTTP entity with headers
            HttpEntity<String> entity = new HttpEntity<>(headers);
            
            // Make REST API call to Supabase - use ParameterizedTypeReference to handle the JSON array directly
            String url = supabaseUrl + "/rest/v1/sensor_data?order=timestamp.desc&limit=" + limit;
            ResponseEntity<List<Map<String, Object>>> response = restTemplate.exchange(
                    url, 
                    HttpMethod.GET, 
                    entity, 
                    new ParameterizedTypeReference<List<Map<String, Object>>>() {});
            
            List<Map<String, Object>> rawData = response.getBody();
            List<SensorData> sensorDataList = new ArrayList<>();
            
            if (rawData != null) {
                log.debug("Received {} records from Supabase API", rawData.size());
                
                // Convert raw data to SensorData objects
                for (Map<String, Object> data : rawData) {
                    try {
                        // Extract device_id safely (could be null)
                        String deviceId = data.get("device_id") != null ? 
                                data.get("device_id").toString() : "unknown";
                        
                        // Parse timestamp with multiple format attempts
                        LocalDateTime timestamp = parseTimestamp(data.get("timestamp"));
                        
                        SensorData sensorData = SensorData.builder()
                                .deviceId(deviceId)
                                .timestamp(timestamp)
                                .feature1(parseDouble(data.get("evaporator_coil_temperature")))
                                .feature2(parseDouble(data.get("fridge_temperature")))
                                .feature3(parseDouble(data.get("freezer_temperature")))
                                .feature4(parseDouble(data.get("air_temperature")))
                                .feature5(parseDouble(data.get("humidity")))
                                .feature6(parseDouble(data.get("compressor_vibration_x")))
                                .feature7(parseDouble(data.get("compressor_vibration_y")))
                                .feature8(parseDouble(data.get("compressor_vibration_z")))
                                .feature9(parseDouble(data.get("compressor_current")))
                                .feature10(parseDouble(data.get("input_voltage")))
                                .feature11(parseDouble(data.get("gas_leakage_level")))
                                .build();
                        sensorDataList.add(sensorData);
                    } catch (Exception e) {
                        log.error("Error parsing sensor data record: {}", e.getMessage(), e);
                    }
                }
            } else {
                log.warn("No data received from Supabase API");
            }
            
            // Reverse to get ascending order by timestamp
            Collections.reverse(sensorDataList);
            
            log.debug("Fetched {} sensor data records via REST API", sensorDataList.size());
            return sensorDataList;
            
        } catch (Exception e) {
            log.error("Error fetching sensor data via REST API: {}", e.getMessage(), e);
            return new ArrayList<>();
        }
    }
    
    /**
     * Parse timestamp with multiple format attempts
     */
    private LocalDateTime parseTimestamp(Object timestampObj) {
        if (timestampObj == null) {
            log.warn("Timestamp is null, using current time");
            return LocalDateTime.now();
        }
        
        String timestampStr = timestampObj.toString();
        
        // Try ISO format first (with T)
        try {
            return LocalDateTime.parse(timestampStr, ISO_FORMATTER);
        } catch (DateTimeParseException e) {
            // Try standard format (without T)
            try {
                return LocalDateTime.parse(timestampStr, STANDARD_FORMATTER);
            } catch (DateTimeParseException e2) {
                // Try ISO-8601 built-in parser
                try {
                    return LocalDateTime.parse(timestampStr);
                } catch (DateTimeParseException e3) {
                    log.warn("Couldn't parse timestamp '{}', using current time", timestampStr);
                    return LocalDateTime.now();
                }
            }
        }
    }
    
    private double parseDouble(Object value) {
        if (value == null) {
            return 0.0;
        }
        
        if (value instanceof Number) {
            return ((Number) value).doubleValue();
        }
        
        try {
            return Double.parseDouble(value.toString());
        } catch (NumberFormatException e) {
            return 0.0;
        }
    }
} 