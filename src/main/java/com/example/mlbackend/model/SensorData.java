package com.example.mlbackend.model;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;

/**
 * Represents a single reading from IoT sensors.
 * 
 * This model maps to the sensor_data table in Supabase and contains
 * all the sensor measurements collected from a specific device.
 * Each feature corresponds to a different sensor measurement such as
 * temperature, vibration, current, etc.
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class SensorData {
    private String deviceId;
    private LocalDateTime timestamp;
    
    // Sensor features - these map to columns in the Supabase sensor_data table
    private double feature1;  // evaporator_coil_temperature
    private double feature2;  // fridge_temperature
    private double feature3;  // freezer_temperature
    private double feature4;  // air_temperature
    private double feature5;  // humidity
    private double feature6;  // compressor_vibration_x
    private double feature7;  // compressor_vibration_y
    private double feature8;  // compressor_vibration_z
    private double feature9;  // compressor_current
    private double feature10; // input_voltage
    private double feature11; // gas_leakage_level
    
    public double[] getFeatureArray() {
        return new double[] {
            feature1, feature2, feature3, feature4, feature5, 
            feature6, feature7, feature8, feature9, feature10, feature11
        };
    }
} 