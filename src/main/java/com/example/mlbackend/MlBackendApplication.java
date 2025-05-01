package com.example.mlbackend;

import io.github.cdimascio.dotenv.Dotenv;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.scheduling.annotation.EnableScheduling;

/**
 * Main entry point for the IoT Predictive Maintenance System
 * This Spring Boot application loads pre-trained ML models and provides
 * real-time predictive maintenance capabilities by analyzing sensor data.
 * 
 * Key features:
 * - Loads and manages 4 different ML models
 * - Fetches sensor data from Supabase
 * - Runs prediction pipeline at scheduled intervals
 * - Stores prediction results back to Supabase
 */
@SpringBootApplication
@EnableScheduling  // Enable scheduled tasks for regular pipeline execution
public class MlBackendApplication {
    public static void main(String[] args) {
        // Load environment variables from .env file
        Dotenv dotenv = Dotenv.configure().load();
        
        // Set system properties from .env file
        dotenv.entries().forEach(entry -> 
            System.setProperty(entry.getKey(), entry.getValue())
        );
        
        SpringApplication.run(MlBackendApplication.class, args);
    }
} 