package com.example.mlbackend.config;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.postgresql.Driver;
import org.springframework.jdbc.datasource.DriverManagerDataSource;
import lombok.extern.slf4j.Slf4j;

@Slf4j
@Configuration
public class SupabaseClientConfig {

    @Value("${supabase.url}")
    private String supabaseUrl;
    
    @Value("${supabase.key}")
    private String supabaseKey;

    @Bean
    public DriverManagerDataSource supabaseDataSource() {
        log.info("Configuring Supabase REST API connection");
        
        try {
            // Try direct REST API connection
            DriverManagerDataSource dataSource = new DriverManagerDataSource();
            dataSource.setDriverClassName(Driver.class.getName());
            
            // Extract database host from SUPABASE_URL
            String dbHost = extractDbHost(supabaseUrl);
            
            // Use a simpler JDBC URL with hostname instead of full URL
            String jdbcUrl = "jdbc:postgresql://" + dbHost + ":5432/postgres";
            
            log.debug("JDBC URL: {}", jdbcUrl);
            
            dataSource.setUrl(jdbcUrl);
            dataSource.setUsername("postgres");
            dataSource.setPassword(supabaseKey);
            
            // Test database connection
            try {
                dataSource.getConnection().close();
                log.info("Successfully connected to database");
            } catch (Exception e) {
                log.warn("Could not connect to database with initial configuration: {}", e.getMessage());
                
                // Fallback to another configuration with alternate port
                jdbcUrl = "jdbc:postgresql://" + dbHost + ":6543/postgres";
                dataSource.setUrl(jdbcUrl);
                log.debug("Trying fallback JDBC URL: {}", jdbcUrl);
            }
            
            return dataSource;
        } catch (Exception e) {
            log.error("Error setting up database connection: {}", e.getMessage(), e);
            throw new RuntimeException("Failed to configure database connection", e);
        }
    }
    
    /**
     * Extract the database host from the Supabase URL
     * Example: https://project-id.supabase.co -> db.project-id.supabase.co
     */
    private String extractDbHost(String supabaseUrl) {
        // Remove https:// or http:// prefix
        String host = supabaseUrl.replace("https://", "").replace("http://", "");
        
        // Add 'db.' prefix
        return "db." + host;
    }
} 