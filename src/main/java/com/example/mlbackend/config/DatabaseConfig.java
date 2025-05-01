package com.example.mlbackend.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.jdbc.core.JdbcTemplate;

@Configuration
public class DatabaseConfig {

    @Bean
    public JdbcTemplate jdbcTemplate(org.springframework.jdbc.datasource.DriverManagerDataSource dataSource) {
        return new JdbcTemplate(dataSource);
    }
} 