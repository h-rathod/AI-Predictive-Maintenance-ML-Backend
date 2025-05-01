package com.example.mlbackend.controller;

import com.example.mlbackend.service.SchedulerService;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

/**
 * REST controller for manual interaction with the prediction pipeline.
 * 
 * While the prediction pipeline runs automatically on a schedule,
 * this controller provides an endpoint to manually trigger the pipeline
 * for testing or immediate analysis of sensor data.
 */
@RestController
@RequestMapping("/api/predictions")
@RequiredArgsConstructor
public class PredictionController {

    private final SchedulerService schedulerService;

    /**
     * Manually triggers the prediction pipeline.
     * Useful for testing or for immediate analysis of current sensor data.
     * 
     * @return Success message if pipeline was triggered successfully
     */
    @PostMapping("/run-pipeline")
    public String runPipeline() {
        schedulerService.scheduledDataPipeline();
        return "Pipeline executed successfully";
    }
} 