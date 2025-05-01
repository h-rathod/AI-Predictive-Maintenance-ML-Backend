package com.example.mlbackend.service;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

@Slf4j
@Service
@RequiredArgsConstructor
public class SchedulerService {

    private final DataFetcherService dataFetcherService;
    
    @Value("${schedule.data-fetch.rate}")
    private long scheduleRate;
    
    private final DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");

    @Scheduled(fixedRateString = "${schedule.data-fetch.rate}")
    public void scheduledDataPipeline() {
        LocalDateTime now = LocalDateTime.now();
        log.info("Starting scheduled pipeline at {} (runs every {} ms)", formatter.format(now), scheduleRate);
        dataFetcherService.runPipeline();
        log.info("Completed scheduled pipeline, next run in approximately {} ms", scheduleRate);
    }
} 