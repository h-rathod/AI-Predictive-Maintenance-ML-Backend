# Setup Instructions

This document explains how to set up and run the IoT Predictive Maintenance System using simple terminal commands.

## Prerequisites

- Java JDK 21
- Maven
- Supabase account with `sensor_data` and `predictions` tables

## Configuration

1. The application uses a `.env` file in the project root to load configuration:

```
# Supabase Configuration
SUPABASE_URL=your-supabase-url
SUPABASE_KEY=your-supabase-api-key

# Schedule Configuration (in milliseconds)
SCHEDULE_RATE=60000
```

Make sure this file exists with your actual Supabase URL and API key.

## Running the Application

To run the application:

1. Clean and package the application:

   ```
   mvn clean package -DskipTests
   ```

2. Run the application using Spring Boot:
   ```
   mvn spring-boot:run
   ```

That's it! The application will:

1. Load the `.env` file
2. Connect to the Supabase database
3. Load models (or create fallback models if the real ones can't be loaded)
4. Run the prediction pipeline every minute
5. Process data through all 4 models
6. Save results to the predictions table

## Monitoring

You can monitor the application through the console logs. Each minute, you should see logs like:

```
Starting scheduled pipeline at 2025-05-01 15:30:00 (runs every 60000 ms)
...
Completed scheduled pipeline, next run in approximately 60000 ms
```

## Troubleshooting

If you encounter issues:

1. Check that your `.env` file has correct Supabase credentials
2. Ensure your model files are in the `model/` directory
3. Verify that the required tables exist in Supabase
4. Check the console logs for specific error messages
