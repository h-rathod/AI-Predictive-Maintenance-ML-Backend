# Running the ML Backend

This document provides step-by-step instructions for running the IoT Predictive Maintenance System.

## First-time Setup

### 1. Configure Supabase Connection

Your application needs credentials to connect to Supabase. The recommended and most secure way to manage these for local development is by using a `.env.local` file. This file is specific to your local environment and should **never** be committed to version control.

**Understanding `.env` files:**
- **`.env.local`**: Use this file for your personal, local-specific settings and sensitive credentials (like API keys). It overrides any settings in `.env`. **This file must be in your `.gitignore` file.**
- **`.env`**: This file can be used for non-sensitive default settings for the project or as a template (e.g., `.env.example`). If it contains no secrets, it can be committed to version control.

**Steps to Configure Supabase:**

1.  **Create or Check `.gitignore`:**
    Ensure your project's `.gitignore` file (usually in the `ml-back` root, create it if it doesn't exist) includes the following line to prevent committing local environment files:
    ```
    .env.local
    *.env.local
    .env.*.local
    ```

2.  **Create `.env.local` file:**
    In the project root directory (`ml-back`), create a file named `.env.local` if it doesn't already exist.

3.  **Add Supabase Credentials to `.env.local`:**
    Open `.env.local` and add the following lines, replacing the placeholder values with your actual Supabase URL and Key:

    ```env
    SUPABASE_URL=https://your-project.supabase.co
    SUPABASE_KEY=your-supabase-anon-key
    ```
    The application will automatically load these variables when it starts.

**(Alternative) Using System Environment Variables:**
If you prefer not to use a `.env.local` file, you can set these as system-wide environment variables. However, managing them via `.env.local` is generally simpler for most development workflows and helps keep project configurations isolated.

*If using system variables:*

**Windows CMD:**
```shell
set SUPABASE_URL=https://your-project.supabase.co
set SUPABASE_KEY=your-supabase-key
```

**Windows PowerShell:**
```powershell
$env:SUPABASE_URL="https://your-project.supabase.co"
$env:SUPABASE_KEY="your-supabase-key"
```

**Linux/macOS (bash/zsh):**
```bash
export SUPABASE_URL=https://your-project.supabase.co
export SUPABASE_KEY=your-supabase-key
```

### 2. Ensure Model Files Exist

Verify that your model files exist in the `src/main/resources/model/` directory:

- `autoencoder.model` - Anomaly detection model
- `rf_failure.model` - Failure prediction model
- `rf_health_index.model` - Health index estimation model
- `part_risk.model` - Part Risk prediction model
- `part_risk_normalizer.bin` - Normalizer for Part Risk model
- `threshold.bin` - Anomaly detection threshold

If any are missing, the system will create fallback models automatically, but these are not as accurate.

## Building the Application

Build the application with Maven:

```
mvn clean install -DskipTests
```

## Running the Application

Run the Spring Boot application:

```
mvn spring-boot:run
```

The application will:

1. Load all ML models at startup
2. Run the prediction pipeline every minute (configurable via SCHEDULE_RATE environment variable)
3. Fetch data from the `sensor_data` table in Supabase via REST API
4. Process it through all 4 models
5. Save results to the `predictions` table in Supabase

## Important Configuration Settings

### 1. Data Collection and Inference Frequencies

For optimal results, configure your sensor data collection and inference frequencies:

- Set the sensor data collection to run every **5.5 seconds**
- The prediction pipeline runs every **60 seconds** by default

This ensures each inference run processes a sufficient number of new data points for accurate predictions.

### 2. Custom Scheduling Rate

To change how often the prediction pipeline runs (default is 60000ms = 1 minute), you can set the `SCHEDULE_RATE` environment variable.

**Recommended Method (using `.env.local`):**
Add or update the `SCHEDULE_RATE` in your `ml-back/.env.local` file:
```env
SCHEDULE_RATE=30000 # Example: 30 seconds
```
Then, simply run the application:
```shell
mvn spring-boot:run
```

**(Alternative) Using System Environment Variables:**

**Windows CMD:**
```shell
set SCHEDULE_RATE=30000
mvn spring-boot:run
```

**Windows PowerShell:**
```powershell
$env:SCHEDULE_RATE="30000"
mvn spring-boot:run
```

**Linux/macOS (bash/zsh):**
```bash
export SCHEDULE_RATE=30000
mvn spring-boot:run
```

### 3. Database Tables

Ensure your Supabase database has these tables:

**sensor_data Table:**

```sql
CREATE TABLE sensor_data (
    id SERIAL PRIMARY KEY,
    device_id TEXT NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    evaporator_coil_temperature DOUBLE PRECISION,
    fridge_temperature DOUBLE PRECISION,
    freezer_temperature DOUBLE PRECISION,
    air_temperature DOUBLE PRECISION,
    humidity DOUBLE PRECISION,
    compressor_vibration_x DOUBLE PRECISION,
    compressor_vibration_y DOUBLE PRECISION,
    compressor_vibration_z DOUBLE PRECISION,
    compressor_current DOUBLE PRECISION,
    input_voltage DOUBLE PRECISION,
    gas_leakage_level DOUBLE PRECISION
);
```

**predictions Table:**

```sql
CREATE TABLE predictions (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    device_id TEXT NOT NULL,
    is_anomaly BOOLEAN,
    failure_prob DOUBLE PRECISION,
    health_index DOUBLE PRECISION,
    rul TEXT -- Contains part at risk information
);
```

## Verifying Operation

1. The application will log pipeline executions in the console:

   ```
   Received 10 records from Supabase API
   Fetched 10 sensor data records via REST API
   Stored prediction result: anomaly=true, failure_prob=0.0, health_index=46.77, rul="Part at risk: Compressor (Warning)"
   Storage response status: 201 CREATED
   ```

2. You can check the `predictions` table in your Supabase project to verify results are being stored

3. You can trigger the pipeline manually via API:
   ```
   curl -X POST http://localhost:8080/api/predictions/run-pipeline
   ```

## Understanding the Output

The prediction results include:

- `anomaly` (boolean): Indicates if current data shows abnormal patterns
- `failure_prob` (0.0-1.0): Probability of imminent failure
- `health_index` (0-100): Equipment health score (higher is better)
- `rul` (text): Part at risk information in format "Part at risk: [component] ([condition])"

## Troubleshooting

- **Model Loading Issues**: Check that model files exist in the correct location
- **API Connection Errors**: Verify your Supabase URL and API key
- **Data Fetching Issues**: Ensure the sensor_data table exists and contains data
- **Timestamp Format Errors**: The system supports multiple timestamp formats but check logs for parsing issues
- **Input Shape Errors**: Ensure sensor data contains all required features for the Part Risk model (12 numeric features)
