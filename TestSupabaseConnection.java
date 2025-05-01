import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;
import io.github.cdimascio.dotenv.Dotenv;

/**
 * Test utility for verifying Supabase REST API connection.
 * This is only used for development/testing and not part of the main application.
 * 
 * Loads credentials from environment variables or .env file.
 */
public class TestSupabaseConnection {
    public static void main(String[] args) {
        // Load environment variables from .env file
        Dotenv dotenv = Dotenv.configure().load();
        
        // Get Supabase credentials from environment variables
        String supabaseUrl = dotenv.get("SUPABASE_URL");
        String supabaseKey = dotenv.get("SUPABASE_KEY");
        
        if (supabaseUrl == null || supabaseKey == null) {
            System.err.println("Error: SUPABASE_URL and SUPABASE_KEY must be set in environment variables or .env file");
            return;
        }
        
        try {
            // REST API endpoint to fetch sensor data
            URL url = new URL(supabaseUrl + "/rest/v1/sensor_data?limit=5");
            HttpURLConnection conn = (HttpURLConnection) url.openConnection();
            conn.setRequestMethod("GET");
            conn.setRequestProperty("apikey", supabaseKey);
            conn.setRequestProperty("Authorization", "Bearer " + supabaseKey);
            
            int responseCode = conn.getResponseCode();
            System.out.println("Response Code: " + responseCode);
            
            BufferedReader in = new BufferedReader(new InputStreamReader(conn.getInputStream()));
            String inputLine;
            StringBuilder response = new StringBuilder();
            
            while ((inputLine = in.readLine()) != null) {
                response.append(inputLine);
            }
            in.close();
            
            System.out.println("Response: " + response.toString());
            
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
} 