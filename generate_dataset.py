import pandas as pd
import numpy as np

# Set a random seed for reproducibility
np.random.seed(42)

# Generate synthetic data
num_samples = 1000
humidity = np.random.uniform(30, 90, num_samples)  # Humidity between 30% and 90%
heat_celcius = np.random.uniform(10, 40, num_samples)  # Temperature between 10°C and 40°C

# Simulate MoisturePercentage based on Humidity and Heatcelcius
moisture_percentage = 50 + 0.5 * humidity - 0.8 * heat_celcius + np.random.normal(0, 5, num_samples)
moisture_percentage = np.clip(moisture_percentage, 0, 100)  # Ensure moisture is between 0% and 100%

# Create a DataFrame
data = pd.DataFrame({
    'Humidity': humidity,
    'Heatcelcius': heat_celcius,
    'MoisturePercentage': moisture_percentage
})

# Save the dataset to a CSV file
data.to_csv('Dataset.csv', index=False)

print("Dataset generated and saved as 'Dataset.csv'")