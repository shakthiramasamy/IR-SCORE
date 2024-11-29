import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score, 
    f1_score
)

def generate_synthetic_data(n_samples=2000):
    """Generate synthetic dataset for interventional oncology"""
    np.random.seed(42)
    
    # Generate patient demographics
    age = np.random.normal(60, 10, n_samples).clip(18, 90)
    gender = np.random.choice(['Male', 'Female'], n_samples, p=[0.6, 0.4])
    bmi = np.random.normal(27, 5, n_samples).clip(18, 45)
    smoking_history = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    
    # Tumor characteristics
    tumor_size = np.random.normal(3, 1.5, n_samples).clip(1, 10)
    num_tumors = np.random.poisson(1.2, n_samples).clip(1, 5)
    vascular_invasion = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
    
    # Clinical and lab data
    bilirubin = np.random.normal(1.2, 0.4, n_samples).clip(0.5, 3.0)
    albumin = np.random.normal(3.5, 0.5, n_samples).clip(2.5, 5.0)
    inr = np.random.normal(1.1, 0.2, n_samples).clip(0.8, 2.0)
    ecog_score = np.random.choice([0, 1, 2, 3], n_samples, p=[0.5, 0.3, 0.15, 0.05])
    
    # Procedure details
    intervention_type = np.random.choice(
        ['RFA', 'MWA', 'TACE', 'Y90'],
        n_samples,
        p=[0.3, 0.2, 0.4, 0.1]
    )
    ablation_duration = np.where(
        np.isin(intervention_type, ['RFA', 'MWA']),
        np.random.normal(15, 5, n_samples).clip(5, 30),
        np.nan
    )
    
    # Outcomes
    response = np.random.choice(
        ['Complete Response', 'Partial Response', 'Stable Disease', 'Progressive Disease'],
        n_samples,
        p=[0.4, 0.3, 0.2, 0.1]
    )
    complications = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
    
    # Compile data into a DataFrame
    data = pd.DataFrame({
        'Age': age,
        'Gender': gender,
        'BMI': bmi,
        'Smoking_History': smoking_history,
        'Tumor_Size': tumor_size,
        'Num_Tumors': num_tumors,
        'Vascular_Invasion': vascular_invasion,
        'Bilirubin': bilirubin,
        'Albumin': albumin,
        'INR': inr,
        'ECOG_Score': ecog_score,
        'Intervention_Type': intervention_type,
        'Ablation_Duration': ablation_duration,
        'Response': response,
        'Complications': complications
    })
    
    # Save the synthetic data to CSV
    data.to_csv('synthetic_ir_data.csv', index=False)
    print("Synthetic data generated and saved to 'synthetic_ir_data.csv'")
    return data

def main():
    # Generate synthetic data
    print("Generating synthetic data...")
    data = generate_synthetic_data(n_samples=2000)
    
    # Print the first few rows and data info
    print("\nFirst few rows of the generated data:")
    print(data.head())
    print("\nDataset Info:")
    print(data.info())

if __name__ == "__main__":
    main()
