import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns

def develop_score_tool(data):
    """
    Develop a simple scoring system based on key predictors
    """
    # Select key predictors based on clinical relevance and importance
    predictors = [
        'Age',
        'Albumin',
        'INR',
        'BMI',
        'Tumor_Size',
        'Num_Tumors',
        'ECOG_Score'
    ]
    
    # Create point system
    def calculate_risk_score(row):
        score = 0
        
        # Age points (>65 years)
        if row['Age'] > 65:
            score += 1
            
        # Albumin points (<3.5 g/dL)
        if row['Albumin'] < 3.5:
            score += 2
            
        # INR points (>1.2)
        if row['INR'] > 1.2:
            score += 2
            
        # BMI points (>30)
        if row['BMI'] > 30:
            score += 1
            
        # Tumor Size points (>3cm)
        if row['Tumor_Size'] > 3:
            score += 1
            
        # Multiple tumors
        if row['Num_Tumors'] > 1:
            score += 1
            
        # ECOG Score points
        if row['ECOG_Score'] >= 2:
            score += 2
            
        return score
    
    # Calculate scores for each patient
    data['Risk_Score'] = data.apply(calculate_risk_score, axis=1)
    
    # Calculate risk by score
    risk_by_score = pd.DataFrame({
        'Total_Patients': data.groupby('Risk_Score').size(),
        'Complications': data.groupby('Risk_Score')['Complications'].sum()
    }).reset_index()
    
    risk_by_score['Risk_Percentage'] = (risk_by_score['Complications'] / 
                                      risk_by_score['Total_Patients'] * 100).round(1)
    
    return risk_by_score

def plot_risk_score_distribution(risk_by_score):
    """Plot risk score distribution and complications"""
    plt.figure(figsize=(12, 6))
    
    # Create bar plot
    plt.bar(risk_by_score['Risk_Score'], risk_by_score['Risk_Percentage'],
            alpha=0.8, color='lightcoral')
    
    # Customize plot
    plt.title('IR Procedure Risk Score', fontsize=14, pad=20)
    plt.xlabel('Risk Score', fontsize=12)
    plt.ylabel('Complication Rate (%)', fontsize=12)
    
    # Add value labels on bars
    for i, v in enumerate(risk_by_score['Risk_Percentage']):
        plt.text(risk_by_score['Risk_Score'][i], v + 0.5, 
                f'{v}%\nn={risk_by_score["Total_Patients"][i]}', 
                ha='center', fontsize=10)
    
    plt.grid(True, alpha=0.3)
    plt.savefig('risk_score_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def print_scoring_system():
    """Print the scoring system explanation"""
    print("\nIR-SCORE: Interventional Radiology Safety and Complication Outcome Risk Evaluation")
    print("\nScoring System:")
    print("1. Age > 65 years: +1 point")
    print("2. Albumin < 3.5 g/dL: +2 points")
    print("3. INR > 1.2: +2 points")
    print("4. BMI > 30: +1 point")
    print("5. Tumor Size > 3cm: +1 point")
    print("6. Multiple Tumors: +1 point")
    print("7. ECOG Score ≥ 2: +2 points")
    print("\nMaximum possible score: 10 points")

def main():
    # Load data
    print("Loading data and developing scoring system...")
    data = pd.read_csv('synthetic_ir_data.csv')
    
    # Develop and analyze risk score
    risk_by_score = develop_score_tool(data)
    
    # Print scoring system
    print_scoring_system()
    
    # Print risk levels
    print("\nRisk Levels by Score:")
    print(risk_by_score.to_string(index=False))
    
    # Plot risk distribution
    plot_risk_score_distribution(risk_by_score)
    print("\nRisk score distribution plot saved as 'risk_score_distribution.png'")
    
    # Calculate and print risk categories
    print("\nRisk Categories:")
    print("Low Risk: 0-3 points")
    print("Moderate Risk: 4-6 points")
    print("High Risk: 7-10 points")

if __name__ == "__main__":
    main()
