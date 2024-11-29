import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from imblearn.over_sampling import SMOTE

def prepare_data(data):
    """Prepare data for machine learning"""
    # Create copy to avoid modifying original data
    df = data.copy()
    
    # Encode categorical variables
    le = LabelEncoder()
    categorical_cols = ['Gender', 'Intervention_Type', 'Response']
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])
    
    # Handle missing values in Ablation_Duration properly
    df = df.assign(
        Ablation_Duration=df['Ablation_Duration'].fillna(df['Ablation_Duration'].mean())
    )
    
    # Scale numerical features
    scaler = StandardScaler()
    numerical_cols = ['Age', 'BMI', 'Tumor_Size', 'Bilirubin', 'Albumin', 'INR', 'Ablation_Duration']
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    return df

def train_prediction_model(data):
    """Train and evaluate the prediction model"""
    # Prepare features and target
    X = data.drop(['Complications'], axis=1)
    y = data['Complications']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Apply SMOTE to handle class imbalance
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    # Train model with balanced data
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42
    )
    model.fit(X_train_balanced, y_train_balanced)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    
    return model, X_test, y_test, y_pred, y_pred_prob, X.columns

def plot_feature_importance(model, feature_names):
    """Plot feature importance with error bars"""
    importances = pd.Series(model.feature_importances_, index=feature_names)
    std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
    
    plt.figure(figsize=(12, 8))
    importances.sort_values().plot(kind='barh', xerr=std)
    plt.title('Feature Importance in Predicting Complications')
    plt.xlabel('Importance Score (with standard deviation)')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return importances

def plot_confusion_matrix(y_test, y_pred):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("Loading and preparing data...")
    data = pd.read_csv('synthetic_ir_data.csv')
    processed_data = prepare_data(data)
    
    print("\nData distribution:")
    print(data['Complications'].value_counts(normalize=True))
    
    print("\nTraining model and generating predictions...")
    model, X_test, y_test, y_pred, y_pred_prob, feature_names = train_prediction_model(processed_data)
    
    print("\nModel Performance:")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # Plot and print feature importance
    print("\nGenerating feature importance plot...")
    importances = plot_feature_importance(model, feature_names)
    print("\nTop 5 Most Important Features:")
    print(importances.sort_values(ascending=False).head())
    
    # Plot confusion matrix
    print("\nGenerating confusion matrix...")
    plot_confusion_matrix(y_test, y_pred)
    
    print("\nAnalysis complete. Visualizations saved as:")
    print("- feature_importance.png")
    print("- confusion_matrix.png")
    
    # Print some model metrics
    sensitivity = confusion_matrix(y_test, y_pred)[1,1] / (confusion_matrix(y_test, y_pred)[1,0] + confusion_matrix(y_test, y_pred)[1,1])
    specificity = confusion_matrix(y_test, y_pred)[0,0] / (confusion_matrix(y_test, y_pred)[0,0] + confusion_matrix(y_test, y_pred)[0,1])
    print(f"\nSensitivity (True Positive Rate): {sensitivity:.2f}")
    print(f"Specificity (True Negative Rate): {specificity:.2f}")

if __name__ == "__main__":
    main()
