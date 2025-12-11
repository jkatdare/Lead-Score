import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier # Upgraded from DecisionTree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score

# --- 1. CONFIGURATION ---
INPUT_FILE = 'synthetic_data.xlsx'
OUTPUT_DIR = os.path.join('files', 'PropensityModel')

# Base columns from the file; add additional columns and upload everything to github
RAW_FEATURE_COLS = [
    'Net Worth', 
    'Household Annual Income', 
    'Market Value of Home', 
    'Annualized Premiums', 
    'Policy Face Value'
]

# Ensure output directory exists
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Plot styling
plt.style.use('ggplot')

def load_data(filepath):
    """Safely loads data."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"CRITICAL ERROR: '{filepath}' not found.")
    print(f"READING: Loading '{filepath}'...")
    df = pd.read_excel(filepath)
    return df

def engineer_features(df):
    """
    Creates derived features to improve model accuracy.
    """
    print("   > Engineering features (Ratios & Interactions)...")
    df_eng = df.copy()
    
    # 1. Coverage Ratio: Face Value / Income
    # Helps model see "Underinsured" people easily
    df_eng['Ratio_Coverage'] = df_eng['Policy Face Value'] / df_eng['Household Annual Income'].replace(0, 1)
    
    # 2. Premium Burden: Premiums / Income
    # Helps model see "Affordability"
    df_eng['Ratio_Premium_Burden'] = df_eng['Annualized Premiums'] / df_eng['Household Annual Income'].replace(0, 1)
    
    # 3. Asset Wealth: Home Value / Net Worth
    # Helps distinguish liquid vs illiquid wealth
    df_eng['Ratio_Asset_Concentration'] = df_eng['Market Value of Home'] / df_eng['Net Worth'].replace(0, 1)

    return df_eng

def generate_propensity_target(df):
    """
    Engineers 'Propensity to Buy' target based on logic: "Underinsured but Wealthy".
    """
    print("   > Generating 'Propensity to Buy' labels...")
    
    # Logic 1: Underinsured Gap
    ideal_coverage = df['Household Annual Income'] * 10
    coverage_ratio = df['Policy Face Value'] / ideal_coverage.replace(0, 1)
    
    # Logic 2: Affordability
    premium_burden = df['Annualized Premiums'] / df['Household Annual Income'].replace(0, 1)
    
    # Logic 3: Wealth Rank
    wealth_score = df['Net Worth'].rank(pct=True)

    # --- SCORING ---
    score = (
        (1 - coverage_ratio) * 40 +   
        (1 - premium_burden) * 20 +   
        (wealth_score) * 30           
    )
    
    # Reduced noise slightly (15 -> 10) to improve learnability
    noise = np.random.normal(0, 10, size=len(df))
    final_score = score + noise
    
    # Target: Top 25%
    threshold = np.percentile(final_score, 75)
    propensity_target = (final_score > threshold).astype(int)
    
    print(f"   > Target Distribution: {np.mean(propensity_target)*100:.1f}% Propensity to Buy")
    return propensity_target

def plot_confusion_matrix(y_true, y_pred, title, filename):
    """Generates a robust confusion matrix visualization."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=False, annot_kws={"size": 14})
    plt.title(title, fontsize=14)
    plt.ylabel('Actual (0=No, 1=Buy)', fontsize=12)
    plt.xlabel('Predicted (0=No, 1=Buy)', fontsize=12)
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"   > Saved Confusion Matrix: {save_path}")

# --- MAIN PIPELINE ---
if __name__ == "__main__":
    print("--- STARTING PROPENSITY MODEL PIPELINE (Optimized) ---")

    # 1. Load Data
    df_raw = load_data(INPUT_FILE)
    
    # 2. Basic Cleaning
    for col in RAW_FEATURE_COLS:
        if col not in df_raw.columns:
            print(f"ERROR: Missing column '{col}'")
            exit()
        df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce').fillna(df_raw[col].median())

    # 3. Generate Target
    y = generate_propensity_target(df_raw)
    
    # 4. Feature Engineering (New Step)
    # We pass the raw data with the 5 columns to create new ratios
    X_enhanced = engineer_features(df_raw[RAW_FEATURE_COLS])
    
    print(f"   > Training with {X_enhanced.shape[1]} features (Raw + Ratios)")

    # 5. Split Data (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X_enhanced, y, test_size=0.20, random_state=42, stratify=y)
    
    # 6. Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\n--- MODEL 1: LOGISTIC REGRESSION (Balanced) ---")
    lr_model = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
    lr_model.fit(X_train_scaled, y_train)
    
    y_pred_lr = lr_model.predict(X_test_scaled)
    lr_acc = accuracy_score(y_test, y_pred_lr)
    lr_rec = recall_score(y_test, y_pred_lr)
    print(f"   > Accuracy: {lr_acc:.2%}, Recall: {lr_rec:.2%}")
    print(classification_report(y_test, y_pred_lr))
    plot_confusion_matrix(y_test, y_pred_lr, 'CM: Logistic Regression (Balanced)', 'propensity_lr_confusion.png')

    # --- MODEL 2: RANDOM FOREST (Recall Optimized) ---
    print("\n--- MODEL 2: RANDOM FOREST (Recall Optimized) ---")
    # Uses class_weight='balanced' to force the model to prioritize the minority class (1=Buy)
    rf_rec_model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, class_weight='balanced')
    rf_rec_model.fit(X_train_scaled, y_train)
    
    y_pred_rf_rec = rf_rec_model.predict(X_test_scaled)
    rf_rec_acc = accuracy_score(y_test, y_pred_rf_rec)
    rf_rec_rec = recall_score(y_test, y_pred_rf_rec)
    print(f"   > Accuracy: {rf_rec_acc:.2%}, Recall: {rf_rec_rec:.2%}")
    print(classification_report(y_test, y_pred_rf_rec))
    plot_confusion_matrix(y_test, y_pred_rf_rec, 'CM: Random Forest (Recall Optimized)', 'propensity_rf_recall_confusion.png')


    # --- MODEL 3: RANDOM FOREST (Accuracy Optimized) ---
    print("\n--- MODEL 3: RANDOM FOREST (Accuracy Optimized) ---")
    # Uses default class_weight=None, maximizing overall correct predictions.
    rf_acc_model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, class_weight=None)
    rf_acc_model.fit(X_train_scaled, y_train)
    
    y_pred_rf_acc = rf_acc_model.predict(X_test_scaled)
    rf_acc_acc = accuracy_score(y_test, y_pred_rf_acc)
    rf_acc_rec = recall_score(y_test, y_pred_rf_acc)
    print(f"   > Accuracy: {rf_acc_acc:.2%}, Recall: {rf_acc_rec:.2%}")
    print(classification_report(y_test, y_pred_rf_acc))
    plot_confusion_matrix(y_test, y_pred_rf_acc, 'CM: Random Forest (Accuracy Optimized)', 'propensity_rf_accuracy_confusion.png')
    
    # 7. Save Best Model (Selecting based on highest Recall, as this is a marketing task)
    
    # We will prioritize the model with the best Recall, as missing a potential buyer (False Negative) 
    # is often more costly than contacting a non-buyer (False Positive) in sales prospecting.
    models = {
        'Logistic Regression (Balanced)': {'model': lr_model, 'recall': lr_rec, 'accuracy': lr_acc},
        'Random Forest (Recall)': {'model': rf_rec_model, 'recall': rf_rec_rec, 'accuracy': rf_rec_acc},
        'Random Forest (Accuracy)': {'model': rf_acc_model, 'recall': rf_acc_rec, 'accuracy': rf_acc_acc},
    }
    
    # Find the model with the highest Recall
    best_recall_model = max(models.items(), key=lambda item: item[1]['recall'])
    
    best_model_name = best_recall_model[0]
    best_model = best_recall_model[1]['model']
    
    joblib.dump(best_model, os.path.join(OUTPUT_DIR, 'best_propensity_model.pkl'))
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, 'propensity_scaler.pkl'))
    
    print("\n--- PIPELINE COMPLETE ---")
    print("----------------------------------------------------------------------")
    print(f"   > BEST MODEL SAVED (Prioritizing RECALL): {best_model_name}")
    print(f"   > Final Recall: {best_recall_model[1]['recall']:.2%}")
    print(f"   > Final Accuracy: {best_recall_model[1]['accuracy']:.2%}")
    print("   > Output directory: {OUTPUT_DIR}")
    print("----------------------------------------------------------------------")
