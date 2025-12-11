import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import yaml

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score

# --- 1. CONFIGURATION ---

# Load path from config.yaml
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Replaces the hardcoded line 15
INPUT_FILE = config['input_file']

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
    Creates derived features (ratios) to improve model accuracy.
    """
    print("    > Engineering features (Ratios & Interactions)...")
    df_eng = df.copy()

    # We check the necessary columns needed to calculate the ratios are present
    required_for_ratios = [
        'Policy Face Value', 'Household Annual Income',
        'Annualized Premiums', 'Net Worth', 'Market Value of Home'
    ]
    for col in required_for_ratios:
        if col not in df_eng.columns:
            raise KeyError(f"Feature Engineering ERROR: Required column '{col}' is missing for ratio calculation.")

    # 1. Coverage Ratio: Face Value / Income
    # Helps model see "Underinsured" people easily
    df_eng['Ratio_Coverage'] = df_eng['Policy Face Value'] / df_eng['Household Annual Income'].replace(0, 1)

    # 2. Premium Burden: Premiums / Income
    # Helps model see "Affordability"
    df_eng['Ratio_Premium_Burden'] = df_eng['Annualized Premiums'] / df_eng['Household Annual Income'].replace(0, 1)

    # 3. Asset Wealth: Home Value / Net Worth
    # Helps distinguish liquid vs illiquid wealth
    df_eng['Ratio_Asset_Concentration'] = df_eng['Market Value of Home'] / df_eng['Net Worth'].replace(0, 1)

    return df_eng.loc[:, ['Ratio_Coverage', 'Ratio_Premium_Burden', 'Ratio_Asset_Concentration']] # Return only the new ratios

def generate_propensity_target(df):
    """
    Engineers 'Propensity to Buy' target based on logic: "Underinsured but Wealthy".
    """
    print("    > Generating 'Propensity to Buy' labels...")

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

    print(f"    > Target Distribution: {np.mean(propensity_target)*100:.1f}% Propensity to Buy")
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
    print(f"    > Saved Confusion Matrix: {save_path}")

# --- MAIN PIPELINE ---
if __name__ == "__main__":
    print("--- STARTING PROPENSITY MODEL PIPELINE (Random Forest - Recall Optimized Only) ---")

    # 1. Load Data
    df_raw = load_data(INPUT_FILE)

    # 2. Basic Cleaning
    print("    > Cleaning/Imputing raw features...")
    
    # Check for missing RAW_FEATURE_COLS (critical check)
    missing_raw_cols = [col for col in RAW_FEATURE_COLS if col not in df_raw.columns]
    if missing_raw_cols:
        raise KeyError(f"CRITICAL ERROR: The following RAW_FEATURE_COLS are missing from the dataset: {missing_raw_cols}")
        
    for col in RAW_FEATURE_COLS:
        # Impute NaNs in the core numerical columns with the column median
        df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce').fillna(df_raw[col].median())

    # 3. Generate Target
    y = generate_propensity_target(df_raw)

    # 4. Feature Engineering (Generate only the new ratio features)
    df_ratios = engineer_features(df_raw[RAW_FEATURE_COLS])

    # 5. Combine All Features
    # Use the RAW_FEATURE_COLS as the raw features
    X_raw = df_raw[RAW_FEATURE_COLS].reset_index(drop=True)

    # Combine all raw features with the engineered ratio features
    X_enhanced = pd.concat([X_raw, df_ratios.reset_index(drop=True)], axis=1)

    print(f"    > Training with {X_enhanced.shape[1]} features (Raw + Ratios)")

    # 6. Split Data (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X_enhanced, y, test_size=0.20, random_state=42, stratify=y)

    # 7. Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- MODEL: RANDOM FOREST (Recall Optimized) ---
    print("\n--- MODEL: RANDOM FOREST (Recall Optimized) ---")
    # Uses class_weight='balanced' to force the model to prioritize the minority class (1=Buy)
    rf_rec_model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, class_weight='balanced')
    rf_rec_model.fit(X_train_scaled, y_train)

    y_pred_rf_rec = rf_rec_model.predict(X_test_scaled)
    rf_rec_acc = accuracy_score(y_test, y_pred_rf_rec)
    rf_rec_rec = recall_score(y_test, y_pred_rf_rec)
    print(f"    > Accuracy: {rf_rec_acc:.2%}, Recall: {rf_rec_rec:.2%}")
    print(classification_report(y_test, y_pred_rf_rec))
    plot_confusion_matrix(y_test, y_pred_rf_rec, 'CM: Random Forest (Recall Optimized)', 'propensity_rf_recall_confusion.png')

    # 8. Save Model
    best_model_name = 'Random Forest (Recall Optimized)'
    best_model = rf_rec_model

    joblib.dump(best_model, os.path.join(OUTPUT_DIR, 'best_propensity_model.pkl'))
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, 'propensity_scaler.pkl'))

    print("\n--- PIPELINE COMPLETE ---")
    print("----------------------------------------------------------------------")
    print(f"    > FINAL MODEL SAVED: {best_model_name}")
    print(f"    > Final Recall: {rf_rec_rec:.2%}")
    print(f"    > Final Accuracy: {rf_rec_acc:.2%}")
    print(f"    > Output directory: {OUTPUT_DIR}")
    print("----------------------------------------------------------------------")

    # --- 9. FEATURE IMPORTANCE ANALYSIS ---
    print("\n--- CALCULATING FEATURE IMPORTANCE ---")

    # Get feature names from the dataframe we created earlier
    feature_names = X_enhanced.columns

    # Create a DataFrame to hold importance data
    importance_df = pd.DataFrame({'Feature': feature_names})

    # Get feature importance (Random Forest logic)
    importance_df['Importance'] = best_model.feature_importances_
    title = f'Feature Importance: {best_model_name}'

    # Sort by absolute value to see the biggest drivers (positive or negative)
    importance_df['Abs_Importance'] = importance_df['Importance'].abs()
    importance_df = importance_df.sort_values(by='Abs_Importance', ascending=False)

    # Display the top drivers in the console
    print(importance_df[['Feature', 'Importance']].head(10))

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')

    plt.title(title, fontsize=15)
    plt.xlabel('Impact on Propensity (Importance)', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1) # Zero line for reference
    plt.tight_layout()

    # Save the plot
    importance_path = os.path.join(OUTPUT_DIR, 'feature_importance_best_model.png')
    plt.savefig(importance_path)
    plt.close()
    print(f"    > Saved Feature Importance Plot: {importance_path}")
