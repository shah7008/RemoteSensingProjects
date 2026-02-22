"""
validate_dataset.py  –  Comprehensive Dataset Validation and Visualisation
==========================================================================
This script:
  1. Loads 'cold_chain_dataset.csv'.
  2. Validates the data (null checks, summary statistics).
  3. Saves the validation results to 'dataset_validation_results.csv'.
  4. Generates data visualisations (Correlation Heatmap, Feature Distributions).
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro
from statsmodels.tsa.stattools import adfuller

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

print(f"\n{'='*60}")
print(f"  Cold Chain Dataset Validation & Analysis")
print(f"{'='*60}\n")

# ─────────────────────────────────────────────────────────────────────────────
# 1 · Load Dataset
# ─────────────────────────────────────────────────────────────────────────────
csv_path = 'cold_chain_dataset.csv'
if not os.path.exists(csv_path):
    print(f"❌ ERROR: {csv_path} not found. Please generate the dataset first.")
    sys.exit(1)

print(f"[1/4] Loading dataset from '{csv_path}' ...")
df = pd.read_csv(csv_path)
print(f"   Loaded shape: {df.shape[0]} rows, {df.shape[1]} columns.\n")

# ─────────────────────────────────────────────────────────────────────────────
# 2 · Validate Dataset
# ─────────────────────────────────────────────────────────────────────────────
print("[2/4] Validating dataset and computing statistics ...")

# Basic validation: check for missing values
missing_values = df.isnull().sum()
total_missing = missing_values.sum()

if total_missing > 0:
    print(f"   ⚠️ Warning: Found {total_missing} missing values across the dataset.")
else:
    print("   ✅ No missing values found.")

# Summary statistics for numeric columns
numeric_df = df.select_dtypes(include=[np.number])
summary_stats = numeric_df.describe().T

# Add missing value count to summary
summary_stats['missing_count'] = missing_values[summary_stats.index]
summary_stats['missing_percent'] = (summary_stats['missing_count'] / len(df)) * 100

print("\n[2.5/4] Running advanced statistical validation ...")


stat_results = []
# 1. Normality Test (Shapiro-Wilk)
for col in numeric_df.columns:
    # Shapiro-Wilk test struggles with > 5000 samples, subsample if necessary
    data_sample = numeric_df[col].dropna().sample(min(len(numeric_df), 4999), random_state=42)
    stat, p_val = shapiro(data_sample)
    is_normal = p_val > 0.05
    
    # 2. Outliers (IQR Method)
    Q1 = numeric_df[col].quantile(0.25)
    Q3 = numeric_df[col].quantile(0.75)
    IQR = Q3 - Q1
    outlier_count = ((numeric_df[col] < (Q1 - 1.5 * IQR)) | (numeric_df[col] > (Q3 + 1.5 * IQR))).sum()
    
    stat_results.append({
        'Feature': col,
        'Shapiro_p_value': p_val,
        'Is_Normal_Dist': is_normal,
        'Outlier_Count_IQR': outlier_count
    })

stat_df = pd.DataFrame(stat_results).set_index('Feature')
summary_stats = pd.concat([summary_stats, stat_df], axis=1)

# 3. Stationarity Test (Augmented Dickey-Fuller) for time-series target
print("   Running ADF Stationarity Test on 'internal_temperature_c'...")
adf_result = adfuller(df['internal_temperature_c'].dropna())
is_stationary = adf_result[1] < 0.05
print(f"   ADF p-value: {adf_result[1]:.4f} -> Stationary: {is_stationary}")
summary_stats.loc['internal_temperature_c', 'ADF_p_value'] = adf_result[1]
summary_stats.loc['internal_temperature_c', 'Is_Stationary'] = is_stationary

# ─────────────────────────────────────────────────────────────────────────────
# 3 · Save Validation Results
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/4] Saving validation results to CSV ...")
validation_csv_path = 'dataset_validation_results.csv'
summary_stats.to_csv(validation_csv_path)

print(f"   ✅ Validation results saved to '{validation_csv_path}'.")
print("\nPreview of Statistics for Key Features:")
preview_cols = ['internal_temperature_c', 'ambient_temperature_c', 'satellite_ambient_temp', 'ndvi']
print(summary_stats.loc[preview_cols, ['mean', 'std', 'min', 'max']].to_string())

# ─────────────────────────────────────────────────────────────────────────────
# 4 · Visualise Dataset
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/4] Generating data visualisations ...")
sns.set_theme(style="whitegrid")

# ── A. Correlation Heatmap
plt.figure(figsize=(14, 10))
corr_matrix = numeric_df.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', 
            vmin=-1, vmax=1, square=True, linewidths=.5, cbar_kws={"shrink": .8})
plt.title('Feature Correlation Heatmap', fontsize=16)
plt.tight_layout()
corr_plot_path = 'dataset_correlation_heatmap.png'
plt.savefig(corr_plot_path, dpi=300)
print(f"   Saved {corr_plot_path}")

# ── B. Distribution of Key Features
features_to_plot = [
    'internal_temperature_c', 'ambient_temperature_c', 
    'satellite_ambient_temp', 'land_surface_temp', 
    'humidity_percent', 'solar_radiation'
]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Distribution of Key Continuous Features', fontsize=16)

axes = axes.flatten()
for idx, feature in enumerate(features_to_plot):
    if feature in df.columns:
        sns.histplot(df[feature], kde=True, ax=axes[idx], color='teal', bins=40)
        axes[idx].set_title(f'Distribution of {feature}', fontsize=12)
        axes[idx].set_xlabel('')
        axes[idx].set_ylabel('Frequency')

plt.tight_layout()
dist_plot_path = 'dataset_feature_distributions.png'
plt.savefig(dist_plot_path, dpi=300)
print(f"   Saved {dist_plot_path}")

# ── C. Time Series Snapshot
if 'timestamp' in df.columns:
    plt.figure(figsize=(15, 5))
    df_preview = df.head(500).copy()
    df_preview['timestamp'] = pd.to_datetime(df_preview['timestamp'])
    
    plt.plot(df_preview['timestamp'], df_preview['internal_temperature_c'], label='Internal Temp (°C)', color='red')
    plt.plot(df_preview['timestamp'], df_preview['ambient_temperature_c'], label='Ambient Temp (°C)', color='blue', alpha=0.6)
    
    plt.title('Time Series Snapshot (First 500 Samples)', fontsize=14)
    plt.xlabel('Timestamp')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    ts_plot_path = 'dataset_timeseries_snapshot.png'
    plt.savefig(ts_plot_path, dpi=300)
    print(f"   Saved {ts_plot_path}")

print(f"\n✅ Dataset validation and visualisation complete!")
print(f"{'='*60}\n")
