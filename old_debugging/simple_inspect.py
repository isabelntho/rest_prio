"""
Simple inspection of CSVs to diagnose spatial agreement issue
"""
import os
import pandas as pd

# Check the CSVs that were already created
csv_path = 'multi_seed_results/comparison_to_base_fg_2303_1.csv'
print(f"Reading CSV: {csv_path}")
print(f"File exists: {os.path.exists(csv_path)}\n")

df = pd.read_csv(csv_path)

print(f"DataFrame shape: {df.shape}")
print(f"Columns: {list(df.columns)}\n")

print("First few rows:")
print(df[['experiment_id', 'spatial_agreement_vs_base']].head(20))

print("\n" + "="*60)
print("Summary of spatial_agreement_vs_base:")
print("="*60)
print(f"Non-null count: {df['spatial_agreement_vs_base'].notna().sum()}")
print(f"Null count: {df['spatial_agreement_vs_base'].isna().sum()}")
print(f"Type: {df['spatial_agreement_vs_base'].dtype}")
print(f"Values:\n{df['spatial_agreement_vs_base'].value_counts(dropna=False)}")

# Check parameter effects CSV
param_csv = 'multi_seed_results/parameter_effects_vs_base_fg_2303_1.csv'
print(f"\n\nReading parameter effects CSV: {param_csv}")
if os.path.exists(param_csv):
    param_df = pd.read_csv(param_csv)
    print(f"Shape: {param_df.shape}")
    print(f"Columns: {list(param_df.columns)}\n")
    print("Spatial agreement by parameter set:")
    if 'mean_spatial_agreement' in param_df.columns:
        print(param_df[['max_restoration_fraction', 'patch_size', 'mean_spatial_agreement', 'sd_spatial_agreement']])
    else:
        print(param_df)
else:
    print(f"File does not exist!")
