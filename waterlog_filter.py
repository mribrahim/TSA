import pandas as pd
import numpy as np

root_path = "dataset/"



# Load the dataset
df_raw = pd.read_excel(root_path + "dataset_SAU_orginal.xlsx", sheet_name="Sheet1")
df_raw['Normal/Attack'] = df_raw['Normal/Attack'].map({'Normal': 0, 'Attack': 1})

# Define the window size
w = 105

# Create a new dataframe to store the processed data
processed_data = []

# Process the data in windows
for start_idx in range(0, len(df_raw), w):
    window = df_raw.iloc[start_idx:start_idx + w]
    
    # Check for attack regions
    attack_rows = window[window['Normal/Attack'] == 1]
    if not attack_rows.empty:
        # Find contiguous attack regions
        attack_regions = (attack_rows.index.to_series().diff() != 1).cumsum()
        
        # If more than one region exists, keep only the first one
        if attack_regions.nunique() > 1:
            first_region = attack_regions == 1
            rows_to_keep = attack_rows[first_region].index
            window = window[~((window['Normal/Attack'] == 1) & (~window.index.isin(rows_to_keep)))]
            
    # Append the processed window to the new dataframe
    processed_data.append(window)

# Combine all processed windows
df_processed = pd.concat(processed_data, ignore_index=True)

# Save the processed data to a new Excel file
output_path = root_path + "dataset_SAU.xlsx"
df_processed.to_excel(output_path, index=False)

print(f"Processed dataset saved to {output_path}")