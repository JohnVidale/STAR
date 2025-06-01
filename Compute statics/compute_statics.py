import pandas as pd
import numpy as np

# Path to the input CSV file
file_path = "/Users/vidale/Documents/Research/STAR/Results/Statics_raw.csv"

# Read the CSV file. Lines starting with "//" are ignored.
df = pd.read_csv(file_path, comment='/', skipinitialspace=True)

# List columns to demean (exclude 'station' to keep original IDs)
cols_to_demean = ['PL1','PL2','PL3','PH1','PH2','PH3',
                  'S11','S12','S13','S21','S22','S23','SH1','SH2']

# Define station ranges to process
station_ranges = [
    (10001, 10100),
    (20001, 20100),
    (30001, 30100),
    (40001, 40100),
    (50001, 50100)
]

# Loop through each station range and demean the specified columns for rows within that range.
# Demeaning is performed and the result is immediately rounded to the nearest integer.
for low, high in station_ranges:
    mask = (df['station'] >= low) & (df['station'] <= high)
    for col in cols_to_demean:
        mean_val = df.loc[mask, col].mean()
        df.loc[mask, col] = np.around(df.loc[mask, col] - mean_val)

# Insert a new column "PL_median" to the right of PL3.
df.insert(df.columns.get_loc('PL3') + 1,
          'PL_median',
          df[['PL1', 'PL2', 'PL3']].median(axis=1))

# Insert a new column "PH_median" to the right of PH3.
df.insert(df.columns.get_loc('PH3') + 1,
          'PH_median',
          df[['PH1', 'PH2', 'PH3']].median(axis=1))

# Insert a new column "S1_median" to the right of S13.
df.insert(df.columns.get_loc('S13') + 1,
          'S1_median',
          df[['S11', 'S12', 'S13']].median(axis=1))

# Insert a new column "S2_median" to the right of S23.
df.insert(df.columns.get_loc('S23') + 1,
          'S2_median',
          df[['S21', 'S22', 'S23']].median(axis=1))

# Insert a new column "SH_median" to the right of SH2.
df.insert(df.columns.get_loc('SH2') + 1,
          'SH_median',
          df[['SH1', 'SH2']].median(axis=1))

#######################################
# Now, after a blank column, repeat PL_median and PH_median, and add their median.
# Append these new columns to the end of the DataFrame.

# Insert a blank column for the PL/PH group.
df['Blank_PL'] = ""

# Duplicate PL_median and PH_median into new columns.
df['PL_median2'] = df['PL_median']
df['PH_median2'] = df['PH_median']

# Compute the median of the repeated PL/PH medians.
df['PLPH_median'] = df[['PL_median2','PH_median2']].median(axis=1)

#######################################
# Repeat for S1, S2, and SH.
# Insert a blank column for the S group.
df['Blank_S'] = ""

# Duplicate the S-group medians.
df['S1_median2'] = df['S1_median']
df['S2_median2'] = df['S2_median']
df['SH_median2'] = df['SH_median']

# Compute the median of these three repeated columns.
df['S1S2SH_median'] = df[['S1_median2','S2_median2','SH_median2']].median(axis=1)

#######################################
# Optionally, extract each column as its own NumPy array (if needed)
station = df['station'].to_numpy()
PL1 = df['PL1'].to_numpy()
PL2 = df['PL2'].to_numpy()
PL3 = df['PL3'].to_numpy()
PL_median = df['PL_median'].to_numpy()

PH1 = df['PH1'].to_numpy()
PH2 = df['PH2'].to_numpy()
PH3 = df['PH3'].to_numpy()
PH_median = df['PH_median'].to_numpy()

S11 = df['S11'].to_numpy()
S12 = df['S12'].to_numpy()
S13 = df['S13'].to_numpy()
S1_median = df['S1_median'].to_numpy()

S21 = df['S21'].to_numpy()
S22 = df['S22'].to_numpy()
S23 = df['S23'].to_numpy()
S2_median = df['S2_median'].to_numpy()

SH1 = df['SH1'].to_numpy()
SH2 = df['SH2'].to_numpy()
SH_median = df['SH_median'].to_numpy()

# Write out a new CSV file with the updated values in the same format as the input file.
output_file = "/Users/vidale/Documents/Research/STAR/Results/Statics_demeaned.csv"
with open(output_file, "w") as f:
    f.write(f"// filepath: {output_file}\n")
    df.to_csv(f, index=False)

print(f"Demeaned file written to {output_file}")