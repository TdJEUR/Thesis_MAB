import pandas as pd
import matplotlib.pyplot as plt

# path = "C:/Users/tommo/Downloads/Thesis_MAB/Data/df_B_50mabs_1sims_10arms_largeA2.xlsx"
# path2 = "C:/Users/tommo/Downloads/Thesis_MAB/Data/df_CP_50mabs_1sims_10arms_largeA2.xlsx"
# path = "C:/Users/tommo/Downloads/Thesis_MAB/Data/df_B_125mabs_1sims_10arms_test3.xlsx"
# path2 = "C:/Users/tommo/Downloads/Thesis_MAB/Data/df_CP_125mabs_1sims_10arms_test3.xlsx"
path = "C:/Users/tommo/Downloads/Thesis_MAB/Data/df_B_400mabs_1sims_10arms_FINAL.xlsx"
path2 = "C:/Users/tommo/Downloads/Thesis_MAB/Data/df_CP_400mabs_1sims_10arms_FINAL.xlsx"

# Load the Excel file and read the data into a DataFrame
dfB = pd.read_excel(path)

# Round the required columns to the specified number of decimals
dfB['Avg_Alpha'] = dfB['Avg_Alpha'].round(4)
dfB['STD'] = dfB['STD'].round(4)
dfB['Skew'] = dfB['Skew'].round(6)

# Extract the required columns for each variable
dfB_avg_alpha = dfB[['Tau=0.0', 'Tau=0.00556', 'Tau=0.01111', 'Tau=0.01667', 'Tau=0.02222',
                     'Tau=0.02778', 'Tau=0.03333', 'Tau=0.03889', 'Tau=0.04444', 'Tau=0.05', 'Avg_Alpha']]
dfB_std = dfB[['Tau=0.0', 'Tau=0.00556', 'Tau=0.01111', 'Tau=0.01667', 'Tau=0.02222',
                'Tau=0.02778', 'Tau=0.03333', 'Tau=0.03889', 'Tau=0.04444', 'Tau=0.05', 'STD']]
dfB_skew = dfB[['Tau=0.0', 'Tau=0.00556', 'Tau=0.01111', 'Tau=0.01667', 'Tau=0.02222',
                 'Tau=0.02778', 'Tau=0.03333', 'Tau=0.03889', 'Tau=0.04444', 'Tau=0.05', 'Skew']]

# Average rows with the same value for each variable
dfB_avg_alpha = dfB_avg_alpha.groupby('Avg_Alpha').mean().reset_index()
dfB_std = dfB_std.groupby('STD').mean().reset_index()
dfB_skew = dfB_skew.groupby('Skew').mean().reset_index()

# Load the second Excel file and read the data into a DataFrame
dfCP = pd.read_excel(path2)

# Round the required columns to the specified number of decimals
dfCP['Avg_Alpha'] = dfCP['Avg_Alpha'].round(4)
dfCP['STD'] = dfCP['STD'].round(4)
dfCP['Skew'] = dfCP['Skew'].round(6)

# Extract the required columns for each variable
dfCP_avg_alpha = dfCP[['Tau=0.0', 'Tau=0.00556', 'Tau=0.01111', 'Tau=0.01667', 'Tau=0.02222',
                       'Tau=0.02778', 'Tau=0.03333', 'Tau=0.03889', 'Tau=0.04444', 'Tau=0.05', 'Avg_Alpha']]
dfCP_std = dfCP[['Tau=0.0', 'Tau=0.00556', 'Tau=0.01111', 'Tau=0.01667', 'Tau=0.02222',
                  'Tau=0.02778', 'Tau=0.03333', 'Tau=0.03889', 'Tau=0.04444', 'Tau=0.05', 'STD']]
dfCP_skew = dfCP[['Tau=0.0', 'Tau=0.00556', 'Tau=0.01111', 'Tau=0.01667', 'Tau=0.02222',
                   'Tau=0.02778', 'Tau=0.03333', 'Tau=0.03889', 'Tau=0.04444', 'Tau=0.05', 'Skew']]

# Average rows with the same value for each variable
dfCP_avg_alpha = dfCP_avg_alpha.groupby('Avg_Alpha').mean().reset_index()
dfCP_std = dfCP_std.groupby('STD').mean().reset_index()
dfCP_skew = dfCP_skew.groupby('Skew').mean().reset_index()

# Determine the number of lines to plot in each variable plot
num_lines = 10

# Calculate the index intervals for the lines to plot
index_interval1_avg_alpha = len(dfB_avg_alpha) // (num_lines - 1)
index_interval2_avg_alpha = len(dfCP_avg_alpha) // (num_lines - 1)
index_interval1_std = len(dfB_std) // (num_lines - 1)
index_interval2_std = len(dfCP_std) // (num_lines - 1)
index_interval1_skew = len(dfB_skew) // (num_lines - 1)
index_interval2_skew = len(dfCP_skew) // (num_lines - 1)

# Plot for Avg_Alpha
fig1, axs1 = plt.subplots(1, 2, figsize=(16, 6))

# Plot for Avg_Alpha (Excel 1)
for i in range(num_lines):
    index = i * index_interval1_avg_alpha
    row = dfB_avg_alpha.iloc[index]
    axs1[0].plot(row[1:], label=f"Avg_Alpha: {row['Avg_Alpha']}")
axs1[0].set_xlabel('Tau')
axs1[0].set_ylabel('Reward')
axs1[0].set_title('Averaging Beliefs')
axs1[0].legend()
axs1[0].tick_params(axis='x', rotation=45)

# Plot for Avg_Alpha (Excel 2)
for i in range(num_lines):
    index = i * index_interval2_avg_alpha
    row = dfCP_avg_alpha.iloc[index]
    axs1[1].plot(row[1:], label=f"Avg_Alpha: {row['Avg_Alpha']}")
axs1[1].set_xlabel('Tau')
axs1[1].set_ylabel('Reward')
axs1[1].set_title('Averaging CPs')
axs1[1].legend()
axs1[1].tick_params(axis='x', rotation=45)

# Plot for STD
fig2, axs2 = plt.subplots(1, 2, figsize=(16, 6))

# Plot for STD (Excel 1)
for i in range(num_lines):
    index = i * index_interval1_std
    row = dfB_std.iloc[index]
    axs2[0].plot(row[1:], label=f"STD: {row['STD']}")
axs2[0].set_xlabel('Tau')
axs2[0].set_ylabel('Reward')
axs2[0].set_title('Averaging Beliefs')
axs2[0].legend()
axs2[0].tick_params(axis='x', rotation=45)

# Plot for STD (Excel 2)
for i in range(num_lines):
    index = i * index_interval2_std
    row = dfCP_std.iloc[index]
    axs2[1].plot(row[1:], label=f"STD: {row['STD']}")
axs2[1].set_xlabel('Tau')
axs2[1].set_ylabel('Reward')
axs2[1].set_title('Averaging CPs')
axs2[1].legend()
axs2[1].tick_params(axis='x', rotation=45)

# Plot for Skew
fig3, axs3 = plt.subplots(1, 2, figsize=(16, 6))

# Plot for Skew (Excel 1)
for i in range(num_lines):
    index = i * index_interval1_skew
    row = dfB_skew.iloc[index]
    axs3[0].plot(row[1:], label=f"Skew: {row['Skew']}")
axs3[0].set_xlabel('Tau')
axs3[0].set_ylabel('Reward')
axs3[0].set_title('Averaging Beliefs')
axs3[0].legend()
axs3[0].tick_params(axis='x', rotation=45)

# Plot for Skew (Excel 2)
for i in range(num_lines):
    index = i * index_interval2_skew
    row = dfCP_skew.iloc[index]
    axs3[1].plot(row[1:], label=f"Skew: {row['Skew']}")
axs3[1].set_xlabel('Tau')
axs3[1].set_ylabel('Reward')
axs3[1].set_title('Averaging CPs')
axs3[1].legend()
axs3[1].tick_params(axis='x', rotation=45)

# Adjust the spacing between subplots
fig1.tight_layout()
fig2.tight_layout()
fig3.tight_layout()

# Display the plots
plt.show()