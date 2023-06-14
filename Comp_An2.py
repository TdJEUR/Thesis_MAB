import pandas as pd
import matplotlib.pyplot as plt


# path = "C:/Users/tommo/Downloads/Thesis_MAB/Data/df_B_125mabs_1sims_10arms_test3.xlsx"
# path2 = "C:/Users/tommo/Downloads/Thesis_MAB/Data/df_CP_125mabs_1sims_10arms_test3.xlsx"
path = "C:/Users/tommo/Downloads/Thesis_MAB/Data/df_B_400mabs_1sims_10arms_FINAL.xlsx"
path2 = "C:/Users/tommo/Downloads/Thesis_MAB/Data/df_CP_400mabs_1sims_10arms_FINAL.xlsx"

# Load the DataFrame from the first Excel file
dfB = pd.read_excel(path)
dfB = dfB.drop('Alpha', axis=1)

# Load the DataFrame from the second Excel file
dfCP = pd.read_excel(path2)
dfCP = dfCP.drop('Alpha', axis=1)

# Round the 'STD' and 'Avg_Alpha' columns to four decimal places for the first DataFrame
dfB[['STD', 'Avg_Alpha']] = dfB[['STD', 'Avg_Alpha']].round(4)

# Round the 'STD' and 'Avg_Alpha' columns to four decimal places for the second DataFrame
dfCP[['STD', 'Avg_Alpha']] = dfCP[['STD', 'Avg_Alpha']].round(4)

# Round the 'Skew' column to eight decimal places for the first DataFrame
dfB['Skew'] = dfB['Skew'].round(8)

# Round the 'Skew' column to eight decimal places for the second DataFrame
dfCP['Skew'] = dfCP['Skew'].round(8)

# Group and average the rows for Avg_Alpha for the first DataFrame
dfB_avg_alpha = dfB.groupby('Avg_Alpha').mean().reset_index()
dfB_avg_alpha = dfB_avg_alpha[['Avg_Alpha'] + [col for col in dfB_avg_alpha.columns if col.startswith('Tau')]]

# Group and average the rows for STD for the first DataFrame
dfB_std = dfB.groupby('STD').mean().reset_index()
dfB_std = dfB_std[['STD'] + [col for col in dfB_std.columns if col.startswith('Tau')]]

# Group and average the rows for Skew for the first DataFrame
dfB_skew = dfB.groupby('Skew').mean().reset_index()
dfB_skew = dfB_skew[['Skew'] + [col for col in dfB_skew.columns if col.startswith('Tau')]]

# Group and average the rows for Avg_Alpha for the second DataFrame
dfCP_avg_alpha = dfCP.groupby('Avg_Alpha').mean().reset_index()
dfCP_avg_alpha = dfCP_avg_alpha[['Avg_Alpha'] + [col for col in dfCP_avg_alpha.columns if col.startswith('Tau')]]

# Group and average the rows for STD for the second DataFrame
dfCP_std = dfCP.groupby('STD').mean().reset_index()
dfCP_std = dfCP_std[['STD'] + [col for col in dfCP_std.columns if col.startswith('Tau')]]

# Group and average the rows for Skew for the second DataFrame
dfCP_skew = dfCP.groupby('Skew').mean().reset_index()
dfCP_skew = dfCP_skew[['Skew'] + [col for col in dfCP_skew.columns if col.startswith('Tau')]]

# Create subplots for Avg_Alpha
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

for col in dfB_avg_alpha.columns[1:]:
    ax1.plot(dfB_avg_alpha['Avg_Alpha'], dfB_avg_alpha[col], label=col)

for col in dfCP_avg_alpha.columns[1:]:
    ax2.plot(dfCP_avg_alpha['Avg_Alpha'], dfCP_avg_alpha[col], label=col)

ax1.set_xlabel('Average Alpha')
ax1.set_ylabel('Reward')
ax1.set_title('Averaging Beliefs')
ax1.legend()

ax2.set_xlabel('Average Alpha')
ax2.set_ylabel('Reward')
ax2.set_title('Averaging CPs')
ax2.legend()

# Adjust the spacing between subplots
plt.tight_layout()

# Create subplots for STD
fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 6))

for col in dfB_std.columns[1:]:
    ax3.plot(dfB_std['STD'], dfB_std[col], label=col)

for col in dfCP_std.columns[1:]:
    ax4.plot(dfCP_std['STD'], dfCP_std[col], label=col)

ax3.set_xlabel('Standard Deviation')
ax3.set_ylabel('Reward')
ax3.set_title('Averaging Beliefs')
ax3.legend()

ax4.set_xlabel('Standard Deviation')
ax4.set_ylabel('Reward')
ax4.set_title('Averaging CPs')
ax4.legend()

# Adjust the spacing between subplots
plt.tight_layout()

# Create subplots for Skew
fig, (ax5, ax6) = plt.subplots(1, 2, figsize=(12, 6))

for col in dfB_skew.columns[1:]:
    ax5.plot(dfB_skew['Skew'], dfB_skew[col], label=col)

for col in dfCP_skew.columns[1:]:
    ax6.plot(dfCP_skew['Skew'], dfCP_skew[col], label=col)

ax5.set_xlabel('Skew')
ax5.set_ylabel('Reward')
ax5.set_title('Averaging Beliefs')
ax5.legend()

ax6.set_xlabel('Skew')
ax6.set_ylabel('Reward')
ax6.set_title('Averaging CPs')
ax6.legend()

# Adjust the spacing between subplots
plt.tight_layout()

# Display the plots
plt.show()
