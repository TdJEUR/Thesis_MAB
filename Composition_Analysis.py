import pandas as pd
import matplotlib.pyplot as plt


path = "C:/Users/tommo/Downloads/Thesis_MAB/Data/df_B_125mabs_1sims_10arms_test3.xlsx"
path2 = "C:/Users/tommo/Downloads/Thesis_MAB/Data/df_CP_125mabs_1sims_10arms_test3.xlsx"

# Load the Excel file into a pandas DataFrame
df = pd.read_excel(path2)

# Extract the Tau columns
tau_columns = [col for col in df.columns if col.startswith('Tau=')]


# Round 'Avg_Alpha', 'STD', and 'Skew' columns to the desired decimal places
df['Avg_Alpha'] = df['Avg_Alpha'].round(4)
df['STD'] = df['STD'].round(4)
df['Skew'] = df['Skew'].round(8)

# Select six rows at similar intervals for each variable plot
interval = len(df) // 6
selected_rows = df.iloc[::interval]

# Create a line plot for Avg_Alpha
fig, ax = plt.subplots()
for index, row in selected_rows.iterrows():
    avg_alpha = row['Avg_Alpha']
    avg_alpha_values = row[tau_columns]
    ax.plot(avg_alpha_values, label=f'Avg_Alpha={avg_alpha}')
ax.set_xlabel('Tau')
ax.set_ylabel('Avg_Alpha')
ax.legend()
plt.show()

# Create a line plot for STD
fig, ax = plt.subplots()
for index, row in selected_rows.iterrows():
    std = row['STD']
    std_values = row[tau_columns]
    ax.plot(std_values, label=f'STD={std}')
ax.set_xlabel('Tau')
ax.set_ylabel('STD')
ax.legend()
plt.show()

# Create a line plot for Skew
fig, ax = plt.subplots()
for index, row in selected_rows.iterrows():
    skew = row['Skew']
    skew_values = row[tau_columns]
    ax.plot(skew_values, label=f'Skew={skew}')
ax.set_xlabel('Tau')
ax.set_ylabel('Skew')
ax.legend()
plt.show()