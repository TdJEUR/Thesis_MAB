import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt

path = "C:/Users/tommo/Downloads/Thesis_MAB/Data/df_B_400mabs_1sims_10arms_FINAL.xlsx"
path2 = "C:/Users/tommo/Downloads/Thesis_MAB/Data/df_CP_400mabs_1sims_10arms_FINAL.xlsx"

# Load the Excel file and read the data into a DataFrame
dfB = pd.read_excel(path)
dfCP = pd.read_excel(path2)
dfB = dfB.drop(dfB.columns[0], axis=1)
dfCP = dfCP.drop(dfCP.columns[0], axis=1)

dfB['Skew'] = dfB['Skew'].round(6)
dfCP['Skew'] = dfB['Skew'].round(6)
dfB['Avg_Alpha'] = dfB['Avg_Alpha'].round(4)
dfCP['Avg_Alpha'] = dfB['Avg_Alpha'].round(4)
dfB['STD'] = dfB['STD'].round(4)
dfCP['STD'] = dfB['STD'].round(4)

skew_df_B = dfB.groupby('Skew').mean().sort_values('Skew', ascending=True)
skew_df_CP = dfCP.groupby('Skew').mean().sort_values('Skew', ascending=True)
STD_df_B = dfB.groupby('STD').mean().sort_values('STD', ascending=True)
STD_df_CP = dfCP.groupby('STD').mean().sort_values('STD', ascending=True)
Avg_Alpha_df_B = dfB.groupby('Avg_Alpha').mean().sort_values('Avg_Alpha', ascending=True)
Avg_Alpha_df_CP = dfCP.groupby('Avg_Alpha').mean().sort_values('Avg_Alpha', ascending=True)

skew_df_B.to_excel(f'C:/Users/tommo/Downloads/Thesis_MAB/Data/skew_dfB.xlsx', engine='openpyxl')
skew_df_CP.to_excel(f'C:/Users/tommo/Downloads/Thesis_MAB/Data/skew_dfCP.xlsx', engine='openpyxl')
STD_df_B.to_excel(f'C:/Users/tommo/Downloads/Thesis_MAB/Data/STD_dfB.xlsx', engine='openpyxl')
STD_df_CP.to_excel(f'C:/Users/tommo/Downloads/Thesis_MAB/Data/STD_dfCP.xlsx', engine='openpyxl')
Avg_Alpha_df_B.to_excel(f'C:/Users/tommo/Downloads/Thesis_MAB/Data/Avg_Alpha_dfB.xlsx', engine='openpyxl')
Avg_Alpha_df_CP.to_excel(f'C:/Users/tommo/Downloads/Thesis_MAB/Data/Avg_Alpha_dfCP.xlsx', engine='openpyxl')
