import pandas as pd
from scipy.stats import ttest_1samp, wilcoxon


def statistical_analysis(df_merged):
    # Calculate the differences between method a and method b for each metric
    df_merged['regret_diff'] = df_merged['Acc_Reg(Beliefs)'] - df_merged['Acc_Reg(CP)']
    df_merged['reward_diff'] = df_merged['Acc_Reward(Beliefs)'] - df_merged['Acc_Reward(CP)']
    df_merged['rate_diff'] = df_merged['ROBR(Beliefs)'] - df_merged['ROBR(CP)']

    print(df_merged.to_string())

    # Perform one-sample t-test
    t_stat_reg, p_value_t_reg = ttest_1samp(df_merged['regret_diff'], 0)
    t_stat_rew, p_value_t_rew = ttest_1samp(df_merged['reward_diff'], 0)
    t_stat_robr, p_value_t_robr = ttest_1samp(df_merged['rate_diff'], 0)

    # Perform Wilcoxon signed-rank test
    stat_reg, p_value_w_reg = wilcoxon(df_merged['regret_diff'])
    stat_rew, p_value_w_rew = wilcoxon(df_merged['reward_diff'])
    stat_rate, p_value_w_robr = wilcoxon(df_merged['rate_diff'])

    # Print the p-values
    print(f"Regret: One-sample t-test stat, p-value:, {t_stat_reg}, {p_value_t_reg}")
    print(f"        Wilcoxon signed-rank test stat, p-value: {stat_reg}, {p_value_w_reg}")
    print(f"Reward: One-sample t-test stat, p-value:, {t_stat_rew}, {p_value_t_rew}")
    print(f"        Wilcoxon signed-rank test stat, p-value: {stat_rew}, {p_value_w_rew}")
    print(f"ROBR: One-sample t-test stat, p-value:: {t_stat_robr}, {p_value_t_robr}")
    print(f"        Wilcoxon signed-rank test stat, p-value: {stat_rate}, {p_value_w_robr}")
