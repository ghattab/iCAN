import scipy.stats as stats
import pandas as pd
import os
import numpy as np

os.chdir('..')

def prep(file,  dir = 'Results/csv'):
    results_path = os.path.join('.', dir, file)
    df = pd.read_csv(results_path, index_col=0)
    return df

hyd2_df = prep('f1_score_level_2_with_hydrogen.csv')
nohyd2_df = prep('f1_score_level_2_without_hydrogen.csv')
dd2_df = prep('f1_score_level_2_data_driven.csv')

bin_shi = prep('f1_score_binary_shifted_levels_1_and_2.csv', 'Results/CMANGOES')
bin_cen = prep('f1_score_binary_centered_levels_1_and_2.csv', 'Results/CMANGOES')
dis_shi = prep('f1_score_discretized_shifted_levels_1_and_2.csv', 'Results/CMANGOES')
dis_cen = prep('f1_score_discretized_centered_levels_1_and_2.csv', 'Results/CMANGOES')

alpha = 0.05

def median(df, dataset):
    row = df.loc[dataset,:]
    medians = [np.median([row[(j*5):(j*5+4)]]) for j in range(10)]
    return medians

df1 = dd2_df
df2 = dis_shi

index = df1.index
p_list = []
p_alpha_list = []

for row_idx in range(len(df1)):
    dataset = index[row_idx]
    sample1 = median(df1, dataset)
    sample2 = median(df2,dataset)
    statistic, p_value = stats.mannwhitneyu(sample1, sample2, alternative='greater')
    p_list.append(p_value)
    p_alpha_list.append(p_value < alpha)


p_value_alpha_df = pd.DataFrame({'columns': index, 'p-value': p_list, 'CENACT significantly better than CMANGOES?': p_alpha_list})

p_value_alpha_df.to_csv('./Visualisation/p_value_alpha05_dis-shi_dd2.csv', index=0, header=True)