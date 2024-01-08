import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import matplotlib.patches as mpatches

pvalues_df = pd.read_csv('p_value_alpha05_dis-shi_dd2.csv', index_col=0)
dataset_selection = [
    'ace_vaxinpad',
    'acp_anticp',
    'acp_iacp',
    'acp_mlacp',
    'afp_amppred',
    'afp_antifp',
    'ai4avp_2',
    'aip_aippred',
    'aip_antiinflam',
    'amp_antibp',
    'amp_antibp2',
    'amp_csamp',
    'amp_fernandes',
    'amp_gonzales',
    'amp_iamp2l',
    'amp_modlamp',
    'amy_albase',
    'amy_hex',
    'atb_antitbp',
    'atb_iantitb',
    'avp_amppred',
    'avp_avppred',
    'bce_ibce',
    'c2pred',
    'cpp_cellppd',
    'cpp_cellppdmod',
    'cpp_cppredfl',
    'cpp_kelmcpp',
    'cpp_mixed',
    'cpp_mlcpp',
    'cpp_mlcppue',
    'cpp_sanders',
    'cppsite2',
    'effectorp',
    'foldamer_b',
    'hem_hemopi',
    'hiv_3tc',
    'hiv_abc',
    'hiv_apv',
    'hiv_azt',
    'hiv_bevirimat',
    'hiv_d4t',
    'hiv_ddi',
    'hiv_dlv',
    'hiv_efv',
    'hiv_idv',
    'hiv_lpv',
    'hiv_nfv',
    'hiv_nvp',
    'hiv_protease',
    'hiv_rtv',
    'hiv_sqv',
    'hiv_v3',
    'isp_il10pred',
    'nep_neuropipred',
    'perm_cyc',
    'pip_pipel',
    'sol_ecoli',
    'tce_zhao',
    'toxinpred_swissprot',
    'toxinpred_trembl',
    'toxinpred2'
]
pvalues_df = pvalues_df.loc[dataset_selection]
pvalues = pvalues_df['p-value'].to_list()
count1 = sum([i < 0.05 for i in pvalues])
count2 = sum([i > 0.05 and i <= 0.95 for i in pvalues])
count3 = sum([i >= 0.95 for i in pvalues])
counts = [count1, count2, count3]
RKI_colors_reduced = ['#005EB8','#F5F5F5','#B87200'] # RKI colours with complement

# explosion
explode = (0.01, 0.01, 0.01, 0.01, 0.01)
 
font = {'size'   : 20}

matplotlib.rc('font', **font)

fig, ax = plt.subplots(figsize=(18,10))
# Pie Chart
plt.pie(counts, labels=counts, colors=RKI_colors_reduced, pctdistance=0, startangle = 360 * (count2 + count3) / (count1 + count2 + count3), textprops={'fontsize': 28})

# draw circle
centre_circle = plt.Circle((0, 0), 0.50, fc='white')
fig = plt.gcf()
 
# Adding Circle in Pie chart

fig.gca().add_artist(centre_circle)
 
# Adding Title of chart
plt.title('Significant differences in F1-scores between\nAlternative (CENACT) and Baseline (CMANGOES)', fontsize = 25, y=-0.01)
patch_1 = mpatches.Patch(color=RKI_colors_reduced[0], label='Alternative (CENACT) significantly\nbetter (95% confidence)')
patch_2 = mpatches.Patch(color=RKI_colors_reduced[1], label='No significant difference')
patch_3 = mpatches.Patch(color=RKI_colors_reduced[2], label='Baseline (CMANGOES) significantly\nbetter (95% confidence)')
plt.legend(handles=[patch_1, patch_2, patch_3], fontsize = 'medium', loc = 'upper left', bbox_to_anchor = (-0.65,1.13))
 
# Displaying Chart
plt.savefig('donut.png')