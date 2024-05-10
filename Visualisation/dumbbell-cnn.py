import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
plt.rcParams.update({'font.size': 12})

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


# take median over 5 splits over all 10 different cross-validations instead of median over 50 f1-scores
def max_median(df, dataset, repeats=10):
    row = df.loc[dataset,:]
    max_median = np.max([np.median([row[(j*5):(j*5+4)]]) for j in range(repeats)])
    return max_median

RKI_colors = ['#005EB8','#71BAFE', '#F5F5F5', '#FECA76', '#B87200']
rki_blue = RKI_colors[0]
rki_grey = RKI_colors[1]
rki_orange = RKI_colors[-1]

file_iCAN_cnn = 'f1_score_level_2_with_hydrogen.csv'
results_path_iCAN_cnn = os.path.join('..', 'Results', 'CNN', file_iCAN_cnn)

iCAN_cnn_f1 = pd.read_csv(results_path_iCAN_cnn, index_col=0)
iCAN_cnn_f1 = iCAN_cnn_f1.loc[dataset_selection]

file_iCAN = 'f1_score_level_2_with_hydrogen.csv'
results_path_iCAN = os.path.join('..', 'Results', 'csv', file_iCAN)

iCAN_f1 = pd.read_csv(results_path_iCAN, index_col=0)
iCAN_f1 = iCAN_f1.loc[dataset_selection]

list_of_datasets = iCAN_f1.index

fig, ax = plt.subplots(figsize=(18, 18))

df = pd.DataFrame(index=list_of_datasets, columns=['Dataset','iCAN','iCAN_cnn'])

for dataset_idx in range(len(list_of_datasets)):
    dataset = list_of_datasets[dataset_idx]
    iCAN_cnn_f1_score = max_median(iCAN_cnn_f1, dataset, 1)
    iCAN_f1_score = max_median(iCAN_f1, dataset)
    df.loc[dataset, 'Dataset'] = dataset
    df.loc[dataset, 'iCAN_cnn'] = iCAN_cnn_f1_score
    df.loc[dataset, 'iCAN'] = iCAN_f1_score

df = df.sort_values(by='iCAN', ascending=True)
df = df.reset_index(drop=True)

for i in df.index:
    ax.plot([df.loc[i,'iCAN_cnn'], df.loc[i,'iCAN']], [i,i], color = 'black', linewidth = 2)
    ax.plot([df.loc[i,'iCAN_cnn'], df.loc[i,'iCAN_cnn']], [i,i], marker ='o', color=rki_blue, markersize=10)
    ax.plot([df.loc[i,'iCAN'], df.loc[i,'iCAN']], [i,i], marker ='o', color=rki_grey, markersize=10)
    
#plt.title('Comparison of f1-scores of CMANGOES encodings and iCAN encoding\n10 medians taken over CV-split with k=5, max median reported', fontsize = 20)
ax.set_yticks(range(0, len(list_of_datasets)))
ax.set_yticklabels(df['Dataset'])
plt.xlabel('f1-scores')
plt.ylabel('Datasets')
plt.xlim(0,1)
plt.ylim(-1,len(dataset_selection))
plt.grid(axis = 'x')

iCAN_cnn_patch = mpatches.Patch(color=rki_blue, label='iCAN with Convolutional Neural Network')
iCAN_patch = mpatches.Patch(color=rki_grey, label='iCAN with Random Forest Classifier')
plt.legend(handles=[iCAN_patch, iCAN_cnn_patch], fontsize = 'xx-large')

plt.savefig('./dumbbell_cnn.png', dpi=300)