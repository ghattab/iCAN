from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import math

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

n_runs = 10

imp_path = os.path.join('.', 'Importance_heatmaps')
if os.path.exists(imp_path) == False:
    os.mkdir(imp_path)

for data_idx in range(len(dataset_selection)):
    dataset = dataset_selection[data_idx]
    print("Running dataset", data_idx + 1, "/", len(dataset_selection))

    enc_path = os.path.join('..', 'Data', 'Encodings', dataset, 'CENACT_level_2_with_hydrogen.csv')
    classes_path = os.path.join('..', 'Data', 'Original_datasets', dataset, 'classes.txt')
    save_path = os.path.join('.', 'Importance_heatmaps', dataset + '.jpg')
    save_path_zoom = os.path.join('.', 'Importance_heatmaps', dataset + '_zoom.jpg')

    X = pd.read_csv(enc_path)
    y = pd.read_csv(classes_path, header=None)
    y = y.astype("category")
    y = y.to_numpy().ravel()

    cum_imp = np.zeros(X.shape[1])
    dim_map = (10, X.shape[1] // 10)

    for n in range(n_runs):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=n, shuffle=True, stratify=None)
        vanilla_rfc = RandomForestClassifier(n_jobs = -1)
        vanilla_rfc.fit(X_train, y_train)
        ft_imp = vanilla_rfc.feature_importances_

        cum_imp = cum_imp + ft_imp

    imp_map = np.reshape(cum_imp, dim_map, "F")

    plt.imshow(imp_map, cmap='gray_r')
    plt.xlim(-0.5,20.5)
    plt.xlabel('Carbon chain')
    plt.ylabel('Neighbourhood-levels and atom types\n \n')
    custom_xticks = [r'$C_{' + str(i) + '}$' for i in range(0, 21, 5)]
    plt.xticks(range(0, 21, 5), custom_xticks)
    plt.axhline(y=4.5, color='gray', linestyle='--', xmin=-.05, clip_on = False)
    custom_yticks =  ['H', 'C', 'N', 'O', 'S', 'H', 'C', 'N', 'O', 'S']
    plt.yticks(range(0, 10, 1), custom_yticks)
    plt.text(-2.5, 2, "Level 1", fontsize=11, verticalalignment='center', rotation='vertical')
    plt.text(-2.5, 7, "Level 2", fontsize=11, verticalalignment='center', rotation='vertical')

    map = plt.gcf()
    map.set_size_inches(6.4, 4.8)
    map.savefig(save_path_zoom, dpi=300)

    plt.clf()

    plt.imshow(imp_map, cmap='gray_r')
    plt.xlabel('Carbon chain')
    #plt.ylabel('Neighbourhood-levels and atom types\n \n')
    locs, labels = plt.xticks()
    locs = [i for i in locs if i > -1]
    custom_xticks = [r'$C_{' + str(int(i)) + '}$' for i in locs]
    plt.xticks(locs, custom_xticks)
    plt.axhline(y=4.5, color='gray', linestyle='--', linewidth=1, xmin=0, clip_on = False)
    plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    #custom_yticks =  ['H', 'C', 'N', 'O', 'S', 'H', 'C', 'N', 'O', 'S']
    #plt.yticks(range(0, 10, 1), custom_yticks)
    #plt.text(-2.5, 2, "Level 1", fontsize=11, verticalalignment='center', rotation='vertical')
    #plt.text(-2.5, 7, "Level 2", fontsize=11, verticalalignment='center', rotation='vertical')
    map = plt.gcf()
    map.set_size_inches(np.min([math.sqrt(locs[-1]),2**13]), 2)
    map.savefig(save_path, dpi=800)

    plt.clf()