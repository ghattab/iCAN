import pandas as pd
import os
import numpy as np

list_of_datasets = [
    'ace_vaxinpad',
    'acp_anticp',
    'acp_iacp',
    'acp_mlacp',
    'afp_amppred',
    'afp_antifp',
    'aip_aippred',
    'aip_antiinflam',
    'amp_antibp',
    'amp_antibp2',
    'amp_csamp',
    'amp_fernandes',
    'amp_gonzales',
    'amp_iamp2l',
    'amp_modlamp',
    'atb_antitbp',
    'atb_iantitb',
    'avp_amppred',
    'avp_avppred',
    'bce_ibce',
    'cpp_cellppd',
    'cpp_cellppdmod',
    'cpp_cppredfl',
    'cpp_kelmcpp',
    'cpp_mixed',
    'cpp_mlcpp',
    'cpp_mlcppue',
    'cpp_sanders',
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
    'pip_pipel',
    'tce_zhao'
]

# add imbalances for new datasets
imbalance_dict = {
    'ace_vaxinpad':0.4404069767,
    'acp_anticp':0.5,
    'acp_iacp':0.4011627907,
    'acp_mlacp':0.3196581197,
    'afp_amppred':0.5,
    'afp_antifp':0.5003429355,
    'aip_aippred':0.4003813155,
    'aip_antiinflam':0.4063088512,
    'amp_antibp':0.5005807201,
    'amp_antibp2':0.5012543904,
    'amp_csamp':0.5,
    'amp_fernandes':0.4978354978,
    'amp_gonzales':0.2093023256,
    'amp_iamp2l':0.2676613886,
    'amp_modlamp':0.4749903063,
    'atb_antitbp':0.5,
    'atb_iantitb':0.5,
    'avp_amppred':0.5,
    'avp_avppred':0.5721107927,
    'bce_ibce':0.4408260524,
    'cpp_cellppd':0.5,
    'cpp_cellppdmod':0.5006839945,
    'cpp_cppredfl':0.5,
    'cpp_kelmcpp':0.5024925224,
    'cpp_mixed':0.7578125,
    'cpp_mlcpp':0.3878087231,
    'cpp_mlcppue':0.5,
    'cpp_sanders':0.7655172414,
    'hem_hemopi':0.472826087,
    'hiv_3tc': 0.3125,
    'hiv_abc': 0.2891760905,
    'hiv_apv': 0.603988604,
    'hiv_azt': 0.5185185185,
    'hiv_bevirimat': 0.2774193548,
    'hiv_d4t': 0.5410628019,
    'hiv_ddi': 0.4911717496,
    'hiv_dlv': 0.6337047354,
    'hiv_efv': 0.6199722607,
    'hiv_idv': 0.5065963061,
    'hiv_lpv': 0.4451097804,
    'hiv_nfv': 0.3909677419,
    'hiv_nvp': 0.5661664393,
    'hiv_protease': 0.1573389652,
    'hiv_rtv': 0.4793956044,
    'hiv_sqv': 0.6005256242,
    'hiv_v3': 0.14803849,
    'isp_il10pred': 0.3172302738,
    'nep_neuropipred': 0.5,
    'pip_pipel': 0.2580545229,
    'tce_zhao': 0.1773399015
}

# add bio_field for new datasets
bio_field_dict = {
    'ace_vaxinpad':'ace',
    'acp_anticp':'acp',
    'acp_iacp':'acp',
    'acp_mlacp':'acp',
    'afp_amppred':'afp',
    'afp_antifp':'afp',
    'aip_aippred':'aip',
    'aip_antiinflam':'aip',
    'amp_antibp':'amp',
    'amp_antibp2':'amp',
    'amp_csamp':'amp',
    'amp_fernandes':'amp',
    'amp_gonzales':'amp',
    'amp_iamp2l':'amp',
    'amp_modlamp':'amp',
    'atb_antitbp':'atb',
    'atb_iantitb':'atb',
    'avp_amppred':'avp',
    'avp_avppred':'avp',
    'bce_ibce':'bce',
    'cpp_cellppd':'cpp',
    'cpp_cellppdmod':'cpp',
    'cpp_cppredfl':'cpp',
    'cpp_kelmcpp':'cpp',
    'cpp_mixed':'cpp',
    'cpp_mlcpp':'cpp',
    'cpp_mlcppue':'cpp',
    'cpp_sanders':'cpp',
    'hem_hemopi':'hem',
    'hiv_3tc':'hiv',
    'hiv_abc':'hiv',
    'hiv_apv':'hiv',
    'hiv_azt':'hiv',
    'hiv_bevirimat':'hiv',
    'hiv_d4t':'hiv',
    'hiv_ddi':'hiv',
    'hiv_dlv':'hiv',
    'hiv_efv':'hiv',
    'hiv_idv':'hiv',
    'hiv_lpv':'hiv',
    'hiv_nfv':'hiv',
    'hiv_nvp':'hiv',
    'hiv_protease':'hiv',
    'hiv_rtv':'hiv',
    'hiv_sqv':'hiv',
    'hiv_v3':'hiv',
    'isp_il10pred':'isp',
    'nep_neuropipred':'nep',
    'pip_pipel':'pip',
    'tce_zhao':'tce'
}

def max_median(df):
    max_medians = []
    for i in range(len(df)):
        row = df.iloc[i,:]
        max_median = np.max([np.median([row[i*5:i*5+4]]) for i in range(10)])
        max_medians.append(max_median)
    return max_medians

def prepare_dataframe(file, encoding):
    path = os.path.join('.', 'csv', file)
    df = pd.read_csv(path, index_col=0)
    df['F1'] = max_median(df)
    df = pd.DataFrame(df['F1'])
    df['Dataset'] = df.index
    df['is_imbalanced'] = df['Dataset'].map(imbalance_dict)
    df['bio_field'] = df['Dataset'].map(bio_field_dict)
    df['Encoding'] = encoding
    df['type'] = "sequence based"
    df = df.reset_index(drop=True)
    return(df)

def best_encoding(df):
    df['Encoding_max'] = None
    for dataset in list_of_datasets:
        df_sub = df[df['Dataset'] == dataset]
        largest_value = df_sub['F1'].max()
        largest_encoding = df_sub.loc[df_sub['F1'] == largest_value, 'Encoding'].values[0]
        df.loc[df['Dataset'] == dataset, 'Encoding_max'] = largest_encoding
    return df


df_hyd = prepare_dataframe('f1_score_level_2_with_hydrogen.csv', 'cenact_hyd')
df_nohyd = prepare_dataframe('f1_score_level_2_without_hydrogen.csv', 'cenact_nohyd')
df_dd = prepare_dataframe('f1_score_level_2_data_driven.csv', 'cenact_dd')

df_comb = pd.concat([df_hyd, df_nohyd, df_dd]).reset_index(drop=True)
df_comb = best_encoding(df_comb)
df_comb = df_comb[['Dataset', 'Encoding', 'Encoding_max', 'F1', 'type', 'is_imbalanced', 'bio_field']]
df_comb.to_csv('./CENACT_f1_score_overview.csv', index=False)

json_data = df_comb.to_json(orient = 'records')

# Save the JSON string as a file
with open('../Data/Visualization_data/data/multiple_datasets/vis/mds_1_Overview/hm_cenact_data.json', 'w') as file:
    file.write(json_data)