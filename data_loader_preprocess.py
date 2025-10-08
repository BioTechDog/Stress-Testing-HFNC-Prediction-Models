import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

data_path = 'data/'

'''-------------HFNO_combined dataset-------------'''

def load_DNI():
    HFNO = pd.read_excel(data_path + 'HFNO.xlsx')

    HFNO_T0 = HFNO[['Id N', 'Age (y)', 'Sex 0=F, 1=M','BMI (km/m2)', 'RICU admission diagnosis 0=pneumonia, 1=ARDS',
                    'Solid tumor 0=no, 1=yes', 'Lymphoma/leukemia 0=no, 1=yes', 'Dementia 0=no, 1=yes', 'Metastatic 0=no, 1=yes',
                    'CHF 0=no, 1=yes', 'COPD 0=no, 1=yes','Charlson index', 'SOFA', 'Creatinine (mg/dl)', 'BORG_T0',
                    'HACOR_T0', 'ROX_T0', 'Δpes_T0 (cmH2O)', 'HR_T0 (bpm)', 'Sistolic_blood_pressure_T0 (mmHg)', 'RR_T0 (bpm)',
                    'FiO2_T0 (%)', 'PaO2_T0  (mmHg)', 'P/F_T0 (mmHg)', 'SpO2_T0 (%)', 'pH_A_T0', 'pCO2_A_T0 (mmHg)',
                    'BE_A_T0 (mmol/L)', 'HCO3-_T0 (mmol/L)', 'Lactate_T0 (mmol/L)', 'HFNO_failure', 'Death']].copy()
    HFNO_T0.drop(['Id N','BORG_T0'], axis=1, inplace=True)
    HFNO_T0.rename(columns = {'Charlson index':'Charlson_index', 'BMI (km/m2)':'BMI (kg/m2)', 'RICU admission diagnosis 0=pneumonia, 1=ARDS':'Diagnosis',
                    'pH_A_T0':'pH_T0', 'pCO2_A_T0 (mmHg)': 'pCO2_T0 (mmHg)', 'BE_A_T0 (mmol/L)': 'BE_T0 (mmol/L)', 'PaO2_T0  (mmHg)':'PaO2_T0 (mmHg)'},inplace=True)

    HFNO_T1 = HFNO[['Id N', 'Age (y)', 'Sex 0=F, 1=M', 'BMI (km/m2)',
           'RICU admission diagnosis 0=pneumonia, 1=ARDS','Charlson index', 'SOFA', 'Creatinine (mg/dl)',
           'BORG_T1', 'HACOR_T1', 'ROX_T1',
           'Δpes_T1 (cmH2O)', 'HR_T1(bpm)', 'Sistolic_blood_pressure_T1 (mmHg)',
           'RR_T1 (bpm)', 'FiO2_T1 (%)', 'PaO2_T1  (mmHg)', 'P/F_T1 (mmHg)',
           'SpO2_T1 (%)', 'pH_A_T1', 'pCO2_A_T1 (mmHg)', 'BE_A_T1 (mmol/L)',
           'HCO3-_T1 (mmol/L)', 'Lactate (mmol/L)', 'HFNO_failure',
           'NIV_upgrade_theoretically', 'NIV_upgrade_actually', 'Death']].copy()

    HFNO_T1.drop(['Id N'],axis=1, inplace=True)
    HFNO_T1.rename(
        columns={'RICU admission diagnosis 0=pneumonia, 1=ARDS': 'Diagnosis',
                 'HR_T1(bpm)': 'HR_T1 (bpm)', 'Δpes_T1 (cmH2O)': 'Δpes_T1 (cmH2O)',
                 'RR_T1 (bpm)': 'RR_T1 (bpm)', 'FiO2_T1 (%)': 'FiO2_T1 (%)', 'PaO2_T1  (mmHg)': 'PaO2_T1 (mmHg)',
                 'P/F_T1 (mmHg)': 'P/F_T1 (mmHg)', 'pH_A_T1': 'pH_T1', 'pCO2_A_T1 (mmHg)': 'pCO2_T1 (mmHg)'},
        inplace=True,
    )


    unique_columns_df1 = HFNO_T1.columns.difference(HFNO_T0.columns)
    df1_unique = HFNO_T1[unique_columns_df1]
    HFNO_DNI_t0_reset = HFNO_T0.reset_index(drop=True)
    df1_unique_reset = df1_unique.reset_index(drop=True)
    HFNO_combined = pd.concat([HFNO_DNI_t0_reset, df1_unique_reset], axis=1)
    HFNO_combined['HACOR_diff'] = HFNO_combined['HACOR_T1'] - HFNO_combined['HACOR_T0']
    HFNO_combined['ROX_diff'] = HFNO_combined['ROX_T1'] - HFNO_combined['ROX_T0']
    HFNO_combined['HR_diff'] = HFNO_combined['HR_T1 (bpm)'] - HFNO_combined['HR_T0 (bpm)']
    HFNO_combined['SpO2_diff'] = HFNO_combined['SpO2_T1 (%)'] - HFNO_combined['SpO2_T0 (%)']
    HFNO_combined['FiO2_diff'] = HFNO_combined['FiO2_T1 (%)'] - HFNO_combined['FiO2_T0 (%)']
    HFNO_combined['PaO2_diff'] = HFNO_combined['PaO2_T1 (mmHg)'] - HFNO_combined['PaO2_T0 (mmHg)']
    HFNO_combined['pCO2_diff'] = HFNO_combined['pCO2_T1 (mmHg)'] - HFNO_combined['pCO2_T0 (mmHg)']
    HFNO_combined['P/F_diff'] = HFNO_combined['P/F_T1 (mmHg)'] - HFNO_combined['P/F_T0 (mmHg)']
    HFNO_combined['RR_diff'] = HFNO_combined['RR_T1 (bpm)'] - HFNO_combined['RR_T0 (bpm)']
    HFNO_combined['1/RR_t0'] = 1/HFNO_combined['RR_T0 (bpm)']
    HFNO_combined['1/RR_t1'] = 1/HFNO_combined['RR_T1 (bpm)']
    return HFNO_combined

'''---------------------------nonDNI----------------------------'''
def load_nonDNI():
    HFNO_nonDNI = pd.read_excel(data_path + 'AJRCCM2023_dataset.xlsx')
    HFNO_nonDNI_t0 = HFNO_nonDNI[(HFNO_nonDNI['Time_point'] == 0)]
    HFNO_nonDNI_t0.rename(
        columns={'HFNC_failure_to_NIV': 'HFNO_failure','Age': 'Age (y)','SOFA_score':'SOFA','HACOR_score':'HACOR_T0',
                 'ROX_index':'ROX_T0','HR':'HR_T0 (bpm)','SpO2':'SpO2_T0 (%)','FiO2':'FiO2_T0 (%)','PO2':'PaO2_T0 (mmHg)',
                'PCO2':'pCO2_T0 (mmHg)','P/F':'P/F_T0 (mmHg)','RR':'RR_T0 (bpm)'},
        inplace=True,
    )
    HFNO_nonDNI_t0.loc[:, 'FiO2_T0 (%)'] = HFNO_nonDNI_t0['FiO2_T0 (%)'] * 100

    HFNO_nonDNI_t1 = HFNO_nonDNI[(HFNO_nonDNI['Time_point'] == 1)]
    HFNO_nonDNI_t1.rename(
        columns={'HFNC_failure_to_NIV': 'HFNO_failure','Age': 'Age (y)','SOFA_score':'SOFA','HACOR_score':'HACOR_T1',
                 'ROX_index':'ROX_T1','HR':'HR_T1 (bpm)','SpO2':'SpO2_T1 (%)','FiO2':'FiO2_T1 (%)','PO2':'PaO2_T1 (mmHg)',
                'PCO2':'pCO2_T1 (mmHg)','P/F':'P/F_T1 (mmHg)','RR':'RR_T1 (bpm)'},
        inplace=True,
    )
    HFNO_nonDNI_t1.loc[:, 'FiO2_T1 (%)'] = HFNO_nonDNI_t1['FiO2_T1 (%)'] * 100

    common_columns = HFNO_nonDNI_t0.columns.intersection(HFNO_nonDNI_t1.columns)
    df0_filtered = HFNO_nonDNI_t0[common_columns]
    unique_columns_df1 = HFNO_nonDNI_t1.columns.difference(HFNO_nonDNI_t0.columns)
    df1_unique = HFNO_nonDNI_t1[unique_columns_df1]
    HFNO_nonDNI_t0_reset = HFNO_nonDNI_t0.reset_index(drop=True)
    df1_unique_reset = df1_unique.reset_index(drop=True)
    HFNO_combined = pd.concat([HFNO_nonDNI_t0_reset, df1_unique_reset], axis=1)

    HFNO_combined['HACOR_diff'] = HFNO_combined['HACOR_T1'] - HFNO_combined['HACOR_T0']
    HFNO_combined['ROX_diff'] = HFNO_combined['ROX_T1'] - HFNO_combined['ROX_T0']
    HFNO_combined['HR_diff'] = HFNO_combined['HR_T1 (bpm)'] - HFNO_combined['HR_T0 (bpm)']
    HFNO_combined['SpO2_diff'] = HFNO_combined['SpO2_T1 (%)'] - HFNO_combined['SpO2_T0 (%)']
    HFNO_combined['FiO2_diff'] = HFNO_combined['FiO2_T1 (%)'] - HFNO_combined['FiO2_T0 (%)']
    HFNO_combined['PaO2_diff'] = HFNO_combined['PaO2_T1 (mmHg)'] - HFNO_combined['PaO2_T0 (mmHg)']
    HFNO_combined['pCO2_diff'] = HFNO_combined['pCO2_T1 (mmHg)'] - HFNO_combined['pCO2_T0 (mmHg)']
    HFNO_combined['P/F_diff'] = HFNO_combined['P/F_T1 (mmHg)'] - HFNO_combined['P/F_T0 (mmHg)']
    HFNO_combined['RR_diff'] = HFNO_combined['RR_T1 (bpm)'] - HFNO_combined['RR_T0 (bpm)']
    HFNO_combined['1/RR_t0'] = 1/HFNO_combined['RR_T0 (bpm)']
    HFNO_combined['1/RR_t1'] = 1/HFNO_combined['RR_T1 (bpm)']

    return HFNO_combined

HFNO_DNI = load_DNI()
HFNO_nonDNI = load_nonDNI()

common_columns = HFNO_DNI.columns.intersection(HFNO_nonDNI.columns)
HFNO_DNI_common = HFNO_DNI[common_columns]
HFNO_nonDNI_common = HFNO_nonDNI[common_columns]
HFNO_combined = pd.concat([HFNO_DNI_common, HFNO_nonDNI_common], axis=0, ignore_index=True)

HFNO_combined['mROX_T0'] = HFNO_combined['PaO2_T0 (mmHg)']/(HFNO_combined['FiO2_T0 (%)']/100*HFNO_combined['RR_T0 (bpm)'])
HFNO_combined['mROX_T1'] = HFNO_combined['PaO2_T1 (mmHg)']/(HFNO_combined['FiO2_T1 (%)']/100*HFNO_combined['RR_T1 (bpm)'])
HFNO_combined['ROX_HR_T0'] = HFNO_combined['SpO2_T0 (%)']/(HFNO_combined['FiO2_T0 (%)']/100 * HFNO_combined['RR_T0 (bpm)']*HFNO_combined['HR_T0 (bpm)'])*100
HFNO_combined['ROX_HR_T1'] = HFNO_combined['SpO2_T1 (%)']/(HFNO_combined['FiO2_T1 (%)']/100 * HFNO_combined['RR_T1 (bpm)']*HFNO_combined['HR_T1 (bpm)'])*100
HFNO_combined['mROX_HR_T0'] = HFNO_combined['PaO2_T0 (mmHg)']/(HFNO_combined['FiO2_T0 (%)']/100 * HFNO_combined['RR_T0 (bpm)']*HFNO_combined['HR_T0 (bpm)'])*100
HFNO_combined['mROX_HR_T1'] = HFNO_combined['PaO2_T1 (mmHg)']/(HFNO_combined['FiO2_T1 (%)']/100 * HFNO_combined['RR_T1 (bpm)']*HFNO_combined['HR_T1 (bpm)'])*100
HFNO_combined['S/F_T0'] = HFNO_combined['SpO2_T0 (%)'] / HFNO_combined['FiO2_T0 (%)'] * 100
HFNO_combined['S/F_T1'] = HFNO_combined['SpO2_T1 (%)'] / HFNO_combined['FiO2_T1 (%)'] * 100
HFNO_combined['mROX_diff'] =  HFNO_combined['mROX_T1'] - HFNO_combined['mROX_T0'] 
HFNO_combined['ROX_HR_diff'] =  HFNO_combined['ROX_HR_T1'] - HFNO_combined['ROX_HR_T0'] 
HFNO_combined['mROX_HR_diff'] =  HFNO_combined['mROX_HR_T1'] - HFNO_combined['mROX_HR_T0'] 
HFNO_combined['S/F_diff'] =  HFNO_combined['S/F_T1'] - HFNO_combined['S/F_T0'] 
HFNO_combined['ROX_diff'] = HFNO_combined['ROX_T1'] - HFNO_combined['ROX_T0']
HFNO_combined['1/RR_t0'] = 1/HFNO_combined['RR_T0 (bpm)']
HFNO_combined['1/RR_t1'] = 1/HFNO_combined['RR_T1 (bpm)']
HFNO_combined['1/FiO2_t0'] = 1/HFNO_combined['FiO2_T0 (%)']
HFNO_combined['1/FiO2_t1'] = 1/HFNO_combined['FiO2_T1 (%)']

'''-------------MIMIC_HFNC dataset-------------'''
def MIMIC_HFNC_t0_t1_supp():
    try:
        MIMIC_HFNC_t0 = pd.read_csv(data_path + 'HFNC_t0.csv')
        MIMIC_HFNC_t1 = pd.read_csv(data_path + 'HFNC_t1.csv')
    except FileNotFoundError as e:
        print(f"MIMIC data files not found: {e}")
        # Return empty DataFrame with expected structure
        return pd.DataFrame()
    
    # Check if we have valid data
    if MIMIC_HFNC_t0.empty or MIMIC_HFNC_t1.empty:
        print("Empty MIMIC datasets, returning empty DataFrame")
        return pd.DataFrame()
        
    MIMIC_HFNC_t0 = MIMIC_HFNC_t0.rename(columns=lambda x: f"{x}_t0" if x != 'stay_id' else x)
    # Merge the two DataFrames on the stay_id column
    MIMIC_HFNC = pd.merge(MIMIC_HFNC_t0, MIMIC_HFNC_t1, on='stay_id')
    MIMIC_HFNC.rename(columns={'po2_t0': 'PaO2_T0 (mmHg)', 'fio2_t0': 'FiO2_T0 (%)', 'pco2_t0': 'pCO2_T0 (mmHg)',
                              'pao2fio2ratio_t0': 'P/F_T0 (mmHg)', 'spo2_t0':'SpO2_T0 (%)',
                              'ph_t0': 'pH_T0', 'age': 'Age (y)', 'resp_rate_t0': 'RR_T0 (bpm)','heart_rate_t0': 'HR_T0 (bpm)',
                              'po2': 'PaO2_T1 (mmHg)', 'pco2': 'pCO2_T1 (mmHg)','spo2':'SpO2_T1 (%)',
                              'fio2': 'FiO2_T1 (%)', 'ph': 'pH_T1', 'resp_rate': 'RR_T1 (bpm)', 'heart_rate': 'HR_T1 (bpm)',
                              'pao2fio2ratio': 'P/F_T1 (mmHg)'}, inplace=True)#'sapsii_t0':'SAPSII'
    MIMIC_HFNC['HFNO_failure'] = np.where(
        (MIMIC_HFNC['intubation_status'] == 'intubated') | (MIMIC_HFNC['survive_days_from_icu'] <= 4), 1, 0)
    MIMIC_HFNC['FiO2_T0 (%)'] = MIMIC_HFNC['FiO2_T0 (%)'].fillna(MIMIC_HFNC['fio2_chartevents_t0'])
    MIMIC_HFNC['FiO2_T0 (%)'] = MIMIC_HFNC['FiO2_T0 (%)'].replace(0, np.nan)  # Replace 0 with NaN to handle it in the next step
    MIMIC_HFNC['FiO2_T0 (%)'] = MIMIC_HFNC['FiO2_T0 (%)'].fillna(21)
    MIMIC_HFNC['FiO2_T1 (%)'] = MIMIC_HFNC['FiO2_T1 (%)'].fillna(MIMIC_HFNC['fio2_chartevents'])
    
    # print(MIMIC_HFNC.isnull().sum())
    # print(len(MIMIC_HFNC))
    
    #
    
    #MIMIC_NIV[['FiO2', 'fio2_chartevents_t0', 'post 1-2h FiO2', 'ΔT0-T1 P/F (mmHg)']].isnull().sum()
    columns_to_impute = ['Age (y)', 'PaO2_T0 (mmHg)','SpO2_T0 (%)', 'pCO2_T0 (mmHg)', 'HFNO_failure', 'FiO2_T1 (%)',
                         'PaO2_T1 (mmHg)', 'SpO2_T1 (%)', 'pCO2_T1 (mmHg)','HR_T0 (bpm)', 'HR_T1 (bpm)']#,'SAPSII','RR_T1 (bpm)' 'P/F_T0 (mmHg)','FiO2_T0 (%)',
    imputer = KNNImputer(n_neighbors=3)
    MIMIC_HFNC[columns_to_impute] = imputer.fit_transform(MIMIC_HFNC[columns_to_impute])
    # use fio2 to calculate P/F
    MIMIC_HFNC['P/F_T0 (mmHg)'] = MIMIC_HFNC['PaO2_T0 (mmHg)'] / MIMIC_HFNC['FiO2_T0 (%)'] * 100
    filter_MIMIC_HFNC = MIMIC_HFNC.dropna(subset=['RR_T0 (bpm)','P/F_T0 (mmHg)', 'P/F_T1 (mmHg)','RR_T1 (bpm)']) #'RR_T0 (bpm)',
    #filter_MIMIC_HFNC = MIMIC_HFNC
    
    #imputr
#     imputer = KNNImputer(n_neighbors=3)
#     filter_MIMIC_HFNC[['RR_T0 (bpm)']] = imputer.fit_transform(filter_MIMIC_HFNC[['RR_T0 (bpm)']])
    
    filter_MIMIC_HFNC['RR_diff'] = filter_MIMIC_HFNC['RR_T1 (bpm)'] - filter_MIMIC_HFNC['RR_T0 (bpm)']
    filter_MIMIC_HFNC['PaO2_diff'] = filter_MIMIC_HFNC['PaO2_T1 (mmHg)'] - filter_MIMIC_HFNC['PaO2_T0 (mmHg)']
    filter_MIMIC_HFNC['FiO2_diff'] = filter_MIMIC_HFNC['FiO2_T1 (%)'] - filter_MIMIC_HFNC['FiO2_T0 (%)']
    filter_MIMIC_HFNC['P/F_diff'] = filter_MIMIC_HFNC['P/F_T1 (mmHg)'] - filter_MIMIC_HFNC['P/F_T0 (mmHg)']
    filter_MIMIC_HFNC['pCO2_diff'] = filter_MIMIC_HFNC['pCO2_T1 (mmHg)'] - filter_MIMIC_HFNC['pCO2_T0 (mmHg)']
    filter_MIMIC_HFNC['SpO2_diff'] = filter_MIMIC_HFNC['SpO2_T1 (%)'] - filter_MIMIC_HFNC['SpO2_T0 (%)']
    filter_MIMIC_HFNC['HR_diff'] = filter_MIMIC_HFNC['HR_T1 (bpm)'] - filter_MIMIC_HFNC['HR_T0 (bpm)']
    
    filter_MIMIC_HFNC['ROX_T0'] = filter_MIMIC_HFNC['SpO2_T0 (%)'] / (filter_MIMIC_HFNC['FiO2_T0 (%)']/100) / filter_MIMIC_HFNC['RR_T0 (bpm)']
    filter_MIMIC_HFNC['ROX_T1'] = filter_MIMIC_HFNC['SpO2_T1 (%)'] / (filter_MIMIC_HFNC['FiO2_T1 (%)']/100) / filter_MIMIC_HFNC['RR_T1 (bpm)']
    filter_MIMIC_HFNC['ROX_diff'] = filter_MIMIC_HFNC['ROX_T1'] - filter_MIMIC_HFNC['ROX_T0']
    
    filter_MIMIC_HFNC = filter_MIMIC_HFNC[filter_MIMIC_HFNC['P/F_T0 (mmHg)'] < 300]
    # filter_MIMIC_NIV = MIMIC_NIV.dropna(subset=helmet_columns_recall)
#     filter_MIMIC_HFNC['ΔT0-T1 RR (bpm)'] = filter_MIMIC_NIV['post 1-2h RR (bpm)'] - filter_MIMIC_NIV['RR (bpm)']
    return filter_MIMIC_HFNC

filter_MIMIC_HFNC_roomair = MIMIC_HFNC_t0_t1_supp()
if not filter_MIMIC_HFNC_roomair.empty:
    print(f"MIMIC HFNC failure distribution: {filter_MIMIC_HFNC_roomair['HFNO_failure'].value_counts()}")
else:
    print("No MIMIC data available")

'''-------------eicu_HFNC dataset-------------'''
def eicu_HFNC_t0_t1():
    HFNC_eicu = pd.read_csv(data_path + 'HFNC_eicu.csv')
    # Calculate the time differences
    HFNC_eicu['labtime_t0_diff'] = abs(HFNC_eicu['labtime_t0'] - HFNC_eicu['vent_start'])
    HFNC_eicu['nursingcharttime_t0_diff'] = abs(HFNC_eicu['nursingcharttime_t0'] - HFNC_eicu['vent_start'])
    HFNC_eicu['labtime_t1_diff'] = abs(HFNC_eicu['labtime_t1'] - (HFNC_eicu['vent_start'] + 120))
    HFNC_eicu['nursingcharttime_t1_diff'] = abs(HFNC_eicu['nursingcharttime_t1'] - (HFNC_eicu['vent_start'] + 120))

    # Sort the DataFrame by icustay_id and time differences
    HFNC_eicu_sorted = HFNC_eicu.sort_values(by=['icustay_id', 'labtime_t0_diff', 'nursingcharttime_t0_diff', 'labtime_t1_diff', 'nursingcharttime_t1_diff'])

    # Group by icustay_id and take the first row for each group (which will have the smallest differences)
    result_HFNC_eicu = HFNC_eicu_sorted.groupby('icustay_id').first().reset_index()

    # Drop the difference columns if you no longer need them
    result_HFNC_eicu = result_HFNC_eicu.drop(columns=['labtime_t0_diff', 'nursingcharttime_t0_diff', 'labtime_t1_diff', 'nursingcharttime_t1_diff'])

    result_HFNC_eicu.rename(columns={'pao2_t0': 'PaO2_T0 (mmHg)', 'fio2_t0': 'FiO2_T0 (%)', 'paco2_t0': 'pCO2_T0 (mmHg)',
                              'o2sat_t0':'SpO2_T0 (%)','ph_t0': 'pH_T0', 'rr_t0': 'RR_T0 (bpm)','hr_t0': 'HR_T0 (bpm)',
                              'pao2_t1': 'PaO2_T1 (mmHg)', 'paco2_t1': 'pCO2_T1 (mmHg)','o2sat_t1':'SpO2_T1 (%)',
                              'fio2_t1': 'FiO2_T1 (%)', 'ph_t1': 'pH_T1', 'rr_t1': 'RR_T1 (bpm)', 'hr_t1': 'HR_T1 (bpm)'}, inplace=True)
    
    result_HFNC_eicu['P/F_T0 (mmHg)'] = result_HFNC_eicu['PaO2_T0 (mmHg)']/result_HFNC_eicu['FiO2_T0 (%)']*100
    result_HFNC_eicu['P/F_T1 (mmHg)'] = result_HFNC_eicu['PaO2_T1 (mmHg)']/result_HFNC_eicu['FiO2_T1 (%)']*100
    
    HFNC_eicu_supp = pd.read_csv(data_path + 'HFNC_eicu_supp.csv')
    HFNC_eicu_supp.rename(columns={'age':'Age (y)', 'patientunitstayid':'icustay_id'}, inplace=True)
    
    merged_df = pd.merge(result_HFNC_eicu, HFNC_eicu_supp, on='icustay_id', how='inner')
    
    # Condition 1: If second_oxygen_therapy_type or last_oxygen_therapy_type is 2, 3, or 4
    condition_1 = merged_df['second_oxygen_therapy_type'].isin([2, 3, 4]) | merged_df['last_oxygen_therapy_type'].isin([2, 3, 4])
    #condition_1 = merged_df['second_oxygen_therapy_type'].isin([2, 3, 4])

    # Condition 2: If actualicumortality is 'EXPIRED' and (hospitaldischargeoffset - vent_start) / 60 <= 48
    condition_2 = (merged_df['actualicumortality'] == 'EXPIRED') & ((merged_df['hospitaldischargeoffset'] - merged_df['vent_start']) / 60 <= 128)

    # Combine conditions
    merged_df['HFNO_failure'] = (condition_1 | condition_2).astype(int)
    
    #print(merged_df.isnull().sum())
    
    columns_to_impute = ['SpO2_T0 (%)', 'SpO2_T1 (%)', 'HR_T0 (bpm)', 'HR_T1 (bpm)']
    imputer = KNNImputer(n_neighbors=3)
    merged_df[columns_to_impute] = imputer.fit_transform(merged_df[columns_to_impute])
    
    merged_df['Age (y)'] = merged_df['Age (y)'].astype('float64')
    
    merged_df['RR_diff'] = merged_df['RR_T1 (bpm)'] - merged_df['RR_T0 (bpm)']
    merged_df['PaO2_diff'] = merged_df['PaO2_T1 (mmHg)'] - merged_df['PaO2_T0 (mmHg)']
    merged_df['FiO2_diff'] = merged_df['FiO2_T1 (%)'] - merged_df['FiO2_T0 (%)']
    merged_df['P/F_diff'] = merged_df['P/F_T1 (mmHg)'] - merged_df['P/F_T0 (mmHg)']
    merged_df['pCO2_diff'] = merged_df['pCO2_T1 (mmHg)'] - merged_df['pCO2_T0 (mmHg)']
    merged_df['SpO2_diff'] = merged_df['SpO2_T1 (%)'] - merged_df['SpO2_T0 (%)']
    merged_df['HR_diff'] = merged_df['HR_T1 (bpm)'] - merged_df['HR_T0 (bpm)']
    
    merged_df['ROX_T0'] = merged_df['SpO2_T0 (%)'] / (merged_df['FiO2_T0 (%)']/100) / merged_df['RR_T0 (bpm)']
    merged_df['ROX_T1'] = merged_df['SpO2_T1 (%)'] / (merged_df['FiO2_T1 (%)']/100) / merged_df['RR_T1 (bpm)']
    merged_df['ROX_diff'] = merged_df['ROX_T1'] - merged_df['ROX_T0']
    
    merged_df = merged_df[merged_df['RR_T1 (bpm)'] != 0]
    
    merged_df = merged_df[merged_df['P/F_T0 (mmHg)'] < 300] #300 220
    return merged_df

HFNC_eicu = eicu_HFNC_t0_t1()
HFNC_eicu.reset_index(drop=True, inplace=True)
HFNC_eicu_supp = eicu_HFNC_t0_t1()
# Try to load flow data if available, otherwise skip filtering
try:
    HFNC_eicu_flow = pd.read_csv(data_path + 'HFNC_eicu_flow.csv')
    # Select rows where respchartvalue > 55
    result = HFNC_eicu_flow[HFNC_eicu_flow['respchartvalue'] >= 30]
    # Select the patientunitstayid column
    patient_ids = result['patientunitstayid'].unique()
    filtered_HFNC_eicu = HFNC_eicu[HFNC_eicu['icustay_id'].isin(patient_ids)]
    filtered_HFNC_eicu_supp = HFNC_eicu_supp[HFNC_eicu_supp['icustay_id'].isin(patient_ids)]
    print(f"Filtered HFNC eICU data using flow criteria: {len(filtered_HFNC_eicu)} samples")
except FileNotFoundError:
    print("HFNC_eicu_flow.csv not found, using all eICU data")
    filtered_HFNC_eicu = HFNC_eicu.copy()
    filtered_HFNC_eicu_supp = HFNC_eicu_supp.copy()

# Combine eICU and MIMIC data
dataframes_to_combine = [filtered_HFNC_eicu]
if not filter_MIMIC_HFNC_roomair.empty:
    dataframes_to_combine.append(filter_MIMIC_HFNC_roomair)
    print(f"Combining eICU ({len(filtered_HFNC_eicu)}) and MIMIC ({len(filter_MIMIC_HFNC_roomair)}) data")
else:
    print(f"Using only eICU data ({len(filtered_HFNC_eicu)} samples)")

combined_df_roomair = pd.concat(dataframes_to_combine, axis=0)
# Reset index if needed (optional)
combined_df_roomair.reset_index(drop=True, inplace=True)

print(f"Combined dataset HFNC failure distribution: {combined_df_roomair['HFNO_failure'].value_counts()}")

combined_df_roomair['mROX_T0'] = combined_df_roomair['PaO2_T0 (mmHg)'] / (combined_df_roomair['FiO2_T0 (%)'] / 100 * combined_df_roomair['RR_T0 (bpm)'])
combined_df_roomair['mROX_T1'] = combined_df_roomair['PaO2_T1 (mmHg)'] / (combined_df_roomair['FiO2_T1 (%)'] / 100 * combined_df_roomair['RR_T1 (bpm)'])
combined_df_roomair['ROX_HR_T0'] = combined_df_roomair['SpO2_T0 (%)'] / (combined_df_roomair['FiO2_T0 (%)'] / 100 * combined_df_roomair['RR_T0 (bpm)'] * combined_df_roomair['HR_T0 (bpm)']) * 100
combined_df_roomair['ROX_HR_T1'] = combined_df_roomair['SpO2_T1 (%)'] / (combined_df_roomair['FiO2_T1 (%)'] / 100 * combined_df_roomair['RR_T1 (bpm)'] * combined_df_roomair['HR_T1 (bpm)']) * 100
combined_df_roomair['mROX_HR_T0'] = combined_df_roomair['PaO2_T0 (mmHg)'] / (combined_df_roomair['FiO2_T0 (%)'] / 100 * combined_df_roomair['RR_T0 (bpm)'] * combined_df_roomair['HR_T0 (bpm)']) * 100
combined_df_roomair['mROX_HR_T1'] = combined_df_roomair['PaO2_T1 (mmHg)'] / (combined_df_roomair['FiO2_T1 (%)'] / 100 * combined_df_roomair['RR_T1 (bpm)'] * combined_df_roomair['HR_T1 (bpm)']) * 100
combined_df_roomair['S/F_T0'] = combined_df_roomair['SpO2_T0 (%)'] / combined_df_roomair['FiO2_T0 (%)'] * 100
combined_df_roomair['S/F_T1'] = combined_df_roomair['SpO2_T1 (%)'] / combined_df_roomair['FiO2_T1 (%)'] * 100
combined_df_roomair['mROX_diff'] = combined_df_roomair['mROX_T1'] - combined_df_roomair['mROX_T0']
combined_df_roomair['ROX_HR_diff'] = combined_df_roomair['ROX_HR_T1'] - combined_df_roomair['ROX_HR_T0']
combined_df_roomair['mROX_HR_diff'] = combined_df_roomair['mROX_HR_T1'] - combined_df_roomair['mROX_HR_T0']
combined_df_roomair['S/F_diff'] =  combined_df_roomair['S/F_T1'] - combined_df_roomair['S/F_T0'] 
combined_df_roomair['1/RR_t0'] = 1/combined_df_roomair['RR_T0 (bpm)']
combined_df_roomair['1/RR_t1'] = 1/combined_df_roomair['RR_T1 (bpm)']
combined_df_roomair['1/FiO2_t0'] = 1/combined_df_roomair['FiO2_T0 (%)']
combined_df_roomair['1/FiO2_t1'] = 1/combined_df_roomair['FiO2_T1 (%)']

HFNO_all = pd.concat([HFNO_combined, combined_df_roomair], axis=0)
HFNO_all.reset_index(drop=True, inplace=True)


'''-------------RENOVATE dataset-------------'''

# 1. Read the original data (RENOVATE_CNAF_Warwick.xlsx) into a dataframe
df_warwick = pd.read_excel(data_path + "RENOVATE/RENOVATE_CNAF_Warwick.xlsx")

# 2. Read the data from Book3.xlsx into another dataframe
df_book3 = pd.read_excel(data_path + "RENOVATE/Book.xlsx")

# 3. Specify the columns you want to bring over from Book3.xlsx
columns_to_merge = [
    "record_id",
    "testa_g1",
    "testa_g2",
    "testa_g3",
    "testa_g4",
    "paco2_hr1",
    "iot_hr1",
    "iot_hr2",
    "iot_hr6",
    "iot_hr12",
    "iot_dia1",
    "pao2_db",
    "pao2_hr1",
    "usovni_dia1",
    "etipricovid_alt",
    'covid_ele',

    
] #    "covid_ele",

# 4. Merge the two dataframes on "record_id"
#    "how='left'" ensures that all rows from df_warwick are preserved
df_merged = pd.merge(
    df_warwick,
    df_book3[columns_to_merge],
    on="record_id",
    how="left"
)

df_merged.to_excel(data_path + "RENOVATE/RENOVATE_merged.xlsx", index=False)

RENOVATE_data = pd.read_excel(data_path + "RENOVATE/RENOVATE_merged.xlsx")

# Define the columns to check for 'yes'
iot_columns = ['iot_hr1', 'iot_hr2', 'iot_hr6', 'iot_hr12', 'iot_dia1', 'usovni_dia1', 'death24h'] #'niv_use', 'death24h'

# Create the 'HFNO_failure' column based on conditions
def determine_hfno_failure(row):
    if row[iot_columns].isna().all():
        return np.nan
    return 1 if 'Yes' in row.values else 0

RENOVATE_data['HFNO_failure'] = RENOVATE_data[iot_columns].apply(determine_hfno_failure, axis=1)

RENOVATE_data.rename(columns={ 'age':'Age (y)','rox_t0':'ROX_T0', 'rox_t1':'ROX_T1', 'heart_rate_t0': 'HR_T0 (bpm)',
                              'heart_rate_t1':'HR_T1 (bpm)', 'resp_rate_t0':'RR_T0 (bpm)',
                              'resp_rate_t1':'RR_T1 (bpm)','pao2fio2_t0':'P/F_T0 (mmHg)','pao2fio2_t1':'P/F_T1 (mmHg)',
                             'spo2_t0':'SpO2_T0 (%)','spo2_t1':'SpO2_T1 (%)','spo2fio2_t0':'S/F_T0', 'spo2fio2_t1':'S/F_T1',
                             'paco2_t0':'pCO2_T0 (mmHg)','paco2_hr1':'pCO2_T1 (mmHg)','pao2_db':'PaO2_T0 (mmHg)','pao2_hr1':'PaO2_T1 (mmHg)' }, inplace=True)

RENOVATE_data['FiO2_T0 (%)'] = RENOVATE_data['PaO2_T0 (mmHg)'] / RENOVATE_data['P/F_T0 (mmHg)'] * 100
RENOVATE_data['FiO2_T1 (%)'] = RENOVATE_data['PaO2_T1 (mmHg)'] / RENOVATE_data['P/F_T1 (mmHg)'] * 100
RENOVATE_data['RR_diff'] = RENOVATE_data['RR_T1 (bpm)'] - RENOVATE_data['RR_T0 (bpm)']
RENOVATE_data['FiO2_diff'] = RENOVATE_data['FiO2_T1 (%)'] - RENOVATE_data['FiO2_T0 (%)']
RENOVATE_data['SpO2_diff'] = RENOVATE_data['SpO2_T1 (%)'] - RENOVATE_data['SpO2_T0 (%)']
RENOVATE_data['P/F_diff'] = RENOVATE_data['P/F_T1 (mmHg)'] - RENOVATE_data['P/F_T0 (mmHg)']
RENOVATE_data['PaO2_diff'] = RENOVATE_data['PaO2_T1 (mmHg)'] - RENOVATE_data['PaO2_T0 (mmHg)']
RENOVATE_data['pCO2_diff'] = RENOVATE_data['pCO2_T1 (mmHg)'] - RENOVATE_data['pCO2_T0 (mmHg)']
RENOVATE_data['HR_diff'] = RENOVATE_data['HR_T1 (bpm)'] - RENOVATE_data['HR_T0 (bpm)']
RENOVATE_data['mROX_T0'] = RENOVATE_data['PaO2_T0 (mmHg)']/(RENOVATE_data['FiO2_T0 (%)']/100* RENOVATE_data['RR_T0 (bpm)'])
RENOVATE_data['mROX_T1'] = RENOVATE_data['PaO2_T1 (mmHg)']/(RENOVATE_data['FiO2_T1 (%)']/100* RENOVATE_data['RR_T1 (bpm)'])
RENOVATE_data['ROX_HR_T0'] = RENOVATE_data['SpO2_T0 (%)']/(RENOVATE_data['FiO2_T0 (%)']/100 * RENOVATE_data['RR_T0 (bpm)']*RENOVATE_data['HR_T0 (bpm)'])*100
RENOVATE_data['ROX_HR_T1'] = RENOVATE_data['SpO2_T1 (%)']/(RENOVATE_data['FiO2_T1 (%)']/100 * RENOVATE_data['RR_T1 (bpm)']*RENOVATE_data['HR_T1 (bpm)'])*100
RENOVATE_data['mROX_HR_T0'] = RENOVATE_data['PaO2_T0 (mmHg)'] / (RENOVATE_data['FiO2_T0 (%)'] / 100 * RENOVATE_data['RR_T0 (bpm)'] * RENOVATE_data['HR_T0 (bpm)']) * 100
RENOVATE_data['mROX_HR_T1'] = RENOVATE_data['PaO2_T1 (mmHg)'] / (RENOVATE_data['FiO2_T1 (%)'] / 100 * RENOVATE_data['RR_T1 (bpm)'] *RENOVATE_data['HR_T1 (bpm)']) * 100
RENOVATE_data['S/F_T0'] = RENOVATE_data['SpO2_T0 (%)'] / RENOVATE_data['FiO2_T0 (%)'] * 100
RENOVATE_data['S/F_T1'] = RENOVATE_data['SpO2_T1 (%)'] / RENOVATE_data['FiO2_T1 (%)'] * 100
RENOVATE_data['S/F_diff'] = RENOVATE_data['S/F_T1'] - RENOVATE_data['S/F_T0']
RENOVATE_data['ROX_diff'] = RENOVATE_data['ROX_T1'] - RENOVATE_data['ROX_T0']
RENOVATE_data['1/RR_t0'] = 1/RENOVATE_data['RR_T0 (bpm)']
RENOVATE_data['1/RR_t1'] = 1/RENOVATE_data['RR_T1 (bpm)']
RENOVATE_data['1/FiO2_t0'] = 1/RENOVATE_data['FiO2_T0 (%)']
RENOVATE_data['1/FiO2_t1'] = 1/RENOVATE_data['FiO2_T1 (%)']

#---------------------------------------------------------------------------------#
RENOVATE_data['ROX_T0'] = RENOVATE_data['SpO2_T0 (%)'] / (RENOVATE_data['FiO2_T0 (%)']/100) / RENOVATE_data['RR_T0 (bpm)']
RENOVATE_data['ROX_T1'] = RENOVATE_data['SpO2_T1 (%)'] / (RENOVATE_data['FiO2_T1 (%)']/100) / RENOVATE_data['RR_T1 (bpm)']
#---------------------------------------------------------------------------------#

def calculate_sofa_score(row):
    # Respiratory System (PaO2/FiO2)
    pao2_fio2_ratio = row["sofa_pao2"] / row["sofa_fio2"]
    if pao2_fio2_ratio >= 400:
        resp_score = 0
    elif pao2_fio2_ratio >= 300:
        resp_score = 1
    elif pao2_fio2_ratio >= 200:
        resp_score = 2
    elif pao2_fio2_ratio >= 100:
        resp_score = 3
    else:
        resp_score = 4

    # Coagulation (Platelet count)
    platelets = row["Sofa_platelets"]
    if platelets >= 150000:
        coag_score = 0
    elif platelets >= 100000:
        coag_score = 1
    elif platelets >= 50000:
        coag_score = 2
    elif platelets >= 20000:
        coag_score = 3
    else:
        coag_score = 4

    # Liver Function (Bilirubin)
    bilirubin = row["sofa_bilirubin"]
    if bilirubin < 1.2:
        liver_score = 0
    elif bilirubin < 2.0:
        liver_score = 1
    elif bilirubin < 6.0:
        liver_score = 2
    elif bilirubin < 12.0:
        liver_score = 3
    else:
        liver_score = 4

    # Cardiovascular System (MAP)
    map_value = row["sofa_map"]
    if map_value >= 70:
        cardio_score = 0
    else:
        cardio_score = 1  # Assumes no vasopressors are used

    # Central Nervous System (Glasgow Coma Scale)
    gcs = row["sofa_glasgow"]
    if gcs == 15:
        cns_score = 0
    elif gcs >= 13:
        cns_score = 1
    elif gcs >= 10:
        cns_score = 2
    elif gcs >= 6:
        cns_score = 3
    else:
        cns_score = 4

    # Renal Function (Creatinine)
    creatinine = row["sofa_creatinine"]
    if creatinine < 1.2:
        renal_score = 0
    elif creatinine < 2.0:
        renal_score = 1
    elif creatinine < 3.5:
        renal_score = 2
    elif creatinine < 5.0:
        renal_score = 3
    else:
        renal_score = 4

    # Calculate total SOFA score
    total_sofa_score = resp_score + coag_score + liver_score + cardio_score + cns_score + renal_score
    return total_sofa_score

# Apply the function to each row
RENOVATE_data["SOFA_Score"] = RENOVATE_data.apply(calculate_sofa_score, axis=1)
RENOVATE_data.rename(columns={'SOFA_Score':'SOFA' }, inplace=True)

# Filter rows where 'testa_g1' or 'testa_g2' is 1 and create a copy
RENOVATE_all = RENOVATE_data.copy()
RENOVATE_hypoxemic = RENOVATE_data[(RENOVATE_data['testa_g1'] == 1)| (RENOVATE_data['testa_g2'] == 1)].copy()
RENOVATE_nonhypoxemic = RENOVATE_data[(RENOVATE_data['testa_g3'] == 1)| (RENOVATE_data['testa_g4'] == 1)].copy()
RENOVATE_data = RENOVATE_data[(RENOVATE_data['testa_g1'] == 1)].copy()

RENOVATE_data_filtered = RENOVATE_data.dropna(subset=['FiO2_T0 (%)', 'PaO2_T0 (mmHg)', 'P/F_T0 (mmHg)', 'SpO2_T0 (%)', 'P/F_T1 (mmHg)', 'PaO2_T1 (mmHg)', 'RR_T1 (bpm)', 'pCO2_T1 (mmHg)','ROX_T0','ROX_T1']).copy()
RENOVATE_hypoxemic_filtered = RENOVATE_hypoxemic.dropna(subset=['FiO2_T0 (%)', 'PaO2_T0 (mmHg)', 'P/F_T0 (mmHg)', 'SpO2_T0 (%)', 'P/F_T1 (mmHg)', 'PaO2_T1 (mmHg)', 'RR_T1 (bpm)', 'pCO2_T1 (mmHg)','ROX_T0','ROX_T1','1/RR_t1']).copy()
RENOVATE_all_filtered = RENOVATE_all.dropna(subset=['FiO2_T0 (%)', 'PaO2_T0 (mmHg)', 'P/F_T0 (mmHg)', 'SpO2_T0 (%)', 'P/F_T1 (mmHg)', 'PaO2_T1 (mmHg)', 'RR_T1 (bpm)', 'pCO2_T1 (mmHg)','ROX_T0','ROX_T1']).copy()
RENOVATE_nonhypoxemic_filtered = RENOVATE_nonhypoxemic.dropna(subset=['FiO2_T0 (%)', 'PaO2_T0 (mmHg)', 'P/F_T0 (mmHg)', 'SpO2_T0 (%)', 'P/F_T1 (mmHg)', 'PaO2_T1 (mmHg)', 'RR_T1 (bpm)', 'pCO2_T1 (mmHg)','ROX_T0','ROX_T1']).copy()

features_all_non_aterial = ['Age (y)', 'HR_T0 (bpm)','RR_T0 (bpm)', 'FiO2_T0 (%)','SpO2_T0 (%)', 'FiO2_T1 (%)','HR_T1 (bpm)', 
                'RR_T1 (bpm)', 'SpO2_T1 (%)', 'HR_diff', 'SpO2_diff', 'FiO2_diff',  'RR_diff', 'S/F_T0', 'S/F_T1', 'S/F_diff']

features_tabpfn = ['Age (y)', 'HR_T0 (bpm)', 'RR_T0 (bpm)', 'FiO2_T1 (%)', 'RR_T1 (bpm)', 'RR_diff', 'S/F_T1']


def get_renovate_data_filtered():
    """
    Returns the RENOVATE_data_filtered dataset for training.
    This dataset contains hypoxemic patients from the RENOVATE study.
    """
    print(RENOVATE_data_filtered['HFNO_failure'].value_counts())
    return RENOVATE_data_filtered.copy()


def get_hfno_all():
    """
    Returns the HFNO_all dataset for testing.
    This dataset contains combined data from multiple sources.
    """
    print(HFNO_all['HFNO_failure'].value_counts())
    return HFNO_all.copy()


def get_features_for_training():
    """
    Returns the feature columns to use for TabPFN training.
    """
    return features_tabpfn.copy()


def get_renovate_data_all():
    return RENOVATE_data.copy()


def prepare_data_for_tabpfn(dataset, target_column='HFNO_failure'):
    """
    Prepares a dataset for TabPFN training by selecting features and target.
    
    Args:
        dataset: DataFrame containing the dataset
        target_column: Name of the target column
    
    Returns:
        X: Feature matrix
        y: Target vector
    """
    # Select the features for TabPFN
    feature_columns = get_features_for_training()
    
    # Ensure all feature columns exist in the dataset
    available_features = [col for col in feature_columns if col in dataset.columns]
    
    if len(available_features) != len(feature_columns):
        missing_features = set(feature_columns) - set(available_features)
        print(f"Warning: Missing features: {missing_features}")
    
    # Select features and target
    X = dataset[available_features].copy()
    y = dataset[target_column].copy()
    
    # Remove rows with NaN values
    # mask = ~(X.isnull().any(axis=1) | y.isnull())
    # X = X[mask]
    # y = y[mask]
    
    return X, y



