
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from typing import Tuple, List
import warnings

warnings.filterwarnings('ignore')

# Configuration
DATA_PATH = 'data/'
MISSING_THRESHOLD = 0.30  # 30% threshold for dropping subjects


def handle_missing_values(
    df: pd.DataFrame,
    missing_threshold: float = MISSING_THRESHOLD,
    n_neighbors: int = 5,
    exclude_cols: List[str] = None
) -> pd.DataFrame:
   
    if df.empty:
        return df
    
    df = df.copy()
    
    # Identify columns to exclude from imputation
    if exclude_cols is None:
        exclude_cols = []
    
    # Add common ID and target columns to exclusion list
    common_exclude = ['stay_id', 'icustay_id', 'record_id', 'Id N', 
                     'HFNO_failure', 'Death', 'patientunitstayid']
    exclude_cols = list(set(exclude_cols + common_exclude))
    
    # Get columns available for imputation
    impute_cols = [col for col in df.columns if col not in exclude_cols]
    
    initial_rows = len(df)
    
    # Step 1: Drop subjects with >30% missing values
    missing_percentage = df[impute_cols].isnull().sum(axis=1) / len(impute_cols)
    valid_subjects = missing_percentage <= missing_threshold
    df = df[valid_subjects].copy()
    
    dropped_rows = initial_rows - len(df)
    if dropped_rows > 0:
        print(f"  Dropped {dropped_rows} subjects with >{missing_threshold*100:.0f}% missing values")
    
    if df.empty:
        print("  Warning: All subjects dropped due to missing values!")
        return df
    
    # Step 2: Forward-fill imputation
    # Group by subject ID if available, otherwise treat as single group
    id_cols = [col for col in ['stay_id', 'icustay_id', 'record_id', 'patientunitstayid'] 
               if col in df.columns]
    
    if id_cols:
        # If we have subject IDs, forward fill within each subject
        for col in impute_cols:
            if df[col].isnull().any():
                df[col] = df.groupby(id_cols[0])[col].fillna(method='ffill')
    else:
        # Otherwise, forward fill across all rows
        df[impute_cols] = df[impute_cols].fillna(method='ffill')
    
    # Step 3: KNN imputation for remaining missing values
    remaining_missing = df[impute_cols].isnull().sum().sum()
    
    if remaining_missing > 0:
        print(f"  Applying KNN imputation to {remaining_missing} remaining missing values")
        
        # Separate numeric and non-numeric columns
        numeric_cols = df[impute_cols].select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            imputer = KNNImputer(n_neighbors=min(n_neighbors, len(df) - 1))
            df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    
    # Final check: fill any remaining NaNs with median (backup)
    for col in impute_cols:
        if df[col].isnull().any():
            if df[col].dtype in [np.float64, np.int64, np.float32, np.int32]:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 0)
    
    return df


def load_DNI() -> pd.DataFrame:
   
    print("\nLoading DNI dataset...")
    
    HFNO = pd.read_excel(DATA_PATH + 'HFNO.xlsx')
    
    # T0 data
    HFNO_T0 = HFNO[[
        'Id N', 'Age (y)', 'Sex 0=F, 1=M', 'BMI (km/m2)', 
        'RICU admission diagnosis 0=pneumonia, 1=ARDS',
        'Solid tumor 0=no, 1=yes', 'Lymphoma/leukemia 0=no, 1=yes', 
        'Dementia 0=no, 1=yes', 'Metastatic 0=no, 1=yes',
        'CHF 0=no, 1=yes', 'COPD 0=no, 1=yes', 'Charlson index', 'SOFA', 
        'Creatinine (mg/dl)', 'BORG_T0',
        'HACOR_T0', 'ROX_T0', 'Δpes_T0 (cmH2O)', 'HR_T0 (bpm)', 
        'Sistolic_blood_pressure_T0 (mmHg)', 'RR_T0 (bpm)',
        'FiO2_T0 (%)', 'PaO2_T0  (mmHg)', 'P/F_T0 (mmHg)', 'SpO2_T0 (%)', 
        'pH_A_T0', 'pCO2_A_T0 (mmHg)',
        'BE_A_T0 (mmol/L)', 'HCO3-_T0 (mmol/L)', 'Lactate_T0 (mmol/L)', 
        'HFNO_failure', 'Death'
    ]].copy()
    
    HFNO_T0.drop(['Id N', 'BORG_T0'], axis=1, inplace=True)
    HFNO_T0.rename(columns={
        'Charlson index': 'Charlson_index', 
        'BMI (km/m2)': 'BMI (kg/m2)', 
        'RICU admission diagnosis 0=pneumonia, 1=ARDS': 'Diagnosis',
        'pH_A_T0': 'pH_T0', 
        'pCO2_A_T0 (mmHg)': 'pCO2_T0 (mmHg)', 
        'BE_A_T0 (mmol/L)': 'BE_T0 (mmol/L)', 
        'PaO2_T0  (mmHg)': 'PaO2_T0 (mmHg)'
    }, inplace=True)
    
    # T1 data
    HFNO_T1 = HFNO[[
        'Id N', 'Age (y)', 'Sex 0=F, 1=M', 'BMI (km/m2)',
        'RICU admission diagnosis 0=pneumonia, 1=ARDS', 'Charlson index', 
        'SOFA', 'Creatinine (mg/dl)',
        'BORG_T1', 'HACOR_T1', 'ROX_T1',
        'Δpes_T1 (cmH2O)', 'HR_T1(bpm)', 'Sistolic_blood_pressure_T1 (mmHg)',
        'RR_T1 (bpm)', 'FiO2_T1 (%)', 'PaO2_T1  (mmHg)', 'P/F_T1 (mmHg)',
        'SpO2_T1 (%)', 'pH_A_T1', 'pCO2_A_T1 (mmHg)', 'BE_A_T1 (mmol/L)',
        'HCO3-_T1 (mmol/L)', 'Lactate (mmol/L)', 'HFNO_failure',
        'NIV_upgrade_theoretically', 'NIV_upgrade_actually', 'Death'
    ]].copy()
    
    HFNO_T1.drop(['Id N'], axis=1, inplace=True)
    HFNO_T1.rename(columns={
        'RICU admission diagnosis 0=pneumonia, 1=ARDS': 'Diagnosis',
        'HR_T1(bpm)': 'HR_T1 (bpm)', 
        'Δpes_T1 (cmH2O)': 'Δpes_T1 (cmH2O)',
        'RR_T1 (bpm)': 'RR_T1 (bpm)', 
        'FiO2_T1 (%)': 'FiO2_T1 (%)', 
        'PaO2_T1  (mmHg)': 'PaO2_T1 (mmHg)',
        'P/F_T1 (mmHg)': 'P/F_T1 (mmHg)', 
        'pH_A_T1': 'pH_T1', 
        'pCO2_A_T1 (mmHg)': 'pCO2_T1 (mmHg)'
    }, inplace=True)
    
    # Combine T0 and T1
    unique_columns_df1 = HFNO_T1.columns.difference(HFNO_T0.columns)
    df1_unique = HFNO_T1[unique_columns_df1]
    HFNO_T0_reset = HFNO_T0.reset_index(drop=True)
    df1_unique_reset = df1_unique.reset_index(drop=True)
    HFNO_combined = pd.concat([HFNO_T0_reset, df1_unique_reset], axis=1)
    
    # Calculate derived features
    HFNO_combined['HACOR_diff'] = HFNO_combined['HACOR_T1'] - HFNO_combined['HACOR_T0']
    HFNO_combined['ROX_diff'] = HFNO_combined['ROX_T1'] - HFNO_combined['ROX_T0']
    HFNO_combined['HR_diff'] = HFNO_combined['HR_T1 (bpm)'] - HFNO_combined['HR_T0 (bpm)']
    HFNO_combined['SpO2_diff'] = HFNO_combined['SpO2_T1 (%)'] - HFNO_combined['SpO2_T0 (%)']
    HFNO_combined['FiO2_diff'] = HFNO_combined['FiO2_T1 (%)'] - HFNO_combined['FiO2_T0 (%)']
    HFNO_combined['PaO2_diff'] = HFNO_combined['PaO2_T1 (mmHg)'] - HFNO_combined['PaO2_T0 (mmHg)']
    HFNO_combined['pCO2_diff'] = HFNO_combined['pCO2_T1 (mmHg)'] - HFNO_combined['pCO2_T0 (mmHg)']
    HFNO_combined['P/F_diff'] = HFNO_combined['P/F_T1 (mmHg)'] - HFNO_combined['P/F_T0 (mmHg)']
    HFNO_combined['RR_diff'] = HFNO_combined['RR_T1 (bpm)'] - HFNO_combined['RR_T0 (bpm)']
    HFNO_combined['1/RR_t0'] = 1 / HFNO_combined['RR_T0 (bpm)']
    HFNO_combined['1/RR_t1'] = 1 / HFNO_combined['RR_T1 (bpm)']
    
    # Handle missing values
    HFNO_combined = handle_missing_values(
        HFNO_combined, 
        exclude_cols=['HFNO_failure', 'Death']
    )
    
    print(f"  Final DNI dataset: {len(HFNO_combined)} samples")
    return HFNO_combined


def load_nonDNI() -> pd.DataFrame:
    """Load and preprocess non-DNI dataset with proper missing value handling."""
    print("\nLoading non-DNI dataset...")
    
    HFNO_nonDNI = pd.read_excel(DATA_PATH + 'AJRCCM2023_dataset.xlsx')
    
    # T0 data
    HFNO_nonDNI_t0 = HFNO_nonDNI[(HFNO_nonDNI['Time_point'] == 0)].copy()
    HFNO_nonDNI_t0.rename(columns={
        'HFNC_failure_to_NIV': 'HFNO_failure', 'Age': 'Age (y)', 
        'SOFA_score': 'SOFA', 'HACOR_score': 'HACOR_T0',
        'ROX_index': 'ROX_T0', 'HR': 'HR_T0 (bpm)', 'SpO2': 'SpO2_T0 (%)', 
        'FiO2': 'FiO2_T0 (%)', 'PO2': 'PaO2_T0 (mmHg)',
        'PCO2': 'pCO2_T0 (mmHg)', 'P/F': 'P/F_T0 (mmHg)', 'RR': 'RR_T0 (bpm)'
    }, inplace=True)
    HFNO_nonDNI_t0.loc[:, 'FiO2_T0 (%)'] = HFNO_nonDNI_t0['FiO2_T0 (%)'] * 100
    
    # T1 data
    HFNO_nonDNI_t1 = HFNO_nonDNI[(HFNO_nonDNI['Time_point'] == 1)].copy()
    HFNO_nonDNI_t1.rename(columns={
        'HFNC_failure_to_NIV': 'HFNO_failure', 'Age': 'Age (y)', 
        'SOFA_score': 'SOFA', 'HACOR_score': 'HACOR_T1',
        'ROX_index': 'ROX_T1', 'HR': 'HR_T1 (bpm)', 'SpO2': 'SpO2_T1 (%)', 
        'FiO2': 'FiO2_T1 (%)', 'PO2': 'PaO2_T1 (mmHg)',
        'PCO2': 'pCO2_T1 (mmHg)', 'P/F': 'P/F_T1 (mmHg)', 'RR': 'RR_T1 (bpm)'
    }, inplace=True)
    HFNO_nonDNI_t1.loc[:, 'FiO2_T1 (%)'] = HFNO_nonDNI_t1['FiO2_T1 (%)'] * 100
    
    # Combine T0 and T1
    unique_columns_df1 = HFNO_nonDNI_t1.columns.difference(HFNO_nonDNI_t0.columns)
    df1_unique = HFNO_nonDNI_t1[unique_columns_df1]
    HFNO_nonDNI_t0_reset = HFNO_nonDNI_t0.reset_index(drop=True)
    df1_unique_reset = df1_unique.reset_index(drop=True)
    HFNO_combined = pd.concat([HFNO_nonDNI_t0_reset, df1_unique_reset], axis=1)
    
    # Calculate derived features
    HFNO_combined['HACOR_diff'] = HFNO_combined['HACOR_T1'] - HFNO_combined['HACOR_T0']
    HFNO_combined['ROX_diff'] = HFNO_combined['ROX_T1'] - HFNO_combined['ROX_T0']
    HFNO_combined['HR_diff'] = HFNO_combined['HR_T1 (bpm)'] - HFNO_combined['HR_T0 (bpm)']
    HFNO_combined['SpO2_diff'] = HFNO_combined['SpO2_T1 (%)'] - HFNO_combined['SpO2_T0 (%)']
    HFNO_combined['FiO2_diff'] = HFNO_combined['FiO2_T1 (%)'] - HFNO_combined['FiO2_T0 (%)']
    HFNO_combined['PaO2_diff'] = HFNO_combined['PaO2_T1 (mmHg)'] - HFNO_combined['PaO2_T0 (mmHg)']
    HFNO_combined['pCO2_diff'] = HFNO_combined['pCO2_T1 (mmHg)'] - HFNO_combined['pCO2_T0 (mmHg)']
    HFNO_combined['P/F_diff'] = HFNO_combined['P/F_T1 (mmHg)'] - HFNO_combined['P/F_T0 (mmHg)']
    HFNO_combined['RR_diff'] = HFNO_combined['RR_T1 (bpm)'] - HFNO_combined['RR_T0 (bpm)']
    HFNO_combined['1/RR_t0'] = 1 / HFNO_combined['RR_T0 (bpm)']
    HFNO_combined['1/RR_t1'] = 1 / HFNO_combined['RR_T1 (bpm)']
    
    # Handle missing values
    HFNO_combined = handle_missing_values(
        HFNO_combined,
        exclude_cols=['HFNO_failure']
    )
    
    print(f"  Final non-DNI dataset: {len(HFNO_combined)} samples")
    return HFNO_combined


def load_MIMIC_HFNC() -> pd.DataFrame:
    """Load and preprocess MIMIC-IV HFNC dataset with proper missing value handling."""
    print("\nLoading MIMIC-IV HFNC dataset...")
    
    try:
        MIMIC_HFNC_t0 = pd.read_csv(DATA_PATH + 'HFNC_t0.csv')
        MIMIC_HFNC_t1 = pd.read_csv(DATA_PATH + 'HFNC_t1.csv')
    except FileNotFoundError as e:
        print(f"  MIMIC data files not found: {e}")
        return pd.DataFrame()
    
    if MIMIC_HFNC_t0.empty or MIMIC_HFNC_t1.empty:
        print("  Empty MIMIC datasets, returning empty DataFrame")
        return pd.DataFrame()
    
    # Rename T0 columns
    MIMIC_HFNC_t0 = MIMIC_HFNC_t0.rename(
        columns=lambda x: f"{x}_t0" if x != 'stay_id' else x
    )
    
    # Merge T0 and T1
    MIMIC_HFNC = pd.merge(MIMIC_HFNC_t0, MIMIC_HFNC_t1, on='stay_id')
    
    MIMIC_HFNC.rename(columns={
        'po2_t0': 'PaO2_T0 (mmHg)', 'fio2_t0': 'FiO2_T0 (%)', 
        'pco2_t0': 'pCO2_T0 (mmHg)',
        'pao2fio2ratio_t0': 'P/F_T0 (mmHg)', 'spo2_t0': 'SpO2_T0 (%)',
        'ph_t0': 'pH_T0', 'age': 'Age (y)', 'resp_rate_t0': 'RR_T0 (bpm)', 
        'heart_rate_t0': 'HR_T0 (bpm)',
        'po2': 'PaO2_T1 (mmHg)', 'pco2': 'pCO2_T1 (mmHg)', 'spo2': 'SpO2_T1 (%)',
        'fio2': 'FiO2_T1 (%)', 'ph': 'pH_T1', 'resp_rate': 'RR_T1 (bpm)', 
        'heart_rate': 'HR_T1 (bpm)',
        'pao2fio2ratio': 'P/F_T1 (mmHg)'
    }, inplace=True)
    
    # Define HFNO failure
    MIMIC_HFNC['HFNO_failure'] = np.where(
        (MIMIC_HFNC['intubation_status'] == 'intubated') | 
        (MIMIC_HFNC['survive_days_from_icu'] <= 4), 1, 0
    )
    
    # Handle FiO2
    MIMIC_HFNC['FiO2_T0 (%)'] = MIMIC_HFNC['FiO2_T0 (%)'].fillna(
        MIMIC_HFNC['fio2_chartevents_t0']
    )
    MIMIC_HFNC['FiO2_T0 (%)'] = MIMIC_HFNC['FiO2_T0 (%)'].replace(0, np.nan)
    MIMIC_HFNC['FiO2_T0 (%)'] = MIMIC_HFNC['FiO2_T0 (%)'].fillna(21)
    MIMIC_HFNC['FiO2_T1 (%)'] = MIMIC_HFNC['FiO2_T1 (%)'].fillna(
        MIMIC_HFNC['fio2_chartevents']
    )
    
    # Calculate P/F from FiO2
    MIMIC_HFNC['P/F_T0 (mmHg)'] = (
        MIMIC_HFNC['PaO2_T0 (mmHg)'] / MIMIC_HFNC['FiO2_T0 (%)'] * 100
    )
    
    # Handle missing values
    MIMIC_HFNC = handle_missing_values(
        MIMIC_HFNC,
        exclude_cols=['stay_id', 'HFNO_failure']
    )
    
    # Calculate derived features
    MIMIC_HFNC['RR_diff'] = MIMIC_HFNC['RR_T1 (bpm)'] - MIMIC_HFNC['RR_T0 (bpm)']
    MIMIC_HFNC['PaO2_diff'] = MIMIC_HFNC['PaO2_T1 (mmHg)'] - MIMIC_HFNC['PaO2_T0 (mmHg)']
    MIMIC_HFNC['FiO2_diff'] = MIMIC_HFNC['FiO2_T1 (%)'] - MIMIC_HFNC['FiO2_T0 (%)']
    MIMIC_HFNC['P/F_diff'] = MIMIC_HFNC['P/F_T1 (mmHg)'] - MIMIC_HFNC['P/F_T0 (mmHg)']
    MIMIC_HFNC['pCO2_diff'] = MIMIC_HFNC['pCO2_T1 (mmHg)'] - MIMIC_HFNC['pCO2_T0 (mmHg)']
    MIMIC_HFNC['SpO2_diff'] = MIMIC_HFNC['SpO2_T1 (%)'] - MIMIC_HFNC['SpO2_T0 (%)']
    MIMIC_HFNC['HR_diff'] = MIMIC_HFNC['HR_T1 (bpm)'] - MIMIC_HFNC['HR_T0 (bpm)']
    MIMIC_HFNC['ROX_T0'] = (
        MIMIC_HFNC['SpO2_T0 (%)'] / (MIMIC_HFNC['FiO2_T0 (%)'] / 100) / 
        MIMIC_HFNC['RR_T0 (bpm)']
    )
    MIMIC_HFNC['ROX_T1'] = (
        MIMIC_HFNC['SpO2_T1 (%)'] / (MIMIC_HFNC['FiO2_T1 (%)'] / 100) / 
        MIMIC_HFNC['RR_T1 (bpm)']
    )
    MIMIC_HFNC['ROX_diff'] = MIMIC_HFNC['ROX_T1'] - MIMIC_HFNC['ROX_T0']
    
    # Filter by P/F ratio
    MIMIC_HFNC = MIMIC_HFNC[MIMIC_HFNC['P/F_T0 (mmHg)'] < 300]
    
    print(f"  Final MIMIC-IV HFNC dataset: {len(MIMIC_HFNC)} samples")
    if not MIMIC_HFNC.empty:
        print(f"  HFNO failure distribution: {MIMIC_HFNC['HFNO_failure'].value_counts().to_dict()}")
    
    return MIMIC_HFNC


def load_eICU_HFNC() -> pd.DataFrame:
    """Load and preprocess eICU HFNC dataset with proper missing value handling."""
    print("\nLoading eICU HFNC dataset...")
    
    try:
        HFNC_eicu = pd.read_csv(DATA_PATH + 'HFNC_eicu.csv')
        HFNC_eicu_supp = pd.read_csv(DATA_PATH + 'HFNC_eicu_supp.csv')
    except FileNotFoundError as e:
        print(f"  eICU data files not found: {e}")
        return pd.DataFrame()
    
    # Calculate time differences and select closest records
    HFNC_eicu['labtime_t0_diff'] = abs(HFNC_eicu['labtime_t0'] - HFNC_eicu['vent_start'])
    HFNC_eicu['nursingcharttime_t0_diff'] = abs(
        HFNC_eicu['nursingcharttime_t0'] - HFNC_eicu['vent_start']
    )
    HFNC_eicu['labtime_t1_diff'] = abs(
        HFNC_eicu['labtime_t1'] - (HFNC_eicu['vent_start'] + 120)
    )
    HFNC_eicu['nursingcharttime_t1_diff'] = abs(
        HFNC_eicu['nursingcharttime_t1'] - (HFNC_eicu['vent_start'] + 120)
    )
    
    HFNC_eicu_sorted = HFNC_eicu.sort_values(by=[
        'icustay_id', 'labtime_t0_diff', 'nursingcharttime_t0_diff',
        'labtime_t1_diff', 'nursingcharttime_t1_diff'
    ])
    
    result_HFNC_eicu = HFNC_eicu_sorted.groupby('icustay_id').first().reset_index()
    result_HFNC_eicu = result_HFNC_eicu.drop(columns=[
        'labtime_t0_diff', 'nursingcharttime_t0_diff',
        'labtime_t1_diff', 'nursingcharttime_t1_diff'
    ])
    
    # Rename columns
    result_HFNC_eicu.rename(columns={
        'pao2_t0': 'PaO2_T0 (mmHg)', 'fio2_t0': 'FiO2_T0 (%)', 
        'paco2_t0': 'pCO2_T0 (mmHg)',
        'o2sat_t0': 'SpO2_T0 (%)', 'ph_t0': 'pH_T0', 
        'rr_t0': 'RR_T0 (bpm)', 'hr_t0': 'HR_T0 (bpm)',
        'pao2_t1': 'PaO2_T1 (mmHg)', 'paco2_t1': 'pCO2_T1 (mmHg)', 
        'o2sat_t1': 'SpO2_T1 (%)',
        'fio2_t1': 'FiO2_T1 (%)', 'ph_t1': 'pH_T1', 
        'rr_t1': 'RR_T1 (bpm)', 'hr_t1': 'HR_T1 (bpm)'
    }, inplace=True)
    
    # Calculate P/F ratios
    result_HFNC_eicu['P/F_T0 (mmHg)'] = (
        result_HFNC_eicu['PaO2_T0 (mmHg)'] / result_HFNC_eicu['FiO2_T0 (%)'] * 100
    )
    result_HFNC_eicu['P/F_T1 (mmHg)'] = (
        result_HFNC_eicu['PaO2_T1 (mmHg)'] / result_HFNC_eicu['FiO2_T1 (%)'] * 100
    )
    
    # Merge with supplementary data
    HFNC_eicu_supp.rename(columns={
        'age': 'Age (y)', 
        'patientunitstayid': 'icustay_id'
    }, inplace=True)
    
    merged_df = pd.merge(result_HFNC_eicu, HFNC_eicu_supp, on='icustay_id', how='inner')
    
    # Define HFNO failure
    condition_1 = (
        merged_df['second_oxygen_therapy_type'].isin([2, 3, 4]) | 
        merged_df['last_oxygen_therapy_type'].isin([2, 3, 4])
    )
    condition_2 = (
        (merged_df['actualicumortality'] == 'EXPIRED') & 
        ((merged_df['hospitaldischargeoffset'] - merged_df['vent_start']) / 60 <= 128)
    )
    merged_df['HFNO_failure'] = (condition_1 | condition_2).astype(int)
    
    # Handle missing values
    merged_df = handle_missing_values(
        merged_df,
        exclude_cols=['icustay_id', 'HFNO_failure']
    )
    
    # Ensure Age is float
    merged_df['Age (y)'] = merged_df['Age (y)'].astype('float64')
    
    # Calculate derived features
    merged_df['RR_diff'] = merged_df['RR_T1 (bpm)'] - merged_df['RR_T0 (bpm)']
    merged_df['PaO2_diff'] = merged_df['PaO2_T1 (mmHg)'] - merged_df['PaO2_T0 (mmHg)']
    merged_df['FiO2_diff'] = merged_df['FiO2_T1 (%)'] - merged_df['FiO2_T0 (%)']
    merged_df['P/F_diff'] = merged_df['P/F_T1 (mmHg)'] - merged_df['P/F_T0 (mmHg)']
    merged_df['pCO2_diff'] = merged_df['pCO2_T1 (mmHg)'] - merged_df['pCO2_T0 (mmHg)']
    merged_df['SpO2_diff'] = merged_df['SpO2_T1 (%)'] - merged_df['SpO2_T0 (%)']
    merged_df['HR_diff'] = merged_df['HR_T1 (bpm)'] - merged_df['HR_T0 (bpm)']
    merged_df['ROX_T0'] = (
        merged_df['SpO2_T0 (%)'] / (merged_df['FiO2_T0 (%)'] / 100) / 
        merged_df['RR_T0 (bpm)']
    )
    merged_df['ROX_T1'] = (
        merged_df['SpO2_T1 (%)'] / (merged_df['FiO2_T1 (%)'] / 100) / 
        merged_df['RR_T1 (bpm)']
    )
    merged_df['ROX_diff'] = merged_df['ROX_T1'] - merged_df['ROX_T0']
    
    # Filter out invalid data
    merged_df = merged_df[merged_df['RR_T1 (bpm)'] != 0]
    merged_df = merged_df[merged_df['P/F_T0 (mmHg)'] < 300]
    
    # Apply flow filter if available
    try:
        HFNC_eicu_flow = pd.read_csv(DATA_PATH + 'HFNC_eicu_flow.csv')
        result = HFNC_eicu_flow[HFNC_eicu_flow['respchartvalue'] >= 30]
        patient_ids = result['patientunitstayid'].unique()
        merged_df = merged_df[merged_df['icustay_id'].isin(patient_ids)]
        print(f"  Applied flow filter (>=30 L/min)")
    except FileNotFoundError:
        print(f"  Flow filter file not found, using all eICU data")
    
    print(f"  Final eICU HFNC dataset: {len(merged_df)} samples")
    if not merged_df.empty:
        print(f"  HFNO failure distribution: {merged_df['HFNO_failure'].value_counts().to_dict()}")
    
    return merged_df


def load_RENOVATE() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
 
    print("\nLoading RENOVATE dataset...")
    
    # Read original data
    df_warwick = pd.read_excel(DATA_PATH + "RENOVATE/RENOVATE_CNAF.xlsx")
    df_book3 = pd.read_excel(DATA_PATH + "RENOVATE/Book.xlsx")
    
    # Columns to merge
    columns_to_merge = [
        "record_id", "testa_g1", "testa_g2", "testa_g3", "testa_g4",
        "paco2_hr1", "iot_hr1", "iot_hr2", "iot_hr6", "iot_hr12",
        "iot_dia1", "pao2_db", "pao2_hr1", "usovni_dia1", 
        "etipricovid_alt", 'covid_ele'
    ]
    
    # Merge dataframes
    df_merged = pd.merge(
        df_warwick,
        df_book3[columns_to_merge],
        on="record_id",
        how="left"
    )
    
    RENOVATE_data = df_merged.copy()
    
    # Define HFNO failure
    iot_columns = ['iot_hr1', 'iot_hr2', 'iot_hr6', 'iot_hr12', 
                   'iot_dia1', 'usovni_dia1', 'death24h']
    
    def determine_hfno_failure(row):
        if row[iot_columns].isna().all():
            return np.nan
        return 1 if 'Yes' in row.values else 0
    
    RENOVATE_data['HFNO_failure'] = RENOVATE_data[iot_columns].apply(
        determine_hfno_failure, axis=1
    )
    

    RENOVATE_data.rename(columns={
        'age': 'Age (y)', 'rox_t0': 'ROX_T0', 'rox_t1': 'ROX_T1',
        'heart_rate_t0': 'HR_T0 (bpm)', 'heart_rate_t1': 'HR_T1 (bpm)',
        'resp_rate_t0': 'RR_T0 (bpm)', 'resp_rate_t1': 'RR_T1 (bpm)',
        'pao2fio2_t0': 'P/F_T0 (mmHg)', 'pao2fio2_t1': 'P/F_T1 (mmHg)',
        'spo2_t0': 'SpO2_T0 (%)', 'spo2_t1': 'SpO2_T1 (%)',
        'spo2fio2_t0': 'S/F_T0', 'spo2fio2_t1': 'S/F_T1',
        'paco2_t0': 'pCO2_T0 (mmHg)', 'paco2_hr1': 'pCO2_T1 (mmHg)',
        'pao2_db': 'PaO2_T0 (mmHg)', 'pao2_hr1': 'PaO2_T1 (mmHg)'
    }, inplace=True)
    
   
    RENOVATE_data['FiO2_T0 (%)'] = (
        RENOVATE_data['PaO2_T0 (mmHg)'] / RENOVATE_data['P/F_T0 (mmHg)'] * 100
    )
    RENOVATE_data['FiO2_T1 (%)'] = (
        RENOVATE_data['PaO2_T1 (mmHg)'] / RENOVATE_data['P/F_T1 (mmHg)'] * 100
    )
    
    # Calculate SOFA score
    def calculate_sofa_score(row):
   
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
        

        map_value = row["sofa_map"]
        cardio_score = 0 if map_value >= 70 else 1
        
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
        
        return resp_score + coag_score + liver_score + cardio_score + cns_score + renal_score
    
    RENOVATE_data["SOFA"] = RENOVATE_data.apply(calculate_sofa_score, axis=1)
    

    RENOVATE_data['RR_diff'] = RENOVATE_data['RR_T1 (bpm)'] - RENOVATE_data['RR_T0 (bpm)']
    RENOVATE_data['FiO2_diff'] = RENOVATE_data['FiO2_T1 (%)'] - RENOVATE_data['FiO2_T0 (%)']
    RENOVATE_data['SpO2_diff'] = RENOVATE_data['SpO2_T1 (%)'] - RENOVATE_data['SpO2_T0 (%)']
    RENOVATE_data['P/F_diff'] = RENOVATE_data['P/F_T1 (mmHg)'] - RENOVATE_data['P/F_T0 (mmHg)']
    RENOVATE_data['PaO2_diff'] = RENOVATE_data['PaO2_T1 (mmHg)'] - RENOVATE_data['PaO2_T0 (mmHg)']
    RENOVATE_data['pCO2_diff'] = RENOVATE_data['pCO2_T1 (mmHg)'] - RENOVATE_data['pCO2_T0 (mmHg)']
    RENOVATE_data['HR_diff'] = RENOVATE_data['HR_T1 (bpm)'] - RENOVATE_data['HR_T0 (bpm)']
    RENOVATE_data['mROX_T0'] = (
        RENOVATE_data['PaO2_T0 (mmHg)'] / 
        (RENOVATE_data['FiO2_T0 (%)'] / 100 * RENOVATE_data['RR_T0 (bpm)'])
    )
    RENOVATE_data['mROX_T1'] = (
        RENOVATE_data['PaO2_T1 (mmHg)'] / 
        (RENOVATE_data['FiO2_T1 (%)'] / 100 * RENOVATE_data['RR_T1 (bpm)'])
    )
    RENOVATE_data['ROX_HR_T0'] = (
        RENOVATE_data['SpO2_T0 (%)'] / 
        (RENOVATE_data['FiO2_T0 (%)'] / 100 * RENOVATE_data['RR_T0 (bpm)'] * 
         RENOVATE_data['HR_T0 (bpm)']) * 100
    )
    RENOVATE_data['ROX_HR_T1'] = (
        RENOVATE_data['SpO2_T1 (%)'] / 
        (RENOVATE_data['FiO2_T1 (%)'] / 100 * RENOVATE_data['RR_T1 (bpm)'] * 
         RENOVATE_data['HR_T1 (bpm)']) * 100
    )
    RENOVATE_data['mROX_HR_T0'] = (
        RENOVATE_data['PaO2_T0 (mmHg)'] / 
        (RENOVATE_data['FiO2_T0 (%)'] / 100 * RENOVATE_data['RR_T0 (bpm)'] * 
         RENOVATE_data['HR_T0 (bpm)']) * 100
    )
    RENOVATE_data['mROX_HR_T1'] = (
        RENOVATE_data['PaO2_T1 (mmHg)'] / 
        (RENOVATE_data['FiO2_T1 (%)'] / 100 * RENOVATE_data['RR_T1 (bpm)'] * 
         RENOVATE_data['HR_T1 (bpm)']) * 100
    )
    RENOVATE_data['S/F_diff'] = RENOVATE_data['S/F_T1'] - RENOVATE_data['S/F_T0']
    RENOVATE_data['ROX_diff'] = RENOVATE_data['ROX_T1'] - RENOVATE_data['ROX_T0']
    RENOVATE_data['1/RR_t0'] = 1 / RENOVATE_data['RR_T0 (bpm)']
    RENOVATE_data['1/RR_t1'] = 1 / RENOVATE_data['RR_T1 (bpm)']
    RENOVATE_data['1/FiO2_t0'] = 1 / RENOVATE_data['FiO2_T0 (%)']
    RENOVATE_data['1/FiO2_t1'] = 1 / RENOVATE_data['FiO2_T1 (%)']
    

    RENOVATE_all = RENOVATE_data.copy()
    RENOVATE_hypoxemic = RENOVATE_data[
        (RENOVATE_data['testa_g1'] == 1) | (RENOVATE_data['testa_g2'] == 1)
    ].copy()
    RENOVATE_nonhypoxemic = RENOVATE_data[
        (RENOVATE_data['testa_g3'] == 1) | (RENOVATE_data['testa_g4'] == 1)
    ].copy()
    RENOVATE_data = RENOVATE_data[(RENOVATE_data['testa_g1'] == 1)].copy()
    

    essential_cols = [
        'FiO2_T0 (%)', 'PaO2_T0 (mmHg)', 'P/F_T0 (mmHg)', 'SpO2_T0 (%)',
        'P/F_T1 (mmHg)', 'PaO2_T1 (mmHg)', 'RR_T1 (bpm)', 'pCO2_T1 (mmHg)',
        'ROX_T0', 'ROX_T1'
    ]
    
    print("\n  Processing RENOVATE subgroups:")
    print("  - Group 1 (Primary hypoxemic)...")
    RENOVATE_data = handle_missing_values(
        RENOVATE_data,
        exclude_cols=['record_id', 'HFNO_failure']
    )
    RENOVATE_data_filtered = RENOVATE_data.dropna(subset=essential_cols).copy()
    print(f"    After filtering: {len(RENOVATE_data_filtered)} samples")
    
    print("  - Hypoxemic (Groups 1&2)...")
    RENOVATE_hypoxemic = handle_missing_values(
        RENOVATE_hypoxemic,
        exclude_cols=['record_id', 'HFNO_failure']
    )
    RENOVATE_hypoxemic_filtered = RENOVATE_hypoxemic.dropna(
        subset=essential_cols + ['1/RR_t1']
    ).copy()
    print(f"    After filtering: {len(RENOVATE_hypoxemic_filtered)} samples")
    
    print("  - All groups...")
    RENOVATE_all = handle_missing_values(
        RENOVATE_all,
        exclude_cols=['record_id', 'HFNO_failure']
    )
    RENOVATE_all_filtered = RENOVATE_all.dropna(subset=essential_cols).copy()
    print(f"    After filtering: {len(RENOVATE_all_filtered)} samples")
    
    print("  - Non-hypoxemic (Groups 3&4)...")
    RENOVATE_nonhypoxemic = handle_missing_values(
        RENOVATE_nonhypoxemic,
        exclude_cols=['record_id', 'HFNO_failure']
    )
    RENOVATE_nonhypoxemic_filtered = RENOVATE_nonhypoxemic.dropna(
        subset=essential_cols
    ).copy()
    print(f"    After filtering: {len(RENOVATE_nonhypoxemic_filtered)} samples")
    
    return (RENOVATE_data_filtered, RENOVATE_hypoxemic_filtered,
            RENOVATE_all_filtered, RENOVATE_nonhypoxemic_filtered)


def add_additional_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add additional derived features to the dataset."""
    df = df.copy()
    
    # Calculate additional ROX variants
    if all(col in df.columns for col in ['PaO2_T0 (mmHg)', 'FiO2_T0 (%)', 'RR_T0 (bpm)']):
        df['mROX_T0'] = (
            df['PaO2_T0 (mmHg)'] / (df['FiO2_T0 (%)'] / 100 * df['RR_T0 (bpm)'])
        )
    if all(col in df.columns for col in ['PaO2_T1 (mmHg)', 'FiO2_T1 (%)', 'RR_T1 (bpm)']):
        df['mROX_T1'] = (
            df['PaO2_T1 (mmHg)'] / (df['FiO2_T1 (%)'] / 100 * df['RR_T1 (bpm)'])
        )
    
    if all(col in df.columns for col in ['SpO2_T0 (%)', 'FiO2_T0 (%)', 'RR_T0 (bpm)', 'HR_T0 (bpm)']):
        df['ROX_HR_T0'] = (
            df['SpO2_T0 (%)'] / 
            (df['FiO2_T0 (%)'] / 100 * df['RR_T0 (bpm)'] * df['HR_T0 (bpm)']) * 100
        )
    if all(col in df.columns for col in ['SpO2_T1 (%)', 'FiO2_T1 (%)', 'RR_T1 (bpm)', 'HR_T1 (bpm)']):
        df['ROX_HR_T1'] = (
            df['SpO2_T1 (%)'] / 
            (df['FiO2_T1 (%)'] / 100 * df['RR_T1 (bpm)'] * df['HR_T1 (bpm)']) * 100
        )
    
    if all(col in df.columns for col in ['PaO2_T0 (mmHg)', 'FiO2_T0 (%)', 'RR_T0 (bpm)', 'HR_T0 (bpm)']):
        df['mROX_HR_T0'] = (
            df['PaO2_T0 (mmHg)'] / 
            (df['FiO2_T0 (%)'] / 100 * df['RR_T0 (bpm)'] * df['HR_T0 (bpm)']) * 100
        )
    if all(col in df.columns for col in ['PaO2_T1 (mmHg)', 'FiO2_T1 (%)', 'RR_T1 (bpm)', 'HR_T1 (bpm)']):
        df['mROX_HR_T1'] = (
            df['PaO2_T1 (mmHg)'] / 
            (df['FiO2_T1 (%)'] / 100 * df['RR_T1 (bpm)'] * df['HR_T1 (bpm)']) * 100
        )
    
    # S/F ratio
    if all(col in df.columns for col in ['SpO2_T0 (%)', 'FiO2_T0 (%)']):
        df['S/F_T0'] = df['SpO2_T0 (%)'] / df['FiO2_T0 (%)'] * 100
    if all(col in df.columns for col in ['SpO2_T1 (%)', 'FiO2_T1 (%)']):
        df['S/F_T1'] = df['SpO2_T1 (%)'] / df['FiO2_T1 (%)'] * 100
    
    # Difference features
    if 'mROX_T1' in df.columns and 'mROX_T0' in df.columns:
        df['mROX_diff'] = df['mROX_T1'] - df['mROX_T0']
    if 'ROX_HR_T1' in df.columns and 'ROX_HR_T0' in df.columns:
        df['ROX_HR_diff'] = df['ROX_HR_T1'] - df['ROX_HR_T0']
    if 'mROX_HR_T1' in df.columns and 'mROX_HR_T0' in df.columns:
        df['mROX_HR_diff'] = df['mROX_HR_T1'] - df['mROX_HR_T0']
    if 'S/F_T1' in df.columns and 'S/F_T0' in df.columns:
        df['S/F_diff'] = df['S/F_T1'] - df['S/F_T0']
    
    # Inverse features
    if 'RR_T0 (bpm)' in df.columns:
        df['1/RR_t0'] = 1 / df['RR_T0 (bpm)']
    if 'RR_T1 (bpm)' in df.columns:
        df['1/RR_t1'] = 1 / df['RR_T1 (bpm)']
    if 'FiO2_T0 (%)' in df.columns:
        df['1/FiO2_t0'] = 1 / df['FiO2_T0 (%)']
    if 'FiO2_T1 (%)' in df.columns:
        df['1/FiO2_t1'] = 1 / df['FiO2_T1 (%)']
    
    return df


# ==================== Main Data Loading Functions ====================

def get_renovate_data_filtered() -> pd.DataFrame:
    """
    Returns the RENOVATE_data_filtered dataset for training.
    This dataset contains hypoxemic patients (Group 1) from the RENOVATE study.
    """
    renovate_data, _, _, _ = load_RENOVATE()
    if not renovate_data.empty:
        print(f"\nRENOVATE filtered - HFNO failure distribution:")
        print(renovate_data['HFNO_failure'].value_counts())
    return renovate_data


def get_hfno_all() -> pd.DataFrame:
    """
    Returns the combined HFNO_all dataset for testing.
    This dataset contains combined data from DNI, non-DNI, MIMIC-IV, and eICU sources.
    """
    print("\n" + "="*60)
    print("BUILDING COMBINED HFNO TEST DATASET")
    print("="*60)
    
    # Load all datasets
    HFNO_DNI = load_DNI()
    HFNO_nonDNI = load_nonDNI()
    filter_MIMIC_HFNC = load_MIMIC_HFNC()
    filtered_HFNC_eicu = load_eICU_HFNC()
    
    # Combine datasets
    dataframes_to_combine = []
    
    # DNI and non-DNI
    if not HFNO_DNI.empty and not HFNO_nonDNI.empty:
        common_columns = HFNO_DNI.columns.intersection(HFNO_nonDNI.columns)
        HFNO_DNI_common = HFNO_DNI[common_columns]
        HFNO_nonDNI_common = HFNO_nonDNI[common_columns]
        HFNO_combined = pd.concat(
            [HFNO_DNI_common, HFNO_nonDNI_common], 
            axis=0, 
            ignore_index=True
        )
        dataframes_to_combine.append(HFNO_combined)
        print(f"\nCombined DNI + non-DNI: {len(HFNO_combined)} samples")
    
    # eICU
    if not filtered_HFNC_eicu.empty:
        dataframes_to_combine.append(filtered_HFNC_eicu)
    
    # MIMIC
    if not filter_MIMIC_HFNC.empty:
        dataframes_to_combine.append(filter_MIMIC_HFNC)
    
    if not dataframes_to_combine:
        print("\nWarning: No valid data sources found!")
        return pd.DataFrame()
    
    # Combine all
    combined_df = pd.concat(dataframes_to_combine, axis=0, ignore_index=True)
    
    # Add additional features
    combined_df = add_additional_features(combined_df)
    
    print(f"\n" + "="*60)
    print(f"FINAL COMBINED DATASET: {len(combined_df)} samples")
    print(f"HFNO failure distribution:")
    print(combined_df['HFNO_failure'].value_counts())
    print("="*60)
    
    return combined_df


def get_features_for_training() -> List[str]:
    """Returns the feature columns to use for TabPFN/TabM training."""
    return [
        'Age (y)', 'HR_T0 (bpm)', 'RR_T0 (bpm)', 
        'FiO2_T1 (%)', 'RR_T1 (bpm)', 'RR_diff', 'S/F_T1'
    ]


def prepare_data_for_tabpfn(
    dataset: pd.DataFrame, 
    target_column: str = 'HFNO_failure'
) -> Tuple[pd.DataFrame, pd.Series]:

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
    
    # Remove rows with NaN values in features or target
    mask = ~(X.isnull().any(axis=1) | y.isnull())
    X = X[mask]
    y = y[mask]
    
    print(f"\nPrepared data: {len(X)} samples with {len(available_features)} features")
    print(f"Target distribution:\n{y.value_counts()}")
    
    return X, y


# ==================== Feature Set Definitions ====================

FEATURES_ALL_NON_ARTERIAL = [
    'Age (y)', 'HR_T0 (bpm)', 'RR_T0 (bpm)', 'FiO2_T0 (%)', 'SpO2_T0 (%)',
    'FiO2_T1 (%)', 'HR_T1 (bpm)', 'RR_T1 (bpm)', 'SpO2_T1 (%)',
    'HR_diff', 'SpO2_diff', 'FiO2_diff', 'RR_diff',
    'S/F_T0', 'S/F_T1', 'S/F_diff'
]


# ==================== Utility Functions ====================

def print_dataset_summary(df: pd.DataFrame, name: str = "Dataset") -> None:
    """Print summary statistics for a dataset."""
    print(f"\n{'='*60}")
    print(f"{name} Summary")
    print(f"{'='*60}")
    print(f"Total samples: {len(df)}")
    print(f"Total features: {len(df.columns)}")
    
    if 'HFNO_failure' in df.columns:
        print(f"\nTarget distribution (HFNO_failure):")
        print(df['HFNO_failure'].value_counts())
        print(f"Failure rate: {df['HFNO_failure'].mean()*100:.2f}%")
    
    missing_counts = df.isnull().sum()
    if missing_counts.sum() > 0:
        print(f"\nMissing values:")
        print(missing_counts[missing_counts > 0].sort_values(ascending=False))
    else:
        print("\nNo missing values!")
    
    print(f"{'='*60}\n")


def get_class_distribution(df: pd.DataFrame, target_col: str = 'HFNO_failure') -> dict:
    """Get class distribution for a dataset."""
    if target_col not in df.columns:
        return {}
    
    counts = df[target_col].value_counts().to_dict()
    total = len(df)
    
    return {
        'counts': counts,
        'percentages': {k: v/total*100 for k, v in counts.items()},
        'imbalance_ratio': max(counts.values()) / min(counts.values()) if counts else 0
    }


def calculate_pos_weight(df: pd.DataFrame, target_col: str = 'HFNO_failure') -> float:
    """
    Calculate pos_weight for imbalanced classification.
    
    Args:
        df: DataFrame with target column
        target_col: Name of target column
    
    Returns:
        pos_weight: ratio of negative to positive samples
    """
    if target_col not in df.columns:
        return 1.0
    
    y = df[target_col].dropna()
    pos = (y == 1).sum()
    neg = (y == 0).sum()
    
    if pos == 0:
        return 1.0
    
    pos_weight = neg / pos
    print(f"\nClass distribution for {target_col}:")
    print(f"  Negative samples: {neg}")
    print(f"  Positive samples: {pos}")
    print(f"  Calculated pos_weight: {pos_weight:.2f}")
    
    return pos_weight

