import time
import pandas as pd
import numpy as np
from multiprocessing import Pool, Manager
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score

# Helper to set random seed for consistency
def set_random_seed(seed):
    np.random.seed(seed)

# Load datasets
def load_data():
    df = pd.read_csv('/Users/satish/Documents/parallel_final_project/mimicIII/ADMISSIONS/ADMISSIONS_sorted.csv')
    df_pat = pd.read_csv('/Users/satish/Documents/parallel_final_project/mimicIII/PATIENTS/PATIENTS_sorted.csv')
    df_diagcode = pd.read_csv('/Users/satish/Documents/parallel_final_project/mimicIII/DIAGNOSES_ICD/DIAGNOSES_ICD_sorted.csv')
    df_icu = pd.read_csv('/Users/satish/Documents/parallel_final_project/mimicIII/ICUSTAYS/ICUSTAYS_sorted.csv')
    return df, df_pat, df_diagcode, df_icu

# Data Parallelism for processing LOS in chunks
def process_los_chunk(chunk):
    chunk['ADMITTIME'] = pd.to_datetime(chunk['ADMITTIME'])
    chunk['DISCHTIME'] = pd.to_datetime(chunk['DISCHTIME'])
    chunk['LOS'] = (chunk['DISCHTIME'] - chunk['ADMITTIME']).dt.total_seconds() / 86400
    chunk = chunk[chunk['LOS'] > 0]
    chunk.drop(columns=['DISCHTIME', 'ROW_ID', 'EDREGTIME', 'EDOUTTIME', 'HOSPITAL_EXPIRE_FLAG', 'HAS_CHARTEVENTS_DATA'], inplace=True)
    return chunk

def parallel_process_los(df, num_workers=4):
    chunks = np.array_split(df, num_workers)
    with Pool(num_workers) as pool:
        df_chunks = pool.map(process_los_chunk, chunks)
    result_df = pd.concat(df_chunks, axis=0).reset_index(drop=True)
    return result_df

# Process ethnicity
def process_ethnicity(df):
    df['ETHNICITY'].replace(regex=r'^ASIAN\D*', value='ASIAN', inplace=True)
    df['ETHNICITY'].replace(regex=r'^WHITE\D*', value='WHITE', inplace=True)
    df['ETHNICITY'].replace(regex=r'^HISPANIC\D*', value='HISPANIC/LATINO', inplace=True)
    df['ETHNICITY'].replace(regex=r'^BLACK\D*', value='BLACK/AFRICAN AMERICAN', inplace=True)
    df['ETHNICITY'].replace(['UNABLE TO OBTAIN', 'OTHER', 'PATIENT DECLINED TO ANSWER', 'UNKNOWN/NOT SPECIFIED'], value='OTHER/UNKNOWN', inplace=True)
    return df

# Process religion
def process_religion(df):
    df.loc[~df['RELIGION'].isin(['NOT SPECIFIED', 'UNOBTAINABLE']), 'RELIGION'] = 'RELIGIOUS'
    return df

# Process admissions
def process_admissions(df):
    df['ADMISSION_TYPE'] = df['ADMISSION_TYPE'].replace({
        'EMERGENCY': 'EMERGENCY',
        'URGENT': 'EMERGENCY',
        'NEWBORN': 'NEWBORN',
        'ELECTIVE': 'ELECTIVE'
    })
    df['INSURANCE'] = df['INSURANCE'].replace({
        'Medicare': 'GOVERNMENT',
        'Medicaid': 'GOVERNMENT',
        'Government': 'GOVERNMENT',
        'Self Pay': 'SELF PAY',
        'Private': 'PRIVATE'
    })
    return df

# Task Parallelism for feature engineering
def task_parallel_feature_engineering(df):
    with Pool(3) as pool:
        dfs = pool.starmap(
            lambda func, data: func(data),
            [(process_ethnicity, df.copy()), (process_religion, df.copy()), (process_admissions, df.copy())]
        )
    # Merge results as needed (this is a simplified example, adjust for real merging needs)
    df_ethnicity, df_religion, df_admissions = dfs
    df_combined = pd.concat([df_ethnicity, df_religion, df_admissions], axis=1)
    return df_combined

# Process death
def process_death(df):
    df['DECEASED'] = df['DEATHTIME'].notnull().map({True: 1, False: 0})
    return df

# Process patients
def process_patients(df, df_pat):
    df_pat['DOB'] = pd.to_datetime(df_pat['DOB'])
    df_pat = df_pat[['SUBJECT_ID', 'DOB', 'GENDER']]
    df = df.merge(df_pat, on='SUBJECT_ID', how='inner')
    return df

# Process ICD codes
def process_icd(df, df_diagcode):
    icd9_ranges = [(1, 140), (140, 240), (240, 280), (280, 290), (290, 320),
                   (320, 390), (390, 460), (460, 520), (520, 580), (580, 630),
                   (630, 680), (680, 710), (710, 740), (740, 760), (760, 780),
                   (780, 800), (800, 1000), (1000, 2000)]
    diag_dict = {0: 'infectious', 1: 'neoplasms', 2: 'endocrine', 3: 'blood',
                 4: 'mental', 5: 'nervous', 6: 'circulatory', 7: 'respiratory',
                 8: 'digestive', 9: 'genitourinary', 10: 'pregnancy', 11: 'skin',
                 12: 'muscular', 13: 'congenital', 14: 'prenatal', 15: 'misc',
                 16: 'injury', 17: 'misc'}
    df_diagcode['RECODE'] = df_diagcode['ICD9_CODE']
    df_diagcode['RECODE'] = df_diagcode['RECODE'][~df_diagcode['RECODE'].str.contains("[a-zA-Z]").fillna(False)]
    df_diagcode['RECODE'].fillna(value='999', inplace=True)
    df_diagcode['RECODE'] = df_diagcode['RECODE'].str.slice(start=0, stop=3).astype(int)

    for num, cat_range in enumerate(icd9_ranges):
        df_diagcode['RECODE'] = np.where(df_diagcode['RECODE'].between(cat_range[0], cat_range[1]), num, df_diagcode['RECODE'])

    df_diagcode['CAT'] = df_diagcode['RECODE'].replace(diag_dict)
    hadm_list = df_diagcode.groupby('HADM_ID')['CAT'].apply(list).reset_index()
    #hadm_item = pd.get_dummies(hadm_list['CAT'].apply(pd.Series).stack()).sum(level=0)
    hadm_item = pd.get_dummies(hadm_list['CAT'].apply(pd.Series).stack()).groupby(level=0).sum()
    hadm_item = hadm_item.join(hadm_list['HADM_ID'], how="outer")
    df = df.merge(hadm_item, how='inner', on='HADM_ID')
    return df

# Process Age
def process_age(df):
    set_random_seed(42)
    df_age_min = df[['SUBJECT_ID', 'ADMITTIME']].groupby('SUBJECT_ID').min().reset_index()
    df_age_min.columns = ['SUBJECT_ID', 'ADMIT_MIN']
    df = df.merge(df_age_min, how='outer', on='SUBJECT_ID')
    df['ADMIT_MIN'] = pd.to_datetime(df['ADMIT_MIN'], errors='coerce')
    df['DOB'] = pd.to_datetime(df['DOB'], errors='coerce')
    valid_mask = (df['ADMIT_MIN'].notna()) & (df['DOB'].notna()) & (df['ADMIT_MIN'] >= df['DOB'])
    df.loc[valid_mask, 'AGE'] = df.loc[valid_mask, 'ADMIT_MIN'].dt.year - df.loc[valid_mask, 'DOB'].dt.year
    df['AGE'] = np.where(df['AGE'].isna(), 90, df['AGE'])
    df.loc[df['AGE'] > 200, 'AGE'] = np.random.randint(0, 101, size=len(df[df['AGE'] > 200]))
    return df

# Preprocess Data
def preprocess_data(df):
    df.drop(columns=['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'ADMISSION_LOCATION', 'DISCHARGE_LOCATION', 'LANGUAGE', 'ADMIT_MIN', 'DOB', 'DIAGNOSIS', 'DECEASED', 'DEATHTIME'], inplace=True)
    df['GENDER'] = df['GENDER'].replace({'M': 0, 'F': 1})
    df = pd.get_dummies(df)
    df.fillna(df.mean(), inplace=True)
    LOS = df['LOS'].values
    features = df.drop(columns=['LOS'])
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    LOS_log = np.log1p(LOS)
    return features_scaled, LOS_log

# Model Training Parallelism
def train_model(func, X_train, y_train):
    return func(X_train, y_train)

def parallel_model_training(X_train, y_train):
    with Pool(3) as pool:
        models = pool.starmap(
            train_model,
            [(LinearRegression().fit, X_train, y_train), (RandomForestRegressor().fit, X_train, y_train), (XGBRegressor().fit, X_train, y_train)]
        )
    return models

# Predictions
def parallel_predictions(models, X_test):
    with Pool(len(models)) as pool:
        predictions = pool.map(lambda model: model.predict(X_test), models)
    final_preds = np.mean(predictions, axis=0)
    return final_preds

# Main function
def main():
    start_time = time.time()
    df, df_pat, df_diagcode, df_icu = load_data()
    df = process_patients(df, df_pat)
    df = parallel_process_los(df)
    df = task_parallel_feature_engineering(df)
    df = process_death(df)
    df = process_icd(df, df_diagcode)
    df = process_age(df)
    df = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(*df, test_size=0.2, random_state=42)
    models = parallel_model_training(X_train, y_train)
    final_preds = parallel_predictions(models, X_test)
    print(f"Metrics: {mean_squared_error(y_test, final_preds)}")
    print(f"Execution Time: {time.time() - start_time}")

if __name__ == "__main__":
    main()
