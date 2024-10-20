import time
import pandas as pd
import numpy as np
from multiprocessing import Pool, Process
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score


# Load datasets
def load_data():
    df = pd.read_csv('/home/jovyan/parallel_final_project/mimicIII/ADMISSIONS/ADMISSIONS_sorted.csv')
    df_pat = pd.read_csv('/home/jovyan/parallel_final_project/mimicIII/PATIENTS/PATIENTS_sorted.csv')
    df_diagcode = pd.read_csv('/home/jovyan/parallel_final_project/mimicIII/DIAGNOSES_ICD/DIAGNOSES_ICD_sorted.csv')
    df_icu = pd.read_csv('/home/jovyan/parallel_final_project/mimicIII/ICUSTAYS/ICUSTAYS_sorted.csv')
    return df, df_pat, df_diagcode, df_icu

# Data Parallelism for feature engineering in chunks
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
    return pd.concat(df_chunks, axis=0)

# Task Parallelism for ethnicity, religion, admissions
def process_ethnicity(df):
    df['ETHNICITY'].replace(regex=r'^ASIAN\D*', value='ASIAN', inplace=True)
    df['ETHNICITY'].replace(regex=r'^WHITE\D*', value='WHITE', inplace=True)
    df['ETHNICITY'].replace(regex=r'^HISPANIC\D*', value='HISPANIC/LATINO', inplace=True)
    df['ETHNICITY'].replace(regex=r'^BLACK\D*', value='BLACK/AFRICAN AMERICAN', inplace=True)
    df['ETHNICITY'].replace(['UNABLE TO OBTAIN', 'OTHER', 'PATIENT DECLINED TO ANSWER', 'UNKNOWN/NOT SPECIFIED'], value='OTHER/UNKNOWN', inplace=True)
    return df

def process_religion(df):
    df.loc[~df['RELIGION'].isin(['NOT SPECIFIED', 'UNOBTAINABLE']), 'RELIGION'] = 'RELIGIOUS'
    return df

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

# Parallel feature engineering task execution
def task_parallel_feature_engineering(df):
    p1 = Process(target=process_ethnicity, args=(df,))
    p2 = Process(target=process_religion, args=(df,))
    p3 = Process(target=process_admissions, args=(df,))
    
    p1.start()
    p2.start()
    p3.start()
    
    p1.join()
    p2.join()
    p3.join()
    
    return df

# Process death column
def process_death(df):
    df['DECEASED'] = df['DEATHTIME'].notnull().map({True: 1, False: 0})
    return df

# Process ICD Codes
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
    hadm_item = pd.get_dummies(hadm_list['CAT'].apply(pd.Series).stack()).sum(level=0)
    hadm_item = hadm_item.join(hadm_list['HADM_ID'], how="outer")
    df = df.merge(hadm_item, how='inner', on='HADM_ID')
    return df

# Process Age
def process_age(df):
    df_age_min = df[['SUBJECT_ID', 'ADMITTIME']].groupby('SUBJECT_ID').min().reset_index()
    df_age_min.columns = ['SUBJECT_ID', 'ADMIT_MIN']
    df = df.merge(df_age_min, how='outer', on='SUBJECT_ID')
    df['ADMIT_MIN'] = pd.to_datetime(df['ADMIT_MIN'], errors='coerce')
    df['DOB'] = pd.to_datetime(df['DOB'], errors='coerce')
    valid_mask = (df['ADMIT_MIN'].notna()) & (df['DOB'].notna()) & (df['ADMIT_MIN'] >= df['DOB'])
    df.loc[valid_mask, 'AGE'] = df.loc[valid_mask, 'ADMIT_MIN'].dt.year - df.loc[valid_mask, 'DOB'].dt.year
    df['AGE'] = np.where(df['AGE'].isna(), 90, df['AGE'])
    age_over_200 = df[df['AGE'] > 200]
    random_ages = np.random.randint(0, 101, size=len(age_over_200))
    df.loc[df['AGE'] > 200, 'AGE'] = random_ages
    return df

# Process ICU Data
def process_icu(df, df_icu):
    df_icu['FIRST_CAREUNIT'].replace({'CCU': 'ICU', 'CSRU': 'ICU', 'MICU': 'ICU', 'SICU': 'ICU', 'TSICU': 'ICU'}, inplace=True)
    df_icu['cat'] = df_icu['FIRST_CAREUNIT']
    icu_list = df_icu.groupby('HADM_ID')['cat'].apply(list).reset_index()
    icu_item = pd.get_dummies(icu_list['cat'].apply(pd.Series).stack()).groupby(level=0).sum()
    icu_item = icu_item.join(icu_list['HADM_ID'], how="outer")
    df = df.merge(icu_item, how='outer', on='HADM_ID')
    df['ICU'].fillna(value=0, inplace=True)
    df['NICU'].fillna(value=0, inplace=True)
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

# Model Parallelism
def train_lr(X_train, y_train):
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    return lr_model

def train_rf(X_train, y_train):
    rf_model = RandomForestRegressor()
    rf_model.fit(X_train, y_train)
    return rf_model

def train_xgb(X_train, y_train):
    xgb_model = XGBRegressor()
    xgb_model.fit(X_train, y_train)
    return xgb_model

def parallel_model_training(X_train, y_train):
    with Pool(3) as pool:
        models = pool.starmap(train_model, [(X_train, y_train) for train_model in [train_lr, train_rf, train_xgb]])
    return models

def parallel_predictions(models, X_test):
    with Pool(len(models)) as pool:
        predictions = pool.map(lambda model: model.predict(X_test), models)
    final_preds = np.mean(predictions, axis=0)
    return final_preds

# Evaluation Metrics
def evaluation_metrics(y_test, final_preds):
    mse = mean_squared_error(y_test, final_preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, final_preds)
    r2 = r2_score(y_test, final_preds)
    evs = explained_variance_score(y_test, final_preds)
    return mse, rmse, mae, r2, evs

def print_evaluation(mse, rmse, mae, r2, evs):
    print(f'MSE: {mse}, RMSE: {rmse}, MAE: {mae}, R2: {r2}, EVS: {evs}')

# Main function to execute the pipeline
def main():
    start_time = time.time()
    
    # Load Data
    df, df_pat, df_diagcode, df_icu = load_data()

    # Data Parallelism: LOS processing
    df = parallel_process_los(df)

    # Task Parallelism: Feature engineering
    df = task_parallel_feature_engineering(df)

    # Process more features (death, ICD, age, ICU)
    df = process_death(df)
    df = process_icd(df, df_diagcode)
    df = process_age(df)
    df = process_icu(df, df_icu)

    # Preprocess Data
    features_scaled, LOS_log = preprocess_data(df)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, LOS_log, test_size=0.2, random_state=42)

    # Model Parallelism
    models = parallel_model_training(X_train, y_train)

    # Parallel Predictions
    final_preds = parallel_predictions(models, X_test)

    # Evaluate the model
    mse, rmse, mae, r2, evs = evaluation_metrics(y_test, final_preds)
    print_evaluation(mse, rmse, mae, r2, evs)

    # Execution time
    print(f'Total execution time: {time.time() - start_time:.2f} seconds')

if __name__ == "__main__":
    main()
