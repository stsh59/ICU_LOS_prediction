#!/usr/bin/env python
# coding: utf-8
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["XGBOOST_NUM_THREADS"] = "1"

import time
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_squared_log_error,explained_variance_score

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Task timing dictionary
task_times = {}

def time_task(task_name, func, *args, **kwargs):
    """
    Utility to measure and record the execution time of a task.
    """
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    task_times[task_name] = end_time - start_time
    print(f"{task_name} completed in {task_times[task_name]:.2f} seconds")
    return result

# Function to load datasets
def load_data():
    # Primary Admissions information
    df = pd.read_csv('/Users/satish/Documents/parallel_final_project/newmimic/ADMISSIONS/admissions_doubled.csv')
    # Patient specific info such as gender
    df_pat = pd.read_csv('/Users/satish/Documents/parallel_final_project/newmimic/PATIENTS/patients_doubled.csv')
    # Diagnosis for each admission to hospital
    df_diagcode = pd.read_csv('/Users/satish/Documents/parallel_final_project/newmimic/DIAGNOSES_ICD/icd_doubled.csv')
    # Intensive Care Unit (ICU) for each admission to hospital
    df_icu = pd.read_csv('/Users/satish/Documents/parallel_final_project/newmimic/ICUSTAYS/icustays_doubled.csv')
    return df, df_pat, df_diagcode, df_icu

# Feature engineering for Length of Stay (LOS)
def process_los(df):
    df['ADMITTIME'] = pd.to_datetime(df['ADMITTIME'])
    df['DISCHTIME'] = pd.to_datetime(df['DISCHTIME'])
    df['LOS'] = (df['DISCHTIME'] - df['ADMITTIME']).dt.total_seconds()/86400   
    
    # Drop rows with negative LOS
    df['LOS'][df['LOS'] > 0].describe()
    df = df[df['LOS'] > 0]  # Remove invalid LOS
    
    # Pre-emptively drop some columns that are no longer needed
    df = df.copy()
    #df.drop(columns=['DISCHTIME', 'ROW_ID', 'EDREGTIME', 'EDOUTTIME',
    #                 'HOSPITAL_EXPIRE_FLAG', 'HAS_CHARTEVENTS_DATA'], inplace=True)
    df.drop(columns=['DISCHTIME', 'EDREGTIME', 'EDOUTTIME',
                     'HOSPITAL_EXPIRE_FLAG'], inplace=True)
    return df

# Feature Engineering for Deceased Column
def process_death(df):
    df['DECEASED'] = df['DEATHTIME'].notnull().map({True: 1, False: 0})
    return df

# Ethnicity Column Reduction
def process_ethnicity(df):
    # Directly assign the replacements back to df['ETHNICITY']
    df['ETHNICITY'] = df['ETHNICITY'].replace(regex=r'^ASIAN\D*', value='ASIAN')
    df['ETHNICITY'] = df['ETHNICITY'].replace(regex=r'^WHITE\D*', value='WHITE')
    df['ETHNICITY'] = df['ETHNICITY'].replace(regex=r'^HISPANIC\D*', value='HISPANIC/LATINO')
    df['ETHNICITY'] = df['ETHNICITY'].replace(regex=r'^BLACK\D*', value='BLACK/AFRICAN AMERICAN')
    df['ETHNICITY'] = df['ETHNICITY'].replace(
        ['UNABLE TO OBTAIN', 'OTHER', 'PATIENT DECLINED TO ANSWER', 'UNKNOWN/NOT SPECIFIED'],
        value='OTHER/UNKNOWN'
    )

    # Further compress the categories by grouping the remaining into 'OTHER/UNKNOWN'
    ethnicity_counts = df['ETHNICITY'].value_counts().nlargest(5).index.tolist()
    df.loc[~df['ETHNICITY'].isin(ethnicity_counts), 'ETHNICITY'] = 'OTHER/UNKNOWN'

    return df

# Religion Column Reduction
def process_religion(df):
    df.loc[~df['RELIGION'].isin(['NOT SPECIFIED', 'UNOBTAINABLE']), 'RELIGION'] = 'RELIGIOUS'
    return df

# Function to process admission type and insurance
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

# Function to preprocess patients data and merge with admissions
def process_patients(df, df_pat):
    df_pat['DOB'] = pd.to_datetime(df_pat['DOB'])
    df_pat = df_pat[['SUBJECT_ID', 'DOB', 'GENDER']]
    df = df.merge(df_pat, on='SUBJECT_ID', how='inner')
    return df

# Function to process ICD codes
def process_icd(df, df_diagcode):
    icd9_ranges = [(1, 140), (140, 240), (240, 280), (280, 290), (290, 320), (320, 390), 
                   (390, 460), (460, 520), (520, 580), (580, 630), (630, 680), (680, 710),
                   (710, 740), (740, 760), (760, 780), (780, 800), (800, 1000), (1000, 2000)]

    diag_dict = {0: 'infectious', 1: 'neoplasms', 2: 'endocrine', 3: 'blood',
                 4: 'mental', 5: 'nervous', 6: 'circulatory', 7: 'respiratory',
                 8: 'digestive', 9: 'genitourinary', 10: 'pregnancy', 11: 'skin', 
                 12: 'muscular', 13: 'congenital', 14: 'prenatal', 15: 'misc',
                 16: 'injury', 17: 'misc'}

    df_diagcode['RECODE'] = df_diagcode['ICD9_CODE']
    df_diagcode['RECODE'] = df_diagcode['RECODE'][~df_diagcode['RECODE'].str.contains("[a-zA-Z]").fillna(False)]
    df_diagcode['RECODE'].fillna(value='999', inplace=True)

    # Convert to integers
    df_diagcode['RECODE'] = df_diagcode['RECODE'].str.slice(start=0, stop=3, step=1)
    df_diagcode['RECODE'] = df_diagcode['RECODE'].astype(int)
    
    for num, cat_range in enumerate(icd9_ranges):
        df_diagcode['RECODE'] = np.where(df_diagcode['RECODE'].between(cat_range[0], cat_range[1]), 
                                         num, df_diagcode['RECODE'])
    
    df_diagcode['CAT'] = df_diagcode['RECODE'].replace(diag_dict)
    
    # Create list of diagnoses for each admission
    hadm_list = df_diagcode.groupby('HADM_ID')['CAT'].apply(list).reset_index()
    
    # Convert diagnoses list into hospital admission-item matrix
    #hadm_item = pd.get_dummies(hadm_list['CAT'].apply(pd.Series).stack()).sum(level=0)
    hadm_item = pd.get_dummies(hadm_list['CAT'].apply(pd.Series).stack()).groupby(level=0).sum()

    # Join back with HADM_ID, will merge with main admissions DF later
    hadm_item = hadm_item.join(hadm_list['HADM_ID'], how="outer")
    
    # Merge with main admissions df
    df = df.merge(hadm_item, how='inner', on='HADM_ID')
    
    return df

# Age Feature Engineering
def process_age(df):
    # Find the first admission time for each patient
    df_age_min = df[['SUBJECT_ID', 'ADMITTIME']].groupby('SUBJECT_ID').min().reset_index()
    df_age_min.columns = ['SUBJECT_ID', 'ADMIT_MIN']
    df = df.merge(df_age_min, how='outer', on='SUBJECT_ID')
    
    # Convert 'ADMIT_MIN' and 'DOB' to datetime, coercing invalid entries to NaT
    df['ADMIT_MIN'] = pd.to_datetime(df['ADMIT_MIN'], errors='coerce')
    df['DOB'] = pd.to_datetime(df['DOB'], errors='coerce')
    
    # Ensure valid rows: both ADMIT_MIN and DOB should be valid dates and ADMIT_MIN >= DOB
    valid_mask = (df['ADMIT_MIN'].notna()) & (df['DOB'].notna()) & (df['ADMIT_MIN'] >= df['DOB'])

    # Safely calculate age only for valid rows by subtracting years directly
    df.loc[valid_mask, 'AGE'] = df.loc[valid_mask, 'ADMIT_MIN'].dt.year - df.loc[valid_mask, 'DOB'].dt.year

    # Replace any NaN age values with 90 (as a default) if required
    df['AGE'] = np.where(df['AGE'].isna(), 90, df['AGE'])
    
    # Check how many records have AGE > 200
    age_over_200 = df[df['AGE'] > 200]

    # Generate random values between 0 and 100 for records where AGE > 200
    random_ages = np.random.randint(0, 101, size=len(age_over_200))

    # Assign these random values to the AGE column for the respective records
    df.loc[df['AGE'] > 200, 'AGE'] = random_ages
    return df


# ICU Feature Engineering (2.4.1)
def process_icu(df, df_icu):
    # Reduce ICU categories
    df_icu['FIRST_CAREUNIT'].replace({'CCU': 'ICU', 'CSRU': 'ICU', 'MICU': 'ICU',
                                      'SICU': 'ICU', 'TSICU': 'ICU'}, inplace=True)

    df_icu['cat'] = df_icu['FIRST_CAREUNIT']
    icu_list = df_icu.groupby('HADM_ID')['cat'].apply(list).reset_index()
    
    # Create admission-ICU matrix
    icu_item = pd.get_dummies(icu_list['cat'].apply(pd.Series).stack()).groupby(level=0).sum()
    icu_item[icu_item >= 1] = 1
    icu_item = icu_item.join(icu_list['HADM_ID'], how="outer")
    
    # Merge ICU data with main dataframe
    df = df.merge(icu_item, how='outer', on='HADM_ID')
    
    # Replace NaNs with 0
    df['ICU'].fillna(value=0, inplace=True)
    df['NICU'].fillna(value=0, inplace=True)
    
    return df

# Function to preprocess features and target
def preprocess_data(df):
    # Drop unused or no longer needed columns
    df.drop(columns=['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'ADMISSION_LOCATION',
                     'DISCHARGE_LOCATION', 'LANGUAGE', 'ADMIT_MIN', 'DOB',
                     'DIAGNOSIS', 'DECEASED', 'DEATHTIME'], inplace=True)

    # Convert GENDER to numeric (0 for Male, 1 for Female)
    df['GENDER'] = df['GENDER'].replace({'M': 0, 'F': 1})
    
    # Create dummy columns for categorical variables
    prefix_cols = ['ADM', 'INS', 'REL', 'ETH', 'AGE', 'MAR', 'RELIGION']
    dummy_cols = ['ADMISSION_TYPE', 'INSURANCE', 'RELIGION',
                  'ETHNICITY', 'AGE', 'MARITAL_STATUS', 'RELIGION']
    df = pd.get_dummies(df, prefix=prefix_cols, columns=dummy_cols)

    # Creating a copy for final processing
    df_final = df.copy()
    
    # Handling missing values - replace NaNs in numeric columns with mean
    df_final.fillna(df_final.mean(), inplace=True)
    
    # Optionally, fill categorical NaNs with the most frequent value
    for col in df_final.select_dtypes(include=['object']).columns:
        df_final[col].fillna(df_final[col].mode()[0], inplace=True)
    
    # Target Variable (Length-of-Stay)
    LOS = df_final['LOS'].values
    # Prediction Features
    features = df_final.drop(columns=['LOS'])
    
    # Step 1: Handling Outliers (Capping outliers at 95th percentile)
    upper_bound = np.percentile(LOS, 95)
    LOS_capped = np.clip(LOS, None, upper_bound)  # Capping LOS at 95th percentile

    # Step 2: Log Transformation (Optional but recommended for long-tailed distributions)
    LOS_log = np.log1p(LOS_capped)  # Use log1p to avoid log(0)

    # Step 3: Feature Scaling using StandardScaler
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    return features_scaled, LOS_log


# Model training and evaluation
def train_models(X_train, X_test, y_train, y_test):
    # Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_preds = lr_model.predict(X_test)

    # Random Forest
    rf_model = RandomForestRegressor(n_jobs=1)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)

    # XGBoost
    xgb_model = XGBRegressor(n_jobs=1)
    xgb_model.fit(X_train, y_train)
    xgb_preds = xgb_model.predict(X_test)

    # Ensemble (average of all models)
    final_preds = (lr_preds + rf_preds + xgb_preds) / 3
    
    return final_preds


def evaluation_metrics(y_test, final_preds):
    mse = mean_squared_error(y_test, final_preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, final_preds)
    r2 = r2_score(y_test, final_preds)
    mape = np.mean(np.abs((y_test - final_preds) / y_test)) * 100
    evs = explained_variance_score(y_test, final_preds)
    msle = mean_squared_log_error(y_test, final_preds)
    
    return mse, rmse, mae, r2, mape, evs, msle


def print_evaluation(mse, rmse, mae, r2, mape, evs, msle):
    print(f'Mean Squared Error (MSE): {mse}')
    print(f'Root Mean Squared Error (RMSE): {rmse}')
    print(f'Mean Absolute Error (MAE): {mae}')
    print(f'R-squared (R²): {r2}')
    print(f'Mean Absolute Percentage Error (MAPE): {mape}%')
    print(f'Explained Variance Score (EVS): {evs}')
    print(f'Mean Squared Logarithmic Error (MSLE): {msle}')

# Main function to run the pipeline
# Main function to run the pipeline
def main():
    start_time = time.time()

    # Task 1: Data Loading and Processing
    print("Starting Data Processing...")
    df, df_pat, df_diagcode, df_icu = time_task("Load Data", load_data)
    df = time_task("Process Length of Stay", process_los, df)
    df = time_task("Process Death", process_death, df)
    df = time_task("Process Ethnicity", process_ethnicity, df)
    df = time_task("Process Religion", process_religion, df)
    df = time_task("Process Admissions", process_admissions, df)
    df = time_task("Process ICD Codes", process_icd, df, df_diagcode)
    df = time_task("Process Patients", process_patients, df, df_pat)
    df = time_task("Process Age", process_age, df)
    df = time_task("Process ICU", process_icu, df, df_icu)
    features_scaled, LOS_log = time_task("Preprocess Data", preprocess_data, df)

    # Task 2: Model Training
    print("\nStarting Model Training...")
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, LOS_log, test_size=0.2, random_state=42)
    final_preds = time_task("Model Training", train_models, X_train, X_test, y_train, y_test)

    # Task 3: Prediction and Evaluation
    print("\nStarting Prediction and Evaluation...")
    mse, rmse, mae, r2, mape, evs, msle = time_task("Evaluation Metrics", evaluation_metrics, y_test, final_preds)
    print_evaluation(mse, rmse, mae, r2, mape, evs, msle)

    # Total pipeline execution time
    total_time = time.time() - start_time
    print(f"\nTotal Execution Time: {total_time:.2f} seconds")

    # Task timing summary
    print("\nTask Timing Summary:")
    for task, duration in task_times.items():
        print(f"{task}: {duration:.2f} seconds")

    # Save task timings to a DataFrame
    timing_df = pd.DataFrame(list(task_times.items()), columns=['Task', 'Time (s)'])

    # Save the DataFrame to a CSV file
    timing_df.to_csv("serial_timing_summary.csv", index=False)
    print("\nTask timings saved to 'serial_timing_summary.csv'.")


# Entry point
if __name__ == "__main__":
    main()