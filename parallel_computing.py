import time
import zipfile
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from multiprocessing import Pool, cpu_count
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_squared_log_error, \
    explained_variance_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Dictionary to store execution times for different tasks
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

def load_data():
    """Load all required datasets"""
    df = pd.read_csv('/Users/satish/Documents/parallel_final_project/newmimic/ADMISSIONS/admissions_doubled.csv')
    df_pat = pd.read_csv('/Users/satish/Documents/parallel_final_project/newmimic/PATIENTS/patients_doubled.csv')
    df_diagcode = pd.read_csv(
        '/Users/satish/Documents/parallel_final_project/newmimic/DIAGNOSES_ICD/icd_doubled.csv')
    df_icu = pd.read_csv('/Users/satish/Documents/parallel_final_project/newmimic/ICUSTAYS/icustays_doubled.csv')

    # Remove any exact duplicates before processing
    df = df.drop_duplicates()
    df_pat = df_pat.drop_duplicates()
    df_diagcode = df_diagcode.drop_duplicates()
    df_icu = df_icu.drop_duplicates()

    return df, df_pat, df_diagcode, df_icu


def process_chunk(args):
    """Process all features for a chunk of data"""
    chunk_data, chunk_id = args
    chunk = chunk_data.copy()

    # Set random seed based on chunk_id for reproducibility
    np.random.seed(42 + chunk_id)

    # Process LOS
    chunk['ADMITTIME'] = pd.to_datetime(chunk['ADMITTIME'])
    chunk['DISCHTIME'] = pd.to_datetime(chunk['DISCHTIME'])
    chunk['LOS'] = (chunk['DISCHTIME'] - chunk['ADMITTIME']).dt.total_seconds() / 86400
    chunk = chunk[chunk['LOS'] > 0]

    # Rest of the processing remains the same
    chunk['DECEASED'] = chunk['DEATHTIME'].notnull().map({True: 1, False: 0})

    chunk['ETHNICITY'].replace(regex=r'^ASIAN\D*', value='ASIAN', inplace=True)
    chunk['ETHNICITY'].replace(regex=r'^WHITE\D*', value='WHITE', inplace=True)
    chunk['ETHNICITY'].replace(regex=r'^HISPANIC\D*', value='HISPANIC/LATINO', inplace=True)
    chunk['ETHNICITY'].replace(regex=r'^BLACK\D*', value='BLACK/AFRICAN AMERICAN', inplace=True)
    chunk['ETHNICITY'].replace(['UNABLE TO OBTAIN', 'OTHER', 'PATIENT DECLINED TO ANSWER',
                                'UNKNOWN/NOT SPECIFIED'], value='OTHER/UNKNOWN', inplace=True)

    chunk.loc[~chunk['RELIGION'].isin(['NOT SPECIFIED', 'UNOBTAINABLE']), 'RELIGION'] = 'RELIGIOUS'

    chunk['ADMISSION_TYPE'] = chunk['ADMISSION_TYPE'].replace({
        'EMERGENCY': 'EMERGENCY',
        'URGENT': 'EMERGENCY',
        'NEWBORN': 'NEWBORN',
        'ELECTIVE': 'ELECTIVE'
    })

    chunk['INSURANCE'] = chunk['INSURANCE'].replace({
        'Medicare': 'GOVERNMENT',
        'Medicaid': 'GOVERNMENT',
        'Government': 'GOVERNMENT',
        'Self Pay': 'SELF PAY',
        'Private': 'PRIVATE'
    })

    return chunk


def process_icd_parallel(args):
    """Process ICD codes in parallel"""
    df_diagcode_chunk, chunk_id = args
    np.random.seed(42 + chunk_id)  # Set random seed based on chunk_id
    chunk = df_diagcode_chunk.copy()

    icd9_ranges = [(1, 140), (140, 240), (240, 280), (280, 290), (290, 320),
                   (320, 390), (390, 460), (460, 520), (520, 580), (580, 630),
                   (630, 680), (680, 710), (710, 740), (740, 760), (760, 780),
                   (780, 800), (800, 1000), (1000, 2000)]

    diag_dict = {0: 'infectious', 1: 'neoplasms', 2: 'endocrine', 3: 'blood',
                 4: 'mental', 5: 'nervous', 6: 'circulatory', 7: 'respiratory',
                 8: 'digestive', 9: 'genitourinary', 10: 'pregnancy', 11: 'skin',
                 12: 'muscular', 13: 'congenital', 14: 'prenatal', 15: 'misc',
                 16: 'injury', 17: 'misc'}

    chunk['RECODE'] = chunk['ICD9_CODE']
    chunk['RECODE'] = chunk['RECODE'][~chunk['RECODE'].str.contains("[a-zA-Z]").fillna(False)]
    chunk['RECODE'].fillna(value='999', inplace=True)
    chunk['RECODE'] = chunk['RECODE'].str.slice(start=0, stop=3).astype(int)

    for num, cat_range in enumerate(icd9_ranges):
        chunk['RECODE'] = np.where(chunk['RECODE'].between(cat_range[0], cat_range[1]),
                                   num, chunk['RECODE'])

    chunk['CAT'] = chunk['RECODE'].replace(diag_dict)
    return chunk


def merge_and_process_data(df, df_pat, processed_diagcode, df_icu):
    """Merge and process all dataframes"""
    # Ensure deterministic processing
    np.random.seed(42)

    # Process patients data
    df_pat['DOB'] = pd.to_datetime(df_pat['DOB'])
    df_pat = df_pat[['SUBJECT_ID', 'DOB', 'GENDER']]
    df = df.merge(df_pat, on='SUBJECT_ID', how='inner')

    # Process ICD codes with stable ordering
    hadm_list = processed_diagcode.sort_values(['HADM_ID', 'CAT']).groupby('HADM_ID')['CAT'].apply(list).reset_index()
    hadm_item = pd.get_dummies(hadm_list['CAT'].apply(pd.Series).stack()).groupby(level=0).sum()
    hadm_item = hadm_item.join(hadm_list['HADM_ID'], how="outer")
    df = df.merge(hadm_item, how='inner', on='HADM_ID')

    # Process age
    df_age_min = df[['SUBJECT_ID', 'ADMITTIME']].groupby('SUBJECT_ID').min().reset_index()
    df_age_min.columns = ['SUBJECT_ID', 'ADMIT_MIN']
    df = df.merge(df_age_min, how='outer', on='SUBJECT_ID')

    df['ADMIT_MIN'] = pd.to_datetime(df['ADMIT_MIN'], errors='coerce')
    df['DOB'] = pd.to_datetime(df['DOB'], errors='coerce')

    valid_mask = (df['ADMIT_MIN'].notna()) & (df['DOB'].notna()) & (df['ADMIT_MIN'] >= df['DOB'])
    df.loc[valid_mask, 'AGE'] = df.loc[valid_mask, 'ADMIT_MIN'].dt.year - df.loc[valid_mask, 'DOB'].dt.year
    df['AGE'] = np.where(df['AGE'].isna(), 90, df['AGE'])

    # Handle age outliers deterministically
    age_over_200 = df[df['AGE'] > 200]
    if len(age_over_200) > 0:
        np.random.seed(42)  # Ensure reproducible random ages
        random_ages = np.random.randint(0, 101, size=len(age_over_200))
        df.loc[df['AGE'] > 200, 'AGE'] = random_ages

    # Process ICU data
    df_icu['FIRST_CAREUNIT'].replace({'CCU': 'ICU', 'CSRU': 'ICU', 'MICU': 'ICU',
                                      'SICU': 'ICU', 'TSICU': 'ICU'}, inplace=True)
    df_icu['cat'] = df_icu['FIRST_CAREUNIT']
    icu_list = df_icu.sort_values(['HADM_ID', 'cat']).groupby('HADM_ID')['cat'].apply(list).reset_index()
    icu_item = pd.get_dummies(icu_list['cat'].apply(pd.Series).stack()).groupby(level=0).sum()
    icu_item[icu_item >= 1] = 1
    icu_item = icu_item.join(icu_list['HADM_ID'], how="outer")
    df = df.merge(icu_item, how='outer', on='HADM_ID')
    df['ICU'].fillna(value=0, inplace=True)
    df['NICU'].fillna(value=0, inplace=True)

    return df


def preprocess_data(df):
    """Preprocess the final dataset for modeling with deterministic processing"""
    # Set random seed for reproducibility
    np.random.seed(42)

    # Drop unnecessary columns
    df.drop(columns=['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'DISCHTIME', 'ADMISSION_LOCATION',
                     'DISCHARGE_LOCATION', 'LANGUAGE', 'ADMIT_MIN', 'DOB', 'DIAGNOSIS',
                     'DECEASED', 'DEATHTIME', 'EDREGTIME', 'EDOUTTIME',
                     'HOSPITAL_EXPIRE_FLAG', 'HAS_CHARTEVENTS_DATA'], inplace=True, errors='ignore')

    # Encode gender consistently
    df['GENDER'] = df['GENDER'].replace({'M': 0, 'F': 1})

    # Create dummy variables with consistent ordering
    prefix_cols = ['ADM', 'INS', 'REL', 'ETH', 'AGE', 'MAR']
    dummy_cols = ['ADMISSION_TYPE', 'INSURANCE', 'RELIGION', 'ETHNICITY', 'AGE', 'MARITAL_STATUS']

    # Sort categories before creating dummies to ensure consistent ordering
    for col in dummy_cols:
        if col in df.columns:
            df[col] = pd.Categorical(df[col]).codes

    df = pd.get_dummies(df, prefix=prefix_cols, columns=dummy_cols)

    # Handle missing values deterministically
    column_means = df.mean()
    df.fillna(column_means, inplace=True)

    # Prepare target variable
    LOS = df['LOS'].values
    upper_bound = np.percentile(LOS, 95)
    LOS_capped = np.clip(LOS, None, upper_bound)
    LOS_log = np.log1p(LOS_capped)

    # Prepare features
    features = df.drop(columns=['LOS'])
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    return features_scaled, LOS_log


def predict_single(args):
    """Make predictions using a single model"""
    model, X = args
    return model.predict(X)


def evaluate_predictions(y_test, final_preds):
    """Calculate and print evaluation metrics"""
    metrics = {
        'MSE': mean_squared_error(y_test, final_preds),
        'RMSE': np.sqrt(mean_squared_error(y_test, final_preds)),
        'MAE': mean_absolute_error(y_test, final_preds),
        'R²': r2_score(y_test, final_preds),
        'MAPE': np.mean(np.abs((y_test - final_preds) / y_test)) * 100,
        'EVS': explained_variance_score(y_test, final_preds)
    }

    for metric, value in metrics.items():
        print(f'{metric}: {value}')

    return metrics
def train_model_parallel(args):
    """Train a single model in parallel with fixed random state"""
    model_class, X_train, y_train, model_id = args

    if model_class == RandomForestRegressor:
        model = model_class(random_state=42)
    elif model_class == XGBRegressor:
        model = model_class(random_state=42)
    else:
        model = model_class()

    model.fit(X_train, y_train)
    return model


def main():
    total_start_time = time.time()  # Start total execution time
    num_cores = cpu_count()
    print(f"Using {num_cores} CPU cores")

    # Task 1: Data Loading
    df, df_pat, df_diagcode, df_icu = time_task("Data Loading", load_data)

    # Task 2: Data Processing in Parallel
    print("Processing data in parallel...")
    data_processing_start = time.time()
    with Pool(num_cores) as pool:
        # Split main dataframe into chunks and process
        df_chunks = np.array_split(df, num_cores)
        chunk_args = [(chunk, i) for i, chunk in enumerate(df_chunks)]
        processed_chunks = pool.map(process_chunk, chunk_args)
        df = pd.concat(processed_chunks)

        # Process ICD codes in parallel
        diagcode_chunks = np.array_split(df_diagcode, num_cores)
        diagcode_args = [(chunk, i) for i, chunk in enumerate(diagcode_chunks)]
        processed_diagcode_chunks = pool.map(process_icd_parallel, diagcode_args)
        processed_diagcode = pd.concat(processed_diagcode_chunks)
    data_processing_end = time.time()
    data_processing_time = data_processing_end - data_processing_start
    task_times["Data Processing"] = data_processing_time
    print(f"Data Processing completed in {data_processing_time:.2f} seconds")

    # Task 3: Merge and Process Remaining Data
    df = time_task("Merging and Processing Data", merge_and_process_data, df, df_pat, processed_diagcode, df_icu)

    # Task 4: Preprocess Data
    features_scaled, LOS_log = time_task("Preprocessing Data", preprocess_data, df)

    # Task 5: Splitting Data
    print("Splitting data...")
    split_start_time = time.time()
    X_train, X_test, y_train, y_test = train_test_split(
        features_scaled, LOS_log, test_size=0.2, random_state=42
    )
    split_end_time = time.time()
    split_time = split_end_time - split_start_time
    task_times["Splitting Data"] = split_time
    print(f"Data Splitting completed in {split_time:.2f} seconds")

    # Task 6: Model Training in Parallel
    print("Training models in parallel...")
    model_training_start = time.time()
    models_to_train = [
        (LinearRegression, X_train, y_train, 0),
        (RandomForestRegressor, X_train, y_train, 1),
        (XGBRegressor, X_train, y_train, 2)
    ]

    with Pool(min(3, cpu_count())) as pool:
        models = pool.map(train_model_parallel, models_to_train)
    model_training_end = time.time()
    model_training_time = model_training_end - model_training_start
    task_times["Model Training"] = model_training_time
    print(f"Model Training completed in {model_training_time:.2f} seconds")

    # Task 7: Making Predictions
    print("Making predictions...")
    prediction_start_time = time.time()
    predict_args = [(model, X_test) for model in models]
    with Pool(min(len(models), cpu_count())) as pool:
        predictions = pool.map(predict_single, predict_args)
    final_preds = np.mean(predictions, axis=0)
    prediction_end_time = time.time()
    prediction_time = prediction_end_time - prediction_start_time
    task_times["Making Predictions"] = prediction_time
    print(f"Making Predictions completed in {prediction_time:.2f} seconds")

    # Task 8: Evaluation
    print("\nEvaluating predictions...")
    evaluation_start_time = time.time()
    metrics = evaluate_predictions(y_test, final_preds)
    evaluation_end_time = time.time()
    evaluation_time = evaluation_end_time - evaluation_start_time
    task_times["Evaluation"] = evaluation_time
    print(f"Evaluation completed in {evaluation_time:.2f} seconds")

    # Total execution time
    total_end_time = time.time()
    total_execution_time = total_end_time - total_start_time
    print(f'\nTotal execution time: {total_execution_time:.2f} seconds')

    # Add total execution time to task_times
    task_times["Total Execution Time"] = total_execution_time

    # Task Timing Summary
    print("\nTask Timing Summary:")
    for task, duration in task_times.items():
        print(f"{task}: {duration:.2f} seconds")

    # Save task timings to a DataFrame
    timing_df = pd.DataFrame(list(task_times.items()), columns=['Task', 'Time (s)'])

    # Save the DataFrame to a CSV file
    timing_df.to_csv("parallel_task_timing_summary.csv", index=False)
    print("\nTask timings saved to 'parallel_task_timing_summary.csv'.")


if __name__ == "__main__":
    main()