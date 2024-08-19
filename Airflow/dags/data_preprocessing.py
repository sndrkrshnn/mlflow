from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Define the default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
}

# Define the DAG
dag = DAG(
    'data_preprocessing',
    default_args=default_args,
    description='A simple ML pipeline for preprocessing the breast cancer dataset',
    schedule_interval='@daily',
)

# Task 1: Download the dataset
def download_data(**kwargs):
    # Load the dataset
    data = load_breast_cancer(as_frame=True)
    df = data['frame']
    
    # Push the dataframe to XCom
    kwargs['ti'].xcom_push(key='breast_cancer_data', value=df)

# Task 2: Perform initial data cleaning
def clean_data(**kwargs):
    # Pull the data from XCom
    df = kwargs['ti'].xcom_pull(key='breast_cancer_data', task_ids='download_data')
    
    # Check for missing values and drop them
    df_cleaned = df.dropna()
    
    # Push the cleaned dataframe to XCom
    kwargs['ti'].xcom_push(key='cleaned_data', value=df_cleaned)

# Task 3: Preprocess the data
def preprocess_data(**kwargs):
    # Pull the cleaned data from XCom
    df_cleaned = kwargs['ti'].xcom_pull(key='cleaned_data', task_ids='clean_data')
    
    # Split the dataset into features and target
    X = df_cleaned.drop('target', axis=1)
    y = df_cleaned['target']
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Store the preprocessed data
    processed_data = {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test
    }
    
    # Push the preprocessed data to XCom
    kwargs['ti'].xcom_push(key='processed_data', value=processed_data)

# Define tasks
download_task = PythonOperator(
    task_id='download_data',
    python_callable=download_data,
    provide_context=True,
    dag=dag,
)

clean_task = PythonOperator(
    task_id='clean_data',
    python_callable=clean_data,
    provide_context=True,
    dag=dag,
)

preprocess_task = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data,
    provide_context=True,
    dag=dag,
)

# Set task dependencies
download_task >> clean_task >> preprocess_task