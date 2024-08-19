from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

# Define default arguments
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
    'feature_engineering',
    default_args=default_args,
    description='Feature engineering for the Breast Cancer dataset',
    schedule_interval='@daily',
)

# Task 1: Load and preprocess data
def load_and_preprocess_data(**kwargs):
    # Load the dataset
    data = load_breast_cancer(as_frame=True)
    df = data['frame']
    
    # Push the dataframe to XCom
    kwargs['ti'].xcom_push(key='breast_cancer_data', value=df)

# Task 2: Feature engineering
def feature_engineering(**kwargs):
    # Pull the data from XCom
    df = kwargs['ti'].xcom_pull(key='breast_cancer_data', task_ids='load_and_preprocess_data')
    
    # Example feature engineering: scaling features
    scaler = StandardScaler()
    df[df.columns[:-1]] = scaler.fit_transform(df[df.columns[:-1]])
    
    # Push the engineered dataframe to XCom
    kwargs['ti'].xcom_push(key='engineered_data', value=df)

# Define tasks
load_and_preprocess_task = PythonOperator(
    task_id='load_and_preprocess_data',
    python_callable=load_and_preprocess_data,
    provide_context=True,
    dag=dag,
)

feature_engineering_task = PythonOperator(
    task_id='feature_engineering',
    python_callable=feature_engineering,
    provide_context=True,
    dag=dag,
)

# Set task dependencies
load_and_preprocess_task >> feature_engineering_task