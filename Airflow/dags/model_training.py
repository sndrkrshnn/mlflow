from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

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
    'model_training',
    default_args=default_args,
    description='Model training and evaluation for the Breast Cancer dataset',
    schedule_interval='@daily',
)

# Task 1: Load engineered data
def load_engineered_data(**kwargs):
    # Pull the engineered data from the previous DAG using XCom
    df = kwargs['ti'].xcom_pull(dag_id='feature_engineering', key='engineered_data', task_ids='feature_engineering')
    
    # Split the dataset into features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Push the splits to XCom
    kwargs['ti'].xcom_push(key='X_train', value=X_train)
    kwargs['ti'].xcom_push(key='X_test', value=X_test)
    kwargs['ti'].xcom_push(key='y_train', value=y_train)
    kwargs['ti'].xcom_push(key='y_test', value=y_test)

# Task 2: Model training
def model_training(**kwargs):
    # Pull the training data from XCom
    X_train = kwargs['ti'].xcom_pull(key='X_train', task_ids='load_engineered_data')
    y_train = kwargs['ti'].xcom_pull(key='y_train', task_ids='load_engineered_data')
    
    # Train an SVM model
    model = SVC(kernel='linear', C=1.0, random_state=42)
    model.fit(X_train, y_train)
    
    # Push the trained model to XCom
    kwargs['ti'].xcom_push(key='trained_model', value=model)

# Task 3: Model evaluation
def model_evaluation(**kwargs):
    # Pull the testing data and trained model from XCom
    X_test = kwargs['ti'].xcom_pull(key='X_test', task_ids='load_engineered_data')
    y_test = kwargs['ti'].xcom_pull(key='y_test', task_ids='load_engineered_data')
    model = kwargs['ti'].xcom_pull(key='trained_model', task_ids='model_training')
    
    # Predict on the test set
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    
    # Print the classification report and accuracy
    print(f"Accuracy: {accuracy}")
    print(f"Classification Report:\n{class_report}")

# Define tasks
load_engineered_data_task = PythonOperator(
    task_id='load_engineered_data',
    python_callable=load_engineered_data,
    provide_context=True,
    dag=dag,
)

model_training_task = PythonOperator(
    task_id='model_training',
    python_callable=model_training,
    provide_context=True,
    dag=dag,
)

model_evaluation_task = PythonOperator(
    task_id='model_evaluation',
    python_callable=model_evaluation,
    provide_context=True,
    dag=dag,
)

# Set task dependencies
load_engineered_data_task >> model_training_task >> model_evaluation_task