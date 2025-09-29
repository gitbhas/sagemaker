import boto3
import sagemaker
from sagemaker.sklearn import SKLearn
from sagemaker.sklearn.model import SKLearnModel
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
import os
from data_split import prepare_data, read_json_from_local

# Initialize SageMaker session and role
sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()

# Load and prepare data
file_path = 'file_audit.json'  # Make sure this path is correct
data = read_json_from_local(file_path)
prepared_data = prepare_data(data)

# Split the data for training and testing
split_date = prepared_data['date'].max() - pd.DateOffset(months=6)
split_date = split_date.date()  # Convert to date object
train_data = prepared_data[prepared_data['date'] < split_date]
test_data = prepared_data[prepared_data['date'] >= split_date]

# Define features and target
features = ['file_count', 'day_of_week', 'month', 'is_weekday']
target = 'total_records'

# Train a model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(train_data[features], train_data[target])

# Evaluate the model
predictions = model.predict(test_data[features])
mse = mean_squared_error(test_data[target], predictions)
print(f"Mean Squared Error: {mse}")

# Save the model locally
model_path = "model"
os.makedirs(model_path, exist_ok=True)
joblib.dump(model, os.path.join(model_path, "model.joblib"))

# Upload the model to S3
model_artifact = sagemaker_session.upload_data(path=os.path.join(model_path, "model.joblib"), key_prefix="sklearn-model")

# Create a SageMaker model
sklearn_model = SKLearnModel(
    model_data=model_artifact,
    role=role,
    entry_point="inference.py",
    framework_version="0.23-1",
    py_version="py3",
    sagemaker_session=sagemaker_session
)

# Deploy the model
predictor = sklearn_model.deploy(
    initial_instance_count=1,
    instance_type="ml.t2.medium",
    endpoint_name="records-predictor"
)

print(f"Endpoint deployed: {predictor.endpoint_name}")

# Test the endpoint
test_data_point = test_data[features].iloc[[0]]  # Use the first row of test data as an example
result = predictor.predict(test_data_point.values)
print(f"Test prediction: {result}")
print(f"Actual value: {test_data[target].iloc[0]}")

# Don't forget to delete the endpoint when you're done to avoid unnecessary charges
# predictor.delete_endpoint()