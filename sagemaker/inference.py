import joblib
import os
import json
import numpy as np

def model_fn(model_dir):
    models = {}
    for file in os.listdir(model_dir):
        if file.endswith("_model.joblib"):
            model_name = file.replace("_model.joblib", "")
            models[model_name] = joblib.load(os.path.join(model_dir, file))
    return models

def input_fn(request_body, request_content_type):
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
        return np.array(input_data)
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, models):
    results = {}
    for model_name, model in models.items():
        results[model_name] = model.predict(input_data)
    return np.array(list(results.values())).T

def output_fn(prediction, accept):
    if accept == 'application/json':
        return json.dumps(prediction.tolist())
    raise ValueError(f"Unsupported accept type: {accept}")