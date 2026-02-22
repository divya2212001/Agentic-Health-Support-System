import joblib
import pandas as pd
import numpy as np

try:
    model = joblib.load('models/best_model.pkl')
    
    print(f"Model type: {type(model)}")
    if hasattr(model, 'steps'):
        print(f"Pipeline steps: {[s[0] for s in model.steps]}")
    
    # Check for feature names
    if hasattr(model, 'feature_names_in_'):
        print(f"Features: {model.feature_names_in_}")
    else:
        print("Feature names not found.")
except Exception as e:
    print(f"Error loading model: {e}")
