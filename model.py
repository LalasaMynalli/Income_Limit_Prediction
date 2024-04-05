import joblib
import numpy as np

model_path = r"C:\Users\Dell\Desktop\ML course\Income_Inequality_Prediction_Project2\Model\catboost_model.joblib"
model = joblib.load(model_path)

def predict(data):
    prediction = model.predict(data)
    return prediction

def ordinal_encoder(input_val, feats): 
    feat_val = list(1+np.arange(len(feats)))
    feat_key = feats
    feat_dict = dict(zip(feat_key, feat_val))
    return feat_dict[input_val]