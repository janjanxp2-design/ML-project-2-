# Zaroori libraries import ki ja rahi hain
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import base64

st.set_page_config(page_title="Student Placement Prediction", page_icon="🎓", layout="centered")
st.title("Student Placement Predictor")
st.markdown("This is a simple ML predictor app built with Streamlit. You can select a model, input feature values, and get predictions.")
ARTIFACT_DIR = "MODEL_FILES"

# back groun video 
def set_video_bg(video_file):
    # 1. ویڈیو فائل کو پڑھنا
    with open(video_file, "rb") as f:
        video_bytes = f.read()
    # 2. Base64 میں تبدیل کرنا
    b64_video = base64.b64encode(video_bytes).decode()
   # 3. HTML اور CSS کے ذریعے ویڈیو کو لائیو پس منظر میں لگانا
    video_html = f"""
    <style>
    #bg-video {{
        position: fixed;
        right: 0;
        bottom: 0;
        min-width: 100%;
        min-height: 100%;
        z-index: -1;
        object-fit: cover;
        filter: brightness(0.6); /* تاکہ تحریر صاف نظر آئے */
    }}
    .stApp {{
        background: transparent;
    }}
    </style>
    <video autoplay loop muted playsinline id="bg-video">
        <source src="data:video/mp4;base64,{b64_video}" type="video/mp4">
    </video>
    """
    st.markdown(video_html, unsafe_allow_html=True)
set_video_bg("bg_video.mp4")

# 1 upload schema.json file 

available=[f.replace('_schema.json','') for f in os.listdir(ARTIFACT_DIR) if f.endswith('_schema.json')]
if not available:
    st.error("No model artifacts found. Run the training notebook first.")
    st.stop()
choice = st.selectbox("Select a model", available)

# 2 load model
@st.cache_data
def load_model(name):
    model=joblib.load(f"{ARTIFACT_DIR}/{name}_model.pkl")
    scaler_path=f"{ARTIFACT_DIR}/{name}_scaler.pkl"
    scaler=joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    with open(f"{ARTIFACT_DIR}/{name}_schema.json", "r") as f:
        schema=json.load(f)
    return model, scaler, schema
model, scaler, schema = load_model(choice)

#3 dynamicully build input form based on schema
st.subheader("Enter Feature values")
user_inputs = {}
with st.form("prediction_form"):
    cols = st.columns(2)
    for i, feat in enumerate(schema['features']):
        target_col = cols[i % 2]
        if feat['dtype'] == 'numeric':
            user_inputs[feat['name']] =target_col.slider(
                feat['name'],
                min_value=float(0),
                max_value=float(feat['max']),
                step=float(feat['step'])
            )
        elif feat['dtype'] == 'categorical':
            user_inputs[feat['name']] = target_col.selectbox(feat['name'], feat['options'])
        else:
            user_inputs[feat['name']] = target_col.number_input(feat['name'], value=0.0)

    submitted = st.form_submit_button("Predict")
# 4 make prediction
if submitted:
    feature_order = [f["name"]for f in schema["features"]]
    input_data = pd.DataFrame([[user_inputs[f] for f in feature_order]] , columns=feature_order)
    input_values = scaler.transform(input_data) if scaler is not None else input_data.values
    if schema['task_type'] == 'regression' :
        prediction = model.predict(input_values)[0]
        st.success(f"Predicted {schema['target_name']}: **{prediction:.4f}**")
    elif schema['task_type'] == 'classification' :
        prediction= model.predict(input_values)[0]
        clas_names= schema.get('class_names' ,[str(prediction)])
        prediction_lable = prediction
        st.success(f"predicted {schema["target_name"]}  :  *  {prediction_lable}  *      ")
        if hasattr (model, "predict_proba"):
            proba = model.predict_proba(input_values)[0]
            proba_df = pd.DataFrame({'Class' : clas_names, 'Probability' : proba})
            st.bar_chart(proba_df.set_index('Class'))
st.markdown("---")
st.caption("Notice: this same app.py serves both the regression model and the classification "
           "model above, with zero code changes, because the form is built from schema.json.")

