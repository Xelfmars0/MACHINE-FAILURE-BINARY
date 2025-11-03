import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import shap
import matplotlib.pyplot as plt


@st.cache_resource
def load_assets():
    # Load the model
    with open('Binary_Machine_Failure_model.pkl', 'rb') as file:
        model = pickle.load(file)
    
    # Load the scaler
    with open('Binary_Machine_Failure_scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    
    # Load the encoder
    with open('Binary_Machine_Failure_encoder.pkl', 'rb') as file:
        encoder = pickle.load(file)
        
    # Load the explainer
    with open('binary_shap_explainer.pkl', 'rb') as file:
        explainer = pickle.load(file)
    return model, scaler, encoder, explainer 

@st.cache_data
def load_config():
    # Load the column order
    with open('binary_model_columns.json', 'r') as f:
        columns = json.load(f)
        
    # Load the threshold
    with open('binary_model__threshconfig.json', 'r') as f:
        config = json.load(f)
        threshold = config['binary_threshold']
        
    return columns, threshold

def main():
    st.title("Binary Machine Failure Prediction")
    
    # Load assets
    model, scaler, encoder ,explainer = load_assets()
    columns, threshold = load_config()
    raw_numerical_features = ['Air temperature [K]', 'Process temperature [K]', 
                            'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
    raw_categorical_features = ['Type']

    # Define ALL features the SCALER expects (raw + engineered)
    features_to_scale = columns.copy()

    st.header("Input Features")
    
    # Collect user input
    with st.sidebar:
        input_data = {}
        input_data['Type'] = st.sidebar.selectbox('Product Type', ['L', 'M', 'H'])
        input_data['Air temperature [K]'] = st.sidebar.slider('Air Temperature (K)', 295.3, 304.5, 300.1, 0.1)
        input_data['Process temperature [K]'] = st.sidebar.slider('Process Temperature (K)', 305.0, 314.0, 310.2, 0.1)
        input_data['Rotational speed [rpm]'] = st.sidebar.slider('Rotational Speed (rpm)', 1160, 2900, 1500)
        input_data['Torque [Nm]'] = st.sidebar.slider('Torque (Nm)', 3.0, 80.0, 40.5, 0.1)
        input_data['Tool wear [min]'] = st.sidebar.slider('Tool wear [min]', 0, 260, 108)

 
    if st.button("Predict"):
        # Prepare the input data
        input_df = pd.DataFrame([input_data])

        processed_df = input_df.copy()

        st.write("Engineering new features...")
        processed_df['TempDiff'] = processed_df['Process temperature [K]'] - processed_df['Air temperature [K]']
        processed_df['Power [W]'] = processed_df['Torque [Nm]'] * (processed_df['Rotational speed [rpm]'] * 2 * np.pi / 60)
        processed_df['OverstrainMetric'] = processed_df['Tool wear [min]'] * processed_df['Torque [Nm]']
        
        st.dataframe(processed_df[['TempDiff', 'Power [W]', 'OverstrainMetric']].style.format("{:.2f}"))
        
        with st.expander("See Raw Input Data"):
            st.dataframe(input_df)

        # Encode categorical features
        categorical_cols = ['Type']
        processed_df['Type'] = encoder.transform(processed_df['Type'])
        

        # Scale numerical features
        processed_df[features_to_scale] = scaler.transform(processed_df[features_to_scale])

        # Ensure the order of columns
        final_preprocessed_df = processed_df[columns]

        # Make prediction
        st.write("Making Predictions...")
        prediction_proba = model.predict_proba(final_preprocessed_df)[:, 1][0]
        prediction = 1 if prediction_proba >= threshold else 0
        
        # Display results
        st.subheader("Prediction Results")
        st.success(f"Predicted Class: {'Failure' if prediction == 1 else 'No Failure'}")
        with st.expander("See Model Prediction Probability"):
            st.write(f"Prediction Probability: {prediction_proba:.4f}")
            st.write(f"A predition probability above {threshold} indicates a 'Failure'.")

        st.subheader("Why did the model decide this?")

        # Calculate SHAP values for THIS ONE PREDICTION
        shap_values = explainer.shap_values(final_preprocessed_df)[:,:,1]

        # Get the expected value (the "base" prediction)
        base_value = explainer.expected_value[1] 

        shap_explanation = shap.Explanation(
        values=shap_values[0], 
        base_values=base_value, 
        data=final_preprocessed_df.iloc[0], 
        feature_names=final_preprocessed_df.columns.tolist()
        )

        # Create the plot
        # st.shap(shap.force_plot(...)) is the new syntax
        fig, ax = plt.subplots(figsize=(10, 5))
        shap.plots.waterfall(shap_explanation, max_display=10, show=False)
        st.pyplot(fig)

if __name__ == "__main__":
    main()