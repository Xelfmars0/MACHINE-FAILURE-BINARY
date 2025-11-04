# Predictive Maintenance: Binary Failure Classifier

This Streamlit application uses a trained Random Forest model to predict the likelihood of a machine failure based on real-time sensor data.

The app not only provides a "Failure" or "No Failure" prediction (based on a custom-tuned threshold) but also uses **SHAP (SHapley Additive exPlanations)** to explain *why* the model made its decision, showing which features contributed most to the risk.



## 🚀 App Features

* **Interactive Sidebar:** Allows a user to input 6 raw machine sensor values.
* **Live Feature Engineering:** Automatically calculates 3 new features (`TempDiff`, `Power [W]`, `OverstrainMetric`) from the raw inputs.
* **Tuned Prediction:** Applies a custom threshold (e.g., 0.60) to the model's risk score for a more accurate binary decision.
* **Model Explainability (XAI):** Generates a SHAP waterfall plot for *every* prediction, showing which features pushed the risk up or down.

## 🛠️ Tech Stack

This project is built using the following libraries:

* **`streamlit`**: For the interactive web app UI.
* **`pandas`**: For data manipulation and feature engineering.
* **`numpy`**: For numerical operations.
* **`scikit-learn`**: For the Random Forest model, scaler, and encoder.
* **`shap`**: For generating the model explanations.
* **`matplotlib`**: For rendering the SHAP plot in the app.

## 🏃 How to Run Locally

1.  **Clone the repository:**
    ```bash
    git clone [your-github-repo-url]
    cd [your-repo-name]
    ```

2.  **Install dependencies:**
    Make sure you have the `requirements.txt` file from our conversation.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the app:**
    ```bash
    streamlit run app.py
    ```
    (Or the robust version: `python -m streamlit run app.py`)
