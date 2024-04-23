import streamlit as st
import joblib
import pandas as pd

# Assuming 'loan_id' is not used for predictions as it's typically an identifier
FEATURE_NAMES = [
    'no_of_dependents', 'education', 'self_employed', 'income_annum', 
    'loan_amount', 'loan_term', 'cibil_score', 'residential_assets_value', 
    'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value'
]


# Load the trained model
@st.experimental_singleton
def load_model():
    try:
        return joblib.load("./RandomForest_model.pkl")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Function to preprocess user input
def preprocess_user_input(user_input):
    # Convert categorical features to numeric format
    user_input['education'] = 1 if user_input['education'] == "Graduate" else 0
    user_input['self_employed'] = 1 if user_input['self_employed'] == "Yes" else 0
    
    return user_input

# Function to predict loan approval
def predict(model, user_input):
    # Convert user input to DataFrame
    user_input_df = pd.DataFrame([user_input])

# Add placeholder values for Movable_assets and Immovable_assets
    user_input_df['Movable_assets'] = 0  # Placeholder value
    user_input_df['Immovable_assets'] = 0
    
    # Load the model
    model = load_model()
    
    # Debug: Print feature names expected by the model
    st.write("Feature names expected by the model:", model.feature_names_in_)
    st.write("Columns in user_input_df:", user_input_df.columns.tolist())

    # Predict loan approval
    prediction = model.predict(user_input_df)
    return prediction

def main():
    st.title('Loan Approval Prediction')
    
    # Load the model
    model = load_model()
    if model is None:
        st.stop()

    # Collect user inputs
    user_input = {}
    user_input['no_of_dependents'] = st.number_input("Number of Dependents", min_value=0, value=0)
    user_input['education'] = st.selectbox("Education Level", ["Graduate", "Not Graduate"], index=0)
    user_input['self_employed'] = st.selectbox("Self Employed", ["Yes", "No"], index=1)
    user_input['income_annum'] = st.number_input("Annual Income", min_value=0)
    user_input['loan_amount'] = st.number_input("Loan Amount", min_value=0)
    user_input['loan_term'] = st.number_input("Loan Term (Months)", min_value=12, max_value=360, value=180)
    user_input['cibil_score'] = st.number_input("CIBIL Score", min_value=300, max_value=900, value=750)

    # Preprocess user input
    user_input_processed = preprocess_user_input(user_input)

    # Print column names for debugging
    st.write("Columns in user_input_df:", user_input_processed.keys())
    st.write("FEATURE_NAMES:", FEATURE_NAMES)

    if st.button('Predict Loan Approval'):
        # Predict loan approval
        prediction = predict(model, user_input_processed)
        st.write('Prediction:', 'Approved' if prediction == 1 else 'Rejected')

if __name__ == '__main__':
    main()
