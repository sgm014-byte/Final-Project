# %% [markdown]
# # A Jupyter Notebook to create STREAMLIT App to create a simple UI for a ML model
# ### Uses a previously created ML model saved as a pickle file and user must enter a new customer

# %%
# Streamlit ML Web App Demo: Lending Default Prediction
import streamlit as st
import pandas as pd
import pickle

# Open the file and load the model
import joblib

# Open the file and load the model
file_to_load = 'lasso_model.pkl'
try:
    loaded_model = joblib.load(file_to_load)
except Exception as error:
    st.error(f"Model load failed: {error}")
    st.stop()

# --- Streamlit UI ---
st.set_page_config(page_title="💰 Loan Default ML Predictor", layout="centered")
st.title("💰 Loan Default Prediction")
st.markdown("Enter a potential loan customer's details to predict their risk of default.")

# User input
fico_range_high = st.checkbox("fico_range_high?")
int_rate = st.slider("Int Rate", 20000, 1000000, 80000)
loan_amnt = st.slider("loan_amnt", 0, 40, 10)
installment = st.slider("installment", 0, 40, 10)
term_num = st.slider("FICO Score", 300, 850, 650)

# Prediction
if st.button("Predict Loan Default"):
    new_customer = pd.DataFrame({
    'fico_range_high': [fico_range_high],
    'int_rate': [int_rate],
    'loan_amnt': [loan_amnt],
    'installment': [installment],
    'term_num': [term_num]
    })

    # Use the model to find the predicted probability of default
    predicted_prob = loaded_model.predict_proba(new_customer)[:, 0]
    # Use the model to find the predicted class
    predicted_class = loaded_model.predict(new_customer)
    
    # Format the predicted probability with two decimals and a leading zero
    formatted_prob = f"{predicted_prob[0]:.2f}"
    
    # Display the predicted probability and class in Streamlit
    st.write(f"Predicted Probability of Default: **{formatted_prob}**")
    if predicted_class[0] == 0:
        st.success("Predicted Class: **Default**")
    else:
        st.success("Predicted Class: **Not Default**")
    
    # Show the default probability and not default as a pie chart
    probabilities = [predicted_prob[0], 1 - predicted_prob[0]]
    labels = ['Default', 'Not Default']
    chart_data = pd.DataFrame({'Probability': probabilities}, index=labels)
    st.write("Prediction Breakdown:")
    # VERY simple chart
    st.bar_chart(chart_data)
    
st.markdown("---")
st.markdown("**Developed by Matt Bailey as simple ML Web App Demo** | Powered by Streamlit")
