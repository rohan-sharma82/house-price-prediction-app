import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
# Make sure 'mlr_model.pkl' is in the same directory as this app.py file
try:
    model = joblib.load('mlr_model.pkl')
except FileNotFoundError:
    st.error("Error: 'mlr_model.pkl' not found. Please ensure the model file is in the same directory.")
    st.stop()

st.title('House Price Prediction App')
st.write('Enter the details of the house to get a price prediction.')

# Define the features based on your original notebook
# Categorical columns (unique values are taken from df_pd in the notebook context)
categorical_features = {
    'BLDGTYPE': ['1Fam', 'TwnhsE', 'Duplex', 'Twnhs', '2fmCon'],
    'HOUSESTYLE': ['2Story', '1Story', '1.5Fin', '1.5Unf', 'SFoyer', 'SLvl', '2.5Unf', '2.5Fin'],
    'ROOFSTYLE': ['Gable', 'Hip', 'Gambrel', 'Mansard', 'Flat', 'Shed'],
    'EXTERCOND': ['TA', 'Gd', 'Fa', 'Po', 'Ex'],
    'FOUNDATION': ['PConc', 'CBlock', 'BrkTil', 'Wood', 'Slab', 'Stone'],
    'BSMTCOND': ['TA', 'Gd', 'Mn', 'Fa', 'Po'], # Imputed 'NA' (no basement) with most frequent 'TA' if there were NaNs in original df
    'HEATING': ['GasA', 'GasW', 'Grav', 'Wall', 'OthW', 'Floor'],
    'HEATINGQC': ['Ex', 'Gd', 'TA', 'Fa', 'Po'],
    'CENTRALAIR': ['Y', 'N'],
    'ELECTRICAL': ['SBrkr', 'FuseF', 'FuseA', 'FuseP', 'Mix'], # Imputed 'NA' with most frequent 'SBrkr' if there were NaNs in original df
    'KITCHENQUAL': ['Gd', 'TA', 'Ex', 'Fa'],
    'FIREPLACEQU': ['Gd', 'TA', 'Fa', 'Ex', 'Po'], # Imputed 'NA' with most frequent 'Gd' if there were NaNs in original df
    'GARAGETYPE': ['Attchd', 'Detchd', 'BuiltIn', 'CarPort', 'Basment', '2Types'], # Imputed 'NA' (no garage) with most frequent 'Attchd' if there were NaNs in original df
    'GARAGEFINISH': ['RFn', 'Unf', 'Fin'], # Imputed 'NA' (no garage) with most frequent 'Unf' if there were NaNs in original df
    'GARAGECOND': ['TA', 'Fa', 'Gd', 'Po', 'Ex'], # Imputed 'NA' (no garage) with most frequent 'TA' if there were NaNs in original df
    'POOLQC': ['Ex', 'Fa', 'Gd'], # Imputed 'NA' (no pool) with most frequent 'Ex' if there were NaNs in original df. Note: This column was mostly NaN and is imputed to most frequent. Consider if this is appropriate for user input.
    'FENCE': ['MnPrv', 'GdWo', 'GdPrv', 'MnWw'] # Imputed 'NA' (no fence) with most frequent 'MnPrv' if there were NaNs in original df
}

# Numerical features
numerical_features = [
    'LOTAREA', 'OVERALLCOND', 'YEARBUILT', 'FULLBATH', 'HALFBATH',
    'BEDROOMABVGR', 'KITCHENABVGR', 'TOTRMSABVGRD', 'FIREPLACES', 'GARAGECARS',
    'POOLAREA', 'MOSOLD', 'YRSOLD'
]

# Create input widgets
input_data = {}

st.sidebar.header('House Features Input')

for feature in numerical_features:
    # Provide default values based on dataset or common sense
    if feature == 'LOTAREA':
        input_data[feature] = st.sidebar.number_input(f'{feature.replace("_", " ").title()}', min_value=0, value=10000)
    elif feature == 'YEARBUILT':
        input_data[feature] = st.sidebar.number_input(f'{feature.replace("_", " ").title()}', min_value=1800, max_value=2024, value=2000)
    elif feature in ['FULLBATH', 'HALFBATH', 'BEDROOMABVGR', 'KITCHENABVGR', 'TOTRMSABVGRD', 'FIREPLACES', 'GARAGECARS']:
        input_data[feature] = st.sidebar.number_input(f'{feature.replace("_", " ").title()}', min_value=0, value=1)
    elif feature in ['OVERALLCOND', 'POOLAREA']:
        input_data[feature] = st.sidebar.number_input(f'{feature.replace("_", " ").title()}', min_value=0, value=5)
    elif feature == 'MOSOLD':
        input_data[feature] = st.sidebar.number_input(f'{feature.replace("_", " ").title()}', min_value=1, max_value=12, value=7)
    elif feature == 'YRSOLD':
        input_data[feature] = st.sidebar.number_input(f'{feature.replace("_", " ").title()}', min_value=2006, max_value=2010, value=2008)

for feature, options in categorical_features.items():
    # Default to the first option or a meaningful default
    default_index = options.index('TA') if 'TA' in options else (options.index('Gd') if 'Gd' in options else 0)
    input_data[feature] = st.sidebar.selectbox(f'{feature.replace("_", " ").title()}', options, index=default_index)

# Convert input data to DataFrame
input_df = pd.DataFrame([input_data])

# Ensure the columns are in the same order as during training
# This is crucial for the preprocessor and model to work correctly
# The order of columns in 'features' variable from your notebook was:
# ['LOTAREA', 'BLDGTYPE', 'HOUSESTYLE', 'OVERALLCOND', 'YEARBUILT', 'ROOFSTYLE',
#  'EXTERCOND', 'FOUNDATION', 'BSMTCOND', 'HEATING', 'HEATINGQC', 'CENTRALAIR',
#  'ELECTRICAL', 'FULLBATH', 'HALFBATH', 'BEDROOMABVGR', 'KITCHENABVGR',
#  'TOTRMSABVGRD', 'FIREPLACES', 'FIREPLACEQU', 'GARAGETYPE', 'GARAGEFINISH',
#  'GARAGECARS', 'GARAGECOND', 'POOLAREA', 'POOLQC', 'FENCE', 'MOSOLD', 'YRSOLD']

expected_columns = [
    'LOTAREA', 'BLDGTYPE', 'HOUSESTYLE', 'OVERALLCOND', 'YEARBUILT', 'ROOFSTYLE',
    'EXTERCOND', 'FOUNDATION', 'BSMTCOND', 'HEATING', 'HEATINGQC', 'CENTRALAIR',
    'ELECTRICAL', 'FULLBATH', 'HALFBATH', 'BEDROOMABVGR', 'KITCHENABVGR',
    'TOTRMSABVGRD', 'FIREPLACES', 'FIREPLACEQU', 'GARAGETYPE', 'GARAGEFINISH',
    'GARAGECARS', 'GARAGECOND', 'POOLAREA', 'POOLQC', 'FENCE', 'MOSOLD', 'YRSOLD'
]

# Reindex input_df to match the expected column order
input_df = input_df[expected_columns]


if st.button('Predict House Price'):
    try:
        # Make prediction (output is log-transformed)
        log_predicted_price = model.predict(input_df)[0]

        # Inverse transform the prediction
        predicted_price = np.exp(log_predicted_price)

        st.success(f'The predicted sale price is: ${predicted_price:,.2f}')
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
