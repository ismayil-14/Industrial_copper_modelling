import pandas as pd
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu
import re
from PIL import Image
import warnings
import pickle
warnings.filterwarnings("ignore")
# Create a Streamlit app
icon = Image.open("icon.jpeg")
st.set_page_config(page_title= "Industrial Copper Modelling | By Mohamed Ismayil",
                page_icon= icon,
                layout= "wide",
                initial_sidebar_state= "expanded",
                menu_items={'About': """# This dashboard app is created by *Mohamed Ismayil*!"""}
                )

st.write("""
<div style='text-align:center'>
    <h1 style='color:#009999;'>Industrial Copper Modeling Application</h1>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    selected = option_menu("Menu", ["Home","PREDICT SELLING PRICE", "PREDICT STATUS"], 
                        icons=["house","graph-up-arrow","graph-up-arrow"],
                        menu_icon= "menu-button-wide",
                        default_index=0,
                        styles={"nav-link": {"font-size": "25px", "text-align": "left", "margin": "-2px", "--hover-color": "#FF5A5F"},
                                "nav-link-selected": {"background-color": "#FF5A5F"}}
                        )

# HOME PAGE
if selected == "Home":
    st.markdown("## :blue[Domain] : Manufacturing")
    st.markdown("## :blue[Technologies used] : Python, Pandas, Plotly, Streamlit, Sciket-Learn")
    st.markdown("## :blue[Overview] : The copper industry deals with data related to sales and pricing, which often suffers from skewness and noise. These data issues can lead to inaccurate manual predictions, making it difficult to make optimal pricing decisions. ")

status_options = ['Won', 'Draft', 'To be approved', 'Lost', 'Not lost for AM', 'Wonderful', 'Revised', 'Offered', 'Offerable']
item_type_options = ['W', 'WI', 'S', 'Others', 'PL', 'IPL', 'SLAWR']
country_options = [28., 25., 30., 32., 38., 78., 27., 77., 113., 79., 26., 39., 40., 84., 80., 107., 89.]
application_options = [10., 41., 28., 59., 15., 4., 38., 56., 42., 26., 27., 19., 20., 66., 29., 22., 40., 25., 67., 79., 3., 99., 2., 5., 39., 69., 70., 65., 58., 68.]
product = ['611112', '611728', '628112', '628117', '628377', '640400', '640405', '640665', '611993', '929423819', '1282007633', '1332077137', '164141591', '164336407', '164337175', '1665572032', '1665572374', '1665584320', '1665584642', '1665584662', '1668701376', '1668701698', '1668701718', '1668701725', '1670798778', '1671863738', '1671876026', '1690738206', '1690738219', '1693867550', '1693867563', '1721130331', '1722207579']

if selected == "PREDICT SELLING PRICE":
    st.header("Predict Selling Price")
    st.markdown("### Enter the details below to predict the selling price.")

    with st.form("predict_price_form"):
        col1, col2 = st.columns([1, 1])

        with col1:
            status = st.selectbox("Status", status_options)
            item_type = st.selectbox("Item Type", item_type_options)
            country = st.selectbox("Country", sorted(country_options))
            application = st.selectbox("Application", sorted(application_options))
            product_ref = st.selectbox("Product Reference", product)
        
        with col2:
            quantity_tons = st.text_input("Enter Quantity Tons (Min: 611728 & Max: 1722207579)")
            thickness = st.text_input("Enter Thickness (Min: 0.18 & Max: 400)")
            width = st.text_input("Enter Width (Min: 1, Max: 2990)")
            customer = st.text_input("Customer ID (Min: 12458, Max: 30408185)")
            submit_button = st.form_submit_button(label="Predict Selling Price")

    if submit_button:
        # Validate inputs
        flag = 0 
        pattern = "^(?:\d+|\d*\.\d+)$"
        for i in [quantity_tons, thickness, width, customer]:             
            if not re.match(pattern, i):
                flag = 1
                break
        
        if flag == 1:
            st.error(f"You have entered an invalid value: {i}. Please enter a valid number without spaces.")
        else:
            # Load models and scalers
            with open("regression_best_model.pkl", 'rb') as file:
                loaded_model = pickle.load(file)
            with open('regression_standard_scaler.pkl', 'rb') as f:
                scaler_loaded = pickle.load(f)
            with open("type_encoder.pkl", 'rb') as f:
                t_loaded = pickle.load(f)
            with open("status_encoder.pkl", 'rb') as f:
                s_loaded = pickle.load(f)

            # Prepare input data
            new_sample = np.array([[np.log(float(quantity_tons)), application, np.log(float(thickness)), float(width), country, float(customer), int(product_ref), item_type, status]])
            new_sample_ohe = t_loaded.transform(new_sample[:, [7]]).toarray()
            new_sample_be = s_loaded.transform(new_sample[:, [8]]).toarray()
            new_sample = np.concatenate((new_sample[:, [0, 1, 2, 3, 4, 5, 6]], new_sample_ohe, new_sample_be), axis=1)
            new_sample = scaler_loaded.transform(new_sample)
            
            # Make prediction
            new_pred = loaded_model.predict(new_sample)[0]
            st.success(f"Predicted Selling Price: ${np.exp(new_pred):.2f}")

elif selected == "PREDICT STATUS":
    st.header("Predict Status")
    st.markdown("### Enter the details below to predict the status.")

    with st.form("predict_status_form"):
        col1, col2 = st.columns([1, 1])

        with col1:
            cquantity_tons = st.text_input("Enter Quantity Tons (Min: 611728 & Max: 1722207579)")
            cthickness = st.text_input("Enter Thickness (Min: 0.18 & Max: 400)")
            cwidth = st.text_input("Enter Width (Min: 1, Max: 2990)")
            ccustomer = st.text_input("Customer ID (Min: 12458, Max: 30408185)")
            cselling = st.text_input("Selling Price (Min: 1, Max: 100001015)")
        
        with col2:
            citem_type = st.selectbox("Item Type", item_type_options)
            ccountry = st.selectbox("Country", sorted(country_options))
            capplication = st.selectbox("Application", sorted(application_options))
            cproduct_ref = st.selectbox("Product Reference", product)
            csubmit_button = st.form_submit_button(label="Predict Status")

    if csubmit_button:
        # Validate inputs
        cflag = 0 
        pattern = "^(?:\d+|\d*\.\d+)$"
        for k in [cquantity_tons, cthickness, cwidth, ccustomer, cselling]:             
            if not re.match(pattern, k):
                cflag = 1
                break
        
        if cflag == 1:
            st.error(f"You have entered an invalid value: {k}. Please enter a valid number without spaces.")
        else:
            # Load models and scalers
            with open("classification_model.pkl", 'rb') as file:
                cloaded_model = pickle.load(file)
            with open('classification_standard_scaler.pkl', 'rb') as f:
                cscaler_loaded = pickle.load(f)
            with open("classification_encoder.pkl", 'rb') as f:
                ct_loaded = pickle.load(f)

            # Prepare input data
            new_sample = np.array([[np.log(float(cquantity_tons)), np.log(float(cselling)), capplication, np.log(float(cthickness)), float(cwidth), ccountry, int(ccustomer), int(cproduct_ref), citem_type]])
            new_sample_ohe = ct_loaded.transform(new_sample[:, [8]]).toarray()
            new_sample = np.concatenate((new_sample[:, [0, 1, 2, 3, 4, 5, 6, 7]], new_sample_ohe), axis=1)
            new_sample = cscaler_loaded.transform(new_sample)
            
            # Make prediction
            new_pred = cloaded_model.predict(new_sample)
            if new_pred == 1:
                st.success("The Status is Won")
            else:
                st.error("The Status is Lost")
