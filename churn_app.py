import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder

# Load the trained model and preprocessors
@st.cache_resource
def load_model():
    model = joblib.load('churn_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

# Load data for analysis
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\91861\OneDrive\deeeeeeeeeeeeee\OneDrive\문서\WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    return df

def main():
    st.set_page_config(page_title="Customer Churn Prediction", layout="wide")
    
    st.title("🔮 Customer Churn Prediction Dashboard")
    st.markdown("---")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Dashboard", "Predict Churn", "Data Analysis"])
    
    if page == "Dashboard":
        dashboard_page()
    elif page == "Predict Churn":
        prediction_page()
    else:
        analysis_page()

def dashboard_page():
    st.header("📊 Churn Analytics Dashboard")
    
    df = load_data()
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_customers = len(df)
        st.metric("Total Customers", f"{total_customers:,}")
    
    with col2:
        churned_customers = len(df[df['Churn'] == 'Yes'])
        st.metric("Churned Customers", f"{churned_customers:,}")
    
    with col3:
        churn_rate = (churned_customers / total_customers) * 100
        st.metric("Churn Rate", f"{churn_rate:.1f}%")
    
    with col4:
        avg_monthly_charges = df['MonthlyCharges'].mean()
        st.metric("Avg Monthly Charges", f"${avg_monthly_charges:.2f}")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Churn by Contract Type
        churn_contract = df.groupby(['Contract', 'Churn']).size().unstack()
        fig = px.bar(churn_contract, title="Churn by Contract Type")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Monthly Charges Distribution
        fig = px.histogram(df, x='MonthlyCharges', color='Churn', 
                          title="Monthly Charges Distribution by Churn")
        st.plotly_chart(fig, use_container_width=True)

def prediction_page():
    st.header("🎯 Customer Churn Prediction")
    
    st.write("Enter customer details to predict churn probability:")
    
    # Create input form
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Partner", ["No", "Yes"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            phone_service = st.selectbox("Phone Service", ["No", "Yes"])
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        
        with col2:
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
            device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
            tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        
        with col3:
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
            payment_method = st.selectbox("Payment Method", 
                                        ["Electronic check", "Mailed check", 
                                         "Bank transfer (automatic)", "Credit card (automatic)"])
            monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 50.0)
            total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 1000.0)
        
        submitted = st.form_submit_button("Predict Churn")
        
        if submitted:
            # Create input dataframe
            input_data = pd.DataFrame({
                'gender': [gender],
                'SeniorCitizen': [1 if senior_citizen == "Yes" else 0],
                'Partner': [partner],
                'Dependents': [dependents],
                'tenure': [tenure],
                'PhoneService': [phone_service],
                'MultipleLines': [multiple_lines],
                'InternetService': [internet_service],
                'OnlineSecurity': [online_security],
                'OnlineBackup': [online_backup],
                'DeviceProtection': [device_protection],
                'TechSupport': [tech_support],
                'StreamingTV': [streaming_tv],
                'StreamingMovies': [streaming_movies],
                'Contract': [contract],
                'PaperlessBilling': [paperless_billing],
                'PaymentMethod': [payment_method],
                'MonthlyCharges': [monthly_charges],
                'TotalCharges': [total_charges]
            })
            
            # Encode categorical variables
            categorical_columns = input_data.select_dtypes(include=['object']).columns
            le = LabelEncoder()
            
            # Load original data to fit encoder
            df = load_data()
            df_encoded = df.copy()
            df_encoded = df_encoded.drop('customerID', axis=1)
            
            for col in categorical_columns:
                if col in df_encoded.columns:
                    le.fit(df_encoded[col])
                    input_data[col] = le.transform(input_data[col])
            
            # Load model and make prediction
            model, scaler = load_model()
            
            # Make prediction
            prediction = model.predict(input_data)[0]
            prediction_proba = model.predict_proba(input_data)[0]
            
            # Display results
            st.markdown("---")
            st.subheader("Prediction Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction == 1:
                    st.error("⚠️ **HIGH RISK**: Customer likely to churn")
                else:
                    st.success("✅ **LOW RISK**: Customer unlikely to churn")
            
            with col2:
                churn_probability = prediction_proba[1] * 100
                st.metric("Churn Probability", f"{churn_probability:.1f}%")
            
            # Probability gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = churn_probability,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Churn Probability"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 80
                    }
                }
            ))
            
            st.plotly_chart(fig, use_container_width=True)

def analysis_page():
    st.header("📈 Data Analysis & Insights")
    
    df = load_data()
    
    # Analysis options
    analysis_type = st.selectbox("Select Analysis Type", 
                                ["Churn by Demographics", "Service Usage Analysis", "Financial Analysis"])
    
    if analysis_type == "Churn by Demographics":
        col1, col2 = st.columns(2)
        
        with col1:
            # Gender analysis
            gender_churn = df.groupby(['gender', 'Churn']).size().unstack()
            fig = px.bar(gender_churn, title="Churn by Gender")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Age analysis
            age_churn = df.groupby(['SeniorCitizen', 'Churn']).size().unstack()
            fig = px.bar(age_churn, title="Churn by Age Group")
            st.plotly_chart(fig, use_container_width=True)
    
    elif analysis_type == "Service Usage Analysis":
        # Internet Service analysis
        internet_churn = df.groupby(['InternetService', 'Churn']).size().unstack()
        fig = px.bar(internet_churn, title="Churn by Internet Service Type")
        st.plotly_chart(fig, use_container_width=True)
        
        # Contract analysis
        contract_churn = df.groupby(['Contract', 'Churn']).size().unstack()
        fig = px.bar(contract_churn, title="Churn by Contract Type")
        st.plotly_chart(fig, use_container_width=True)
    
    else:  # Financial Analysis
        col1, col2 = st.columns(2)
        
        with col1:
            # Monthly charges by churn
            fig = px.box(df, x='Churn', y='MonthlyCharges', title="Monthly Charges by Churn")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Total charges by churn
            fig = px.box(df, x='Churn', y='TotalCharges', title="Total Charges by Churn")
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
