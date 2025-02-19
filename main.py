import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle
import _pickle

# Page config
st.set_page_config(
    page_title="Churn Prediction Dashboard",
    layout="wide"
)

# Load the models
@st.cache_resource
def load_models():
    """Load the trained models"""
    try:
        with open('models/logistic_regression.pkl', 'rb') as f:
            lr_model = pickle.load(f)
        with open('models/random_forest.pkl', 'rb') as f:
            rf_model = pickle.load(f)
        with open('models/xgboost.pkl', 'rb') as f:
            xgb_model = pickle.load(f)
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('models/feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        return lr_model, rf_model, xgb_model, scaler, feature_names
    except (FileNotFoundError, _pickle.UnpicklingError) as e:
        st.error("⚠️ Error: Model files not found or corrupted. Please ensure model files are present in the 'models' directory.")
        st.stop()

# Load models
lr_model, rf_model, xgb_model, scaler, feature_names = load_models()

# Title
st.title("📊 Customer Churn Prediction Dashboard")

# Sidebar for dataset upload
st.sidebar.subheader("📂 Upload Customer Data")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    # Read the uploaded dataset
    df = pd.read_csv('churn.csv')

    # Show a preview of the dataset
    st.sidebar.write("### Preview of Uploaded Data:")
    st.sidebar.dataframe(df.head())

    # Data Preprocessing
    if "tenure" in df.columns:
        df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce")
    if "MonthlyCharges" in df.columns:
        df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
    
    # Calculate AverageMonthlySpend
    df["AverageMonthlySpend"] = df["MonthlyCharges"]  # Since we're not using TotalCharges anymore
    df["AverageMonthlySpend"] = df["AverageMonthlySpend"].round(2)

    # Drop unnecessary columns
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
    if "Churn" in df.columns:
        df = df.drop(columns=["Churn"])
    if "TotalCharges" in df.columns:  # Add this to remove TotalCharges
        df = df.drop(columns=["TotalCharges"])
    
    # Handle missing values
    df = df.fillna(0)

    # Convert to the same format as training data
    df = pd.get_dummies(df)
    
    # Ensure columns match training data
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    
    # Keep only the columns used during training
    df = df[feature_names]

    # Standardize numerical features
    numerical_features = ["tenure", "MonthlyCharges", "AverageMonthlySpend"]
    df[numerical_features] = scaler.transform(df[numerical_features])

    # Predict Churn Probability
    st.subheader("🔍 Predicted Churn Probabilities")
    cols = st.columns(3)

    predictions = {}
    models = {
        "Logistic Regression": lr_model,
        "Random Forest": rf_model,
        "XGBoost": xgb_model
    }

    for idx, (name, model) in enumerate(models.items()):
        with cols[idx]:
            probs = model.predict_proba(df)[:, 1]
            avg_prob = np.mean(probs)
            predictions[name] = avg_prob
            st.metric(
                name,
                f"{avg_prob:.1%}",
                delta="⚠ High Risk" if avg_prob > 0.5 else "✅ Low Risk"
            )

    # Feature Importance
    st.subheader("🔬 Feature Importance Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        # First graph (Feature Importance)
        importances = pd.DataFrame({
            "Feature": feature_names,
            "Importance": rf_model.feature_importances_
        })
        importances = importances[~importances['Feature'].str.contains('TotalCharges', case=False)]
        importances = importances.sort_values("Importance", ascending=False).head(10)

        fig1 = px.bar(
            importances,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Top 10 Factors Influencing Churn",
            custom_data=['Feature', 'Importance']
        )
        fig1.update_layout(
            yaxis={"categoryorder": "total ascending"},
            hoverlabel={"bgcolor": "black", "font_size": 14, "font": {"color": "white"}},
            hovermode='y unified',
            title_font_size=20,
            font=dict(size=14, color="white"),
            plot_bgcolor='black',
            paper_bgcolor='black',
            xaxis=dict(
                title_font=dict(size=16, color="white"),
                tickfont=dict(size=14, color="white"),
                gridcolor='#333333',
                showgrid=True
            ),
            yaxis_title=dict(
                font=dict(size=16, color="white")
            ),
            yaxis_tickfont=dict(size=14, color="white")
        )
        fig1.update_traces(
            hovertemplate="<b>%{customdata[0]}</b><br>Importance: %{customdata[1]:.3f}<extra></extra>",
            marker_color='#3498db'
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Second graph (Distribution)
        all_predictions = pd.DataFrame({
            'Model': np.repeat(['Logistic Regression', 'Random Forest', 'XGBoost'], len(df)),
            'Probability': np.concatenate([
                lr_model.predict_proba(df)[:, 1],
                rf_model.predict_proba(df)[:, 1],
                xgb_model.predict_proba(df)[:, 1]
            ])
        })
        
        fig2 = px.histogram(
            all_predictions,
            x="Probability",
            color="Model",
            nbins=20,
            title="Distribution of Churn Probabilities",
            labels={'Probability': 'Churn Probability', 'count': 'Number of Customers'},
            opacity=0.7,
            color_discrete_sequence=['#3498db', '#2ecc71', '#e74c3c']
        )
        fig2.update_layout(
            xaxis_title="Churn Probability",
            yaxis_title="Number of Customers",
            barmode='overlay',
            hovermode='x unified',
            hoverlabel={"bgcolor": "black", "font_size": 14, "font": {"color": "white"}},
            title_font_size=20,
            font=dict(size=14, color="white"),
            plot_bgcolor='black',
            paper_bgcolor='black',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99,
                font=dict(size=14, color="white"),
                bgcolor='rgba(0,0,0,0)'
            ),
            xaxis=dict(
                title_font=dict(size=16, color="white"),
                tickfont=dict(size=14, color="white"),
                gridcolor='#333333',
                showgrid=True
            ),
            yaxis=dict(
                title_font=dict(size=16, color="white"),
                tickfont=dict(size=14, color="white"),
                gridcolor='#333333',
                showgrid=True
            )
        )
        fig2.update_traces(
            hovertemplate="<b>%{x:.2%}</b><br>Count: %{y}<br>Model: %{legendgroup}<extra></extra>"
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Summary Insights
    st.subheader("📌 Key Insights")
    avg_prob = np.mean(list(predictions.values()))
    st.write(f"- **Overall churn risk**: {'🔴 High' if avg_prob > 0.5 else '🟢 Low'} ({avg_prob:.1%})")
    st.write(f"- **Most influential factor**: {importances.iloc[0]['Feature']}")

else:
    # Manual Input Form
    st.subheader("📝 Enter Customer Details for Prediction")
    
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("📋 Customer Information")
        
        # Basic Info
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=120, value=12)
        
        # Services
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        phone = st.selectbox("Phone Service", ["No", "Yes"])
        security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        
        # Contract and Billing
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["No", "Yes"])
        payment = st.selectbox("Payment Method", 
            ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0)

    with col2:
        if st.button("🚀 Predict Churn Probability"):
            # Prepare input data
            input_data = {
                'gender': gender,
                'SeniorCitizen': 1 if senior == "Yes" else 0,
                'Partner': partner,
                'Dependents': dependents,
                'tenure': tenure,
                'PhoneService': phone,
                'InternetService': internet,
                'OnlineSecurity': security,
                'OnlineBackup': backup,
                'DeviceProtection': protection,
                'TechSupport': support,
                'Contract': contract,
                'PaperlessBilling': paperless,
                'PaymentMethod': payment,
                'MonthlyCharges': monthly_charges,
                'AverageMonthlySpend': monthly_charges
            }
            
            # Create DataFrame and preprocess
            df = pd.DataFrame([input_data])
            df = pd.get_dummies(df)
            
            # Ensure columns match training data
            for col in feature_names:
                if col not in df.columns:
                    df[col] = 0
            df = df[feature_names]

            # Scale numerical features
            numerical_features = ["tenure", "MonthlyCharges", "AverageMonthlySpend"]
            df[numerical_features] = scaler.transform(df[numerical_features])

            # Make predictions
            st.subheader("📊 Churn Predictions")
            cols = st.columns(3)
            
            models_dict = {
                "Logistic Regression": lr_model,
                "Random Forest": rf_model,
                "XGBoost": xgb_model
            }

            for idx, (name, model) in enumerate(models_dict.items()):
                with cols[idx]:
                    prob = model.predict_proba(df)[0][1]
                    st.metric(
                        name,
                        f"{prob:.1%}",
                        delta="⚠ High Risk" if prob > 0.5 else "✅ Low Risk"
                    )


