import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# Load model and selected features
model = joblib.load('rf_churn_model_tuned.pkl')
selected_features = joblib.load('feature_list.pkl')

# Page config
st.set_page_config(page_title="Churn Prediction", layout="wide")
st.title("📊 Customer Churn Prediction Dashboard")

st.sidebar.header("📁 Upload File")
uploaded_file = st.sidebar.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx", "xls"])

# Instructions
with st.expander("📘 How to Use"):
    st.markdown("""
    - Upload a CSV or Excel file containing customer data.
    - The app will automatically use the correct features.
    - It shows churn predictions, feature importance, and SHAP explanations.
    """)

# Handle file upload
if uploaded_file:
    try:
        # Detect and read file
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            st.stop()

        st.success("✅ File uploaded successfully!")
        st.markdown("### 🔍 Data Preview")
        st.dataframe(df.head())

        # Match required features
        matched_features = [f for f in selected_features if f in df.columns]
        missing_features = list(set(selected_features) - set(matched_features))

        st.markdown(f"### ✅ Matched {len(matched_features)} / 14 required features")
        if missing_features:
            st.warning(f"Missing features: {', '.join(missing_features)}")

        # Continue if sufficient features
        if len(matched_features) >= 10:
            # Predict churn
            probs = model.predict_proba(df[matched_features])[:, 1]
            preds = (probs >= 0.40).astype(int)

            df['Churn_Probability'] = probs
            df['Churn_Prediction'] = preds

            st.markdown("### 🎯 Prediction Results")
            st.dataframe(df[['Churn_Prediction', 'Churn_Probability'] + matched_features].head(10))

            # Churn distribution chart
            st.markdown("### 📈 Churn Prediction Distribution")
            st.bar_chart(df['Churn_Prediction'].value_counts())

            # Feature importance plot
            st.markdown("### 📊 Feature Importance (Model-Based)")
            importances = model.feature_importances_
            fi_df = pd.DataFrame({'Feature': selected_features, 'Importance': importances})
            fi_df = fi_df[fi_df['Feature'].isin(matched_features)].sort_values(by='Importance', ascending=True)

            fig, ax = plt.subplots()
            ax.barh(fi_df['Feature'], fi_df['Importance'], color='skyblue')
            ax.set_xlabel("Importance")
            ax.set_title("Feature Importance")
            st.pyplot(fig)

            # SHAP Explanation for first row
            if len(matched_features) >= 5:
                st.markdown("### 🧠 SHAP Explanation (First Customer)")
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(df[matched_features])
                shap.initjs()
                st.pyplot(shap.force_plot(explainer.expected_value[1],
                                          shap_values[1][0],
                                          df[matched_features].iloc[0],
                                          matplotlib=True))
            else:
                st.info("SHAP requires at least 5 matched features.")

            # Download button
            st.markdown("### ⬇️ Download Predictions")
            st.download_button(
                label="Download CSV",
                data=df.to_csv(index=False).encode('utf-8'),
                file_name='churn_predictions.csv',
                mime='text/csv'
            )
        else:
            st.error("❌ Too few required features. Please include at least 10 of the 14 key features.")

    except Exception as e:
        st.error(f"⚠️ Error processing file: {e}")

else:
    st.info("Please upload a file to start.")
