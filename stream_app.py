{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "f6c06bee-5c71-493b-8518-df5b67859c90",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-07-26 09:24:59.466 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run C:\\Users\\91861\\anaconda\\Lib\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n",
      "2025-07-26 09:24:59.477 Session state does not function when running a script without `streamlit run`\n"
     ]
    }
   ],
   "source": [
    "# stream_app.py\n",
    "\n",
    "import streamlit as st\n",
    "import pandas as pd\n",
    "import pickle\n",
    "\n",
    "# Load model and preprocessing artifacts\n",
    "with open('churn_model.pkl', 'rb') as f:\n",
    "    clf, scaler, feature_names, classes = pickle.load(f)\n",
    "\n",
    "st.title(\"📞 Telco Customer Churn Predictor\")\n",
    "\n",
    "# Collect user input via sidebar\n",
    "st.sidebar.header(\"Enter Customer Details\")\n",
    "input_data = {}\n",
    "for feat in feature_names:\n",
    "    if feat in ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']:\n",
    "        input_data[feat] = st.sidebar.number_input(feat, value=0.0)\n",
    "    else:\n",
    "        # For simplicity, number-encode categorical as integer dropdowns\n",
    "        input_data[feat] = st.sidebar.selectbox(feat, range(len(classes)))\n",
    "\n",
    "df_input = pd.DataFrame([input_data])[feature_names]\n",
    "\n",
    "# Scale input\n",
    "X_scaled = scaler.transform(df_input)\n",
    "\n",
    "if st.sidebar.button(\"Predict Churn\"):\n",
    "    pred = clf.predict(X_scaled)[0]\n",
    "    proba = clf.predict_proba(X_scaled)[0][1]\n",
    "    result = \"✅ No churn\" if pred == 0 else \"⚠ Will churn\"\n",
    "    st.subheader(f\"Prediction: {result}\")\n",
    "    st.write(f\"Churn probability: *{proba:.2%}*\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 0,
   "id": "35793601-44ea-4cff-8a05-aee50e7e1a61",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
