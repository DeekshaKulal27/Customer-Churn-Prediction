{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "d6548779-1166-47b8-be46-33af6809611a",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\91861\\AppData\\Local\\Temp\\ipykernel_10204\\1089736815.py:15: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.\n",
      "The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.\n",
      "\n",
      "For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.\n",
      "\n",
      "\n",
      "  df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.84      0.85      0.85      1035\n",
      "           1       0.58      0.57      0.57       374\n",
      "\n",
      "    accuracy                           0.78      1409\n",
      "   macro avg       0.71      0.71      0.71      1409\n",
      "weighted avg       0.77      0.78      0.78      1409\n",
      "\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.preprocessing import LabelEncoder, StandardScaler\n",
    "from sklearn.metrics import classification_report\n",
    "from imblearn.over_sampling import SMOTE\n",
    "import pickle\n",
    "\n",
    "# 1. Load data\n",
    "df = pd.read_csv(r\"C:\\Users\\91861\\OneDrive\\deeeeeeeeeeeeee\\OneDrive\\문서\\WA_Fn-UseC_-Telco-Customer-Churn.csv\")\n",
    "\n",
    "# 2. Clean TotalCharges column\n",
    "df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')\n",
    "df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)\n",
    "\n",
    "# 3. Drop unnecessary columns\n",
    "df = df.drop(['customerID'], axis=1)\n",
    "\n",
    "# 4. Encode categorical\n",
    "le = LabelEncoder()\n",
    "for col in df.select_dtypes(include='object').columns:\n",
    "    df[col] = le.fit_transform(df[col])\n",
    "\n",
    "# 5. Split features and target\n",
    "X = df.drop('Churn', axis=1)\n",
    "y = df['Churn']\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y,\n",
    "                                                    test_size=0.2, random_state=42,\n",
    "                                                    stratify=y)\n",
    "\n",
    "# 6. Handle imbalance using SMOTE\n",
    "smote = SMOTE(random_state=42)\n",
    "X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)\n",
    "\n",
    "# 7. Scale numeric features\n",
    "scaler = StandardScaler()\n",
    "X_train_sm = scaler.fit_transform(X_train_sm)\n",
    "X_test = scaler.transform(X_test)\n",
    "\n",
    "# 8. Train model\n",
    "clf = RandomForestClassifier(random_state=42)\n",
    "clf.fit(X_train_sm, y_train_sm)\n",
    "\n",
    "# 9. Evaluate\n",
    "y_pred = clf.predict(X_test)\n",
    "print(classification_report(y_test, y_pred))\n",
    "\n",
    "# 10. Save artifacts\n",
    "with open('churn_model.pkl', 'wb') as f:\n",
    "    pickle.dump((clf, scaler, X.columns.tolist(), le.classes_), f)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "53393576-d379-4aec-b552-f9b978951c34",
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
