# data_pipeline.py

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# -------------------------
# Phase 1: Load CSV for Numerical Model Only
# -------------------------
df1 = pd.read_csv("phase1_data.csv")  # <-- CSV must contain Social_media_followers, Sold_out

X1 = df1[['Social_media_followers']]
y1 = df1['Sold_out']

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.3, random_state=19)

pipe1 = make_pipeline(
    SimpleImputer(strategy='mean'),
    LogisticRegression()
)

pipe1.fit(X1_train, y1_train)
print("Simple Model Train Accuracy:", pipe1.score(X1_train, y1_train))
print("Simple Model Test Accuracy:", pipe1.score(X1_test, y1_test))

# -------------------------
# Phase 2: Load CSV for Full Pipeline with Categorical + Numerical
# -------------------------
df2 = pd.read_csv("phase2_data.csv")  # <-- CSV must contain Genre, Social_media_followers, Sold_out

X = df2[['Genre', 'Social_media_followers']]
y = df2['Sold_out']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=75)

num_cols = ['Social_media_followers']
cat_cols = ['Genre']

num_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

col_transformer = ColumnTransformer(transformers=[
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols)
])

full_pipeline = make_pipeline(col_transformer, DecisionTreeClassifier())

full_pipeline.fit(X_train, y_train)
print("Full Pipeline Test Accuracy:", full_pipeline.score(X_test, y_test))

joblib.dump(full_pipeline, "pipe.joblib")

loaded_pipeline = joblib.load("pipe.joblib")
print("Loaded Pipeline Test Accuracy:", loaded_pipeline.score(X_test, y_test))