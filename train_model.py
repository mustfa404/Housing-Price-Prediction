import pandas as pd
import numpy as np
import pickle
from model import LinearRegressionScratch

data = pd.read_csv("housing.csv")
data.dropna(inplace=True)


X = data.drop("median_house_value", axis=1)
y = data["median_house_value"].values


def preprocess(df):
    df = df.copy()
    df["ocean_proximity"] = df["ocean_proximity"].str.strip()


    for col in ["total_rooms", "total_bedrooms", "population", "households"]:
        df[col] = df[col].clip(lower=1)
        df[col] = np.log(df[col] + 1)


    df["bedroom_ratio"] = df["total_bedrooms"] / df["total_rooms"]
    df["household_rooms"] = df["total_rooms"] / df["households"]


    df = pd.get_dummies(df, columns=["ocean_proximity"])
    return df

X = preprocess(X)


np.random.seed(42)
indices = np.random.permutation(len(X))
split = int(0.8 * len(X))
train_idx, test_idx = indices[:split], indices[split:]

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]
y_train = y[train_idx]
y_test = y[test_idx]


num_cols = ["longitude", "latitude", "housing_median_age", "total_rooms",
            "total_bedrooms", "population", "households", "median_income",
            "bedroom_ratio", "household_rooms"]

X_mean = X_train[num_cols].mean()
X_std = X_train[num_cols].std()

X_train[num_cols] = (X_train[num_cols] - X_mean) / X_std
X_test[num_cols] = (X_test[num_cols] - X_mean) / X_std


model = LinearRegressionScratch(lr=0.01, epochs=3000)
model.fit(X_train.values, y_train)

with open("model.pkl", "wb") as f:
    pickle.dump({
        "model": model,
        "mean": X_mean,
        "std": X_std,
        "columns": X_train.columns
    }, f)

print("✅ Model trained and saved successfully.")

