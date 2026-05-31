from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
from model import LinearRegressionScratch

app = Flask(__name__)


with open("model.pkl", "rb") as f:
    saved = pickle.load(f)

model = saved["model"]
mean = saved["mean"]
std = saved["std"]
columns = saved["columns"]


def preprocess_input(df):
    df = df.copy()
    df["ocean_proximity"] = df["ocean_proximity"].str.strip()


    for col in ["total_rooms", "total_bedrooms", "population", "households"]:
        df[col] = df[col].clip(lower=1)
        df[col] = np.log(df[col] + 1)


    df["bedroom_ratio"] = df["total_bedrooms"] / df["total_rooms"]
    df["household_rooms"] = df["total_rooms"] / df["households"]


    df = pd.get_dummies(df, columns=["ocean_proximity"])
    return df

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None
    if request.method == "POST":
        try:

            user_data = {
                "longitude": float(request.form.get("longitude", 0)),
                "latitude": float(request.form.get("latitude", 0)),
                "housing_median_age": float(request.form.get("age", 0)),
                "total_rooms": float(request.form.get("rooms", 1)),
                "total_bedrooms": float(request.form.get("bedrooms", 1)),
                "population": float(request.form.get("population", 1)),
                "households": float(request.form.get("households", 1)),
                "median_income": float(request.form.get("income", 0.1)),
                "ocean_proximity": request.form.get("ocean", "INLAND")
            }

            df_user = pd.DataFrame([user_data])
            df_user = preprocess_input(df_user)


            df_user = df_user.reindex(columns=columns, fill_value=0)


            num_cols = ["longitude", "latitude", "housing_median_age", "total_rooms",
                        "total_bedrooms", "population", "households", "median_income",
                        "bedroom_ratio", "household_rooms"]
            df_user[num_cols] = (df_user[num_cols] - mean) / std


            prediction = model.predict(df_user.values)[0]

        except Exception as e:
            error = "Invalid input. Please check your values."
            print("Error:", e)

    return render_template("index.html", prediction=prediction, error=error)

if __name__ == "__main__":
    app.run(debug=True)
