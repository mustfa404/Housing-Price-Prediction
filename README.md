
# House Price Prediction Web Application

## Overview

This project is a Flask-based web application that predicts California house prices using a custom Linear Regression model implemented from scratch. Users can enter housing-related features through a web interface, and the application returns an estimated house price.

## Features

* Interactive web interface built with Flask
* Custom Linear Regression model (implemented from scratch)
* Input preprocessing and feature engineering
* Automatic categorical encoding using one-hot encoding
* Standardized numerical features
* Real-time house price prediction

---

## Project Structure

```text
project/
│
├── app.py                 # Flask application
├── model.py               # Custom LinearRegressionScratch class
├── model.pkl              # Trained model and preprocessing data
├── templates/
│   └── index.html         # Frontend HTML template
├── static/                # CSS, JS, images (optional)
└── README.md
```

---

## Dataset Features

The model uses the following input features:

| Feature            | Description                  |
| ------------------ | ---------------------------- |
| longitude          | Longitude coordinate         |
| latitude           | Latitude coordinate          |
| housing_median_age | Median age of houses         |
| total_rooms        | Total number of rooms        |
| total_bedrooms     | Total number of bedrooms     |
| population         | Population in the block      |
| households         | Number of households         |
| median_income      | Median household income      |
| ocean_proximity    | Distance from ocean category |

---

## Preprocessing Pipeline

Before prediction, the application performs:

### 1. Data Cleaning

* Removes extra spaces from `ocean_proximity`
* Prevents zero values in selected numerical columns

### 2. Log Transformation

Applied to:

* total_rooms
* total_bedrooms
* population
* households

Formula:

```python
np.log(x + 1)
```

### 3. Feature Engineering

Creates:

#### Bedroom Ratio

```python
bedroom_ratio = total_bedrooms / total_rooms
```

#### Household Rooms

```python
household_rooms = total_rooms / households
```

### 4. One-Hot Encoding

Converts the categorical feature:

```text
ocean_proximity
```

into binary columns.

### 5. Standardization

Numerical features are normalized using the mean and standard deviation saved during training.

---

## Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/house-price-prediction.git

cd house-price-prediction
```

### Create Virtual Environment

```bash
python -m venv venv
```

Activate:

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / macOS

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install flask pandas numpy
```

---

## Running the Application

Start the Flask server:

```bash
python app.py
```

The application will be available at:

```text
http://127.0.0.1:5000/
```

---

## Model Loading

The application loads a serialized model from:

```python
model.pkl
```

The pickle file contains:

```python
{
    "model": trained_model,
    "mean": feature_means,
    "std": feature_stds,
    "columns": training_columns
}
```

---

## Prediction Workflow

1. User submits housing information.
2. Input data is converted to a Pandas DataFrame.
3. Preprocessing and feature engineering are applied.
4. Features are aligned with training columns.
5. Data is standardized.
6. Model predicts house value.
7. Result is displayed on the webpage.

---

## Error Handling

The application catches invalid user inputs and displays:

```text
Invalid input. Please check your values.
```

instead of crashing.

---

## Technologies Used

* Python
* Flask
* Pandas
* NumPy
* Pickle
* HTML/CSS
* Custom Linear Regression Implementation

---

## Future Improvements

* Add model confidence intervals
* Improve UI/UX design
* Input validation on frontend
* Docker deployment
* Cloud deployment (Render, Railway, AWS, Azure)
* Support multiple machine learning models

---

## Author

Developed as a Machine Learning and Flask deployment project for house price prediction.
