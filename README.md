Car Price Prediction – Linear Regression with Gradient Descent
📌 Project Overview
This project implements a multivariate linear regression model from scratch using gradient descent to predict car prices based on various features such as make, fuel type, transmission, mileage, engine size, and vehicle age.
The model is trained and evaluated on a provided dataset, with preprocessing steps like feature encoding and standardization included.

📂 Dataset
The dataset file used is:

Copy
Edit
Car_Price_Prediction.csv
Original Features:
Make – Car manufacturer (categorical)

Model – Specific model (categorical, dropped in preprocessing)

Year – Manufacturing year (numeric, used to compute age)

Transmission – Manual or Automatic (categorical)

Mileage – Vehicle mileage (numeric)

Engine Size – Engine capacity in liters (numeric)

Fuel Type – Petrol, Diesel, Electric (categorical)

Price – Target variable (numeric, price in currency units)

🔄 Data Preprocessing
Feature Engineering

Created new feature:

ini
Copy
Edit
Age = 2025 - Year
Dropped Year and Model as they are not directly needed after transformation.

Categorical Encoding

Fuel Type:

nginx
Copy
Edit
Petrol → 0
Electric → 1
Diesel → 2
Make:

nginx
Copy
Edit
Honda → 0, Ford → 1, BMW → 2, Audi → 3, Toyota → 4
Transmission:

mathematica
Copy
Edit
Manual → 0, Automatic → 1
Feature Scaling

Applied StandardScaler to numerical and encoded features before training.

🧮 Model Implementation
Cost Function: Mean Squared Error (MSE)

Gradient Calculation: Manual implementation of partial derivatives with respect to weights and bias

Optimization: Batch Gradient Descent

Gradient Descent Parameters:
Initial weights: all zeros

Learning rate (α): 0.001

Iterations: 6000

📊 Training & Evaluation
Train-Test Split:

80% Training data

20% Test data

Performance:

Training R²: 0.8354

Test R²: 0.8544

Test RMSE: printed during evaluation

📈 Visualizations
The project includes:

Scatter plots of each feature vs. Price

Cost function vs. iterations (full and tail view)

Predicted vs. actual prices for training data across first 4 features

🚀 How to Run
Install dependencies:

bash
Copy
Edit
pip install numpy pandas matplotlib scikit-learn
Place Car_Price_Prediction.csv in the same directory as the script.

Run the Python script or Jupyter Notebook cells in order.

📌 Key Takeaways
Implemented Linear Regression from scratch without using sklearn's built-in regression model.

Achieved high R² scores on both training and testing sets, indicating good generalization.

Demonstrated feature encoding, scaling, and gradient descent optimization on a real dataset.
