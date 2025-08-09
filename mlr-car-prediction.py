#%%
!pip install numpy
!pip install pandas
!pip install matplotlib
!pip install scikit-learn
!pip install seaborn
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
#%%
df = pd.read_csv('Car_Price_Prediction.csv')
df.head(5)
#%%
df_2 = df.copy()
#%%
df_2['Age'] = 2025 - df_2['Year']
df_2.drop(['Year', 'Model'], axis=1, inplace=True)
#%%
df_2['Fuel Type'].unique()
#%%
df_2['Make'].unique()
#%%
df_2['Transmission'].unique()
#%%
fuel_type = {
    'Petrol' : 0,
    'Electric' : 1,
    'Diesel' : 2
}

make = {
    'Honda' : 0,
    'Ford' : 1,
    'BMW' : 2,
    'Audi' : 3,
    'Toyota' : 4
}

transmission = {
    'Manual' : 0,
    'Automatic' : 1
}
#%%
df_2['Fuel Type'] = df['Fuel Type'].map(fuel_type)
df_2['Make'] = df['Make'].map(make)
df_2['Transmission'] = df['Transmission'].map(transmission)
#%%
df_2.head(5)
#%%
df_2.columns
#%%
for i in df_2.columns:
    if i != 'Price':
        plt.figure()
        plt.scatter(x=df_2[i], y=df_2['Price'])
        plt.title(f"{i} vs. Price")
    else :
        continue
#%%
def compute_cost(X, y, w, b):
    m = X.shape[0]
    cost = 0
    for i in range(m):
        f_x = np.dot(X[i], w) + b
        cost += (f_x - y[i]) ** 2
    cost = cost / (2*m)
    return cost
#%%
def compute_gradient(X, y, w, b):
    m,n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0
    for i in range(m):
        f_x = np.dot(X[i], w) + b
        err = f_x - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err * X[ i, j ]
            dj_db += err
    dj_dw /= m
    dj_db /= m
    return dj_dw, dj_db

#%%
def gradient_descent(X, y, w, b, alpha, num_iters):
    w = w.copy()
    b = b
    J_history = []
    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(X, y, w, b)

        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        if i < 100000:
            J_history.append(compute_cost(X, y, w, b))

            if i % math.ceil(num_iters / 10) == 0:
                print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")
    return w, b, J_history
#%%
X = df_2.copy().drop("Price", axis=1)
y = df_2["Price"]
#%%
y
#%%
X_train, X_test, y_train, y_test = X[:int(len(X)*0.80)], X[int(len(X)*0.80):], y[:int(len(y)*0.80)], y[int(len(y)*0.80):]
#%%
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
#%%
n = X_train_scaled.shape[1]

initial_w = np.zeros(n)
initial_b = 0

iterations = 6000
alpha = 1e-3

w_final, b_final, J_hist = gradient_descent(X_train_scaled, y_train, initial_w, initial_b, alpha, iterations)
print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")
#%%
# plot cost versus iteration
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
ax1.plot(J_hist)
ax2.plot(100 + np.arange(len(J_hist[100:])), J_hist[100:])
ax1.set_title("Cost vs. iteration");  ax2.set_title("Cost vs. iteration (tail)")
ax1.set_ylabel('Cost')             ;  ax2.set_ylabel('Cost')
ax1.set_xlabel('iteration step')   ;  ax2.set_xlabel('iteration step')
plt.show()
#%%
y_test
#%%
def predict(X, w, b):
    f_x = np.dot(X, w) + b
    return f_x
#%%
X_test_scaled  = scaler.transform(X_test)
#%%
y_true = y_test
y_pred = predict(X_test_scaled, w_final, b_final)

# RMSE
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

# R² score
r2 = r2_score(y_true, y_pred)

print(f"RMSE: {rmse:.4f}")
print(f"R²:   {r2:.4f}")
#%%
X_features = df_2.drop(columns=['Price']).columns
#%%
# Make predictions on the training set
y_pred_train = predict(X_train_scaled, w_final, b_final)

fig, ax = plt.subplots(1, 4, figsize=(15, 3), sharey=True)

for i in range(4):  # loop over first 4 features
    ax[i].scatter(X_train_scaled[:, i], y_train, label='Target', alpha=0.6)
    ax[i].scatter(X_train_scaled[:, i], y_pred_train, color='orange', label='Predict', alpha=0.6)
    ax[i].set_xlabel(X_features[i])

ax[0].set_ylabel("Price")
ax[0].legend()
fig.suptitle("Target vs Prediction using z-score normalized model")
plt.show()

#%%
# Training predictions
y_pred_train = predict(X_train_scaled, w_final, b_final)
r2_train = r2_score(y_train, y_pred_train)

# Test predictions
y_pred_test = predict(X_test_scaled, w_final, b_final)
r2_test = r2_score(y_test, y_pred_test)

print(f"Training R²: {r2_train:.4f}")
print(f"Test R²:     {r2_test:.4f}")

#%%
# Compute correlation matrix
corr = df_2.corr()

# Plot
plt.figure(figsize=(10, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlation Matrix")
plt.show()

# Sort correlation with Price
print("Correlation with Price:")
print(corr["Price"].sort_values(ascending=False))