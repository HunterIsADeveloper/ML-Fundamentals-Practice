# Use Least Squares to fit a linear regression model
# Calculate R-squared to evaluate the model
# Calculate a P-value to assess significance

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

ds = pd.read_csv('SOCR-HeightWeight.csv')
ds = ds.sample(frac=0.02, random_state=42)

plt.style.use('seaborn-v0_8-darkgrid')

#generate x and y data
x = np.array(ds['Height(Inches)']) 
y = np.array(ds['Weight(Pounds)']) 

def least_squares(x, y):
    # Create design matrix A by stacking x and ones
    A = np.vstack((x, np.ones_like(x))).T
    Y = y[:, np.newaxis]  # reshape y to column vector

    # Direct least square regression
    alpha = np.dot((np.dot(np.linalg.inv(np.dot(A.T,A)),A.T)),Y)
    return alpha

# r^2 = (Var(mean) - Var(residuals)) / Var(mean)
def r_squared(x, y, alpha):
    y_pred = alpha[0]*x + alpha[1]
    ss_total = np.sum((y - np.mean(y))**2)
    ss_residual = np.sum((y - y_pred.flatten())**2)
    r2 = 1 - (ss_residual / ss_total)
    return r2

def p_value(n, k=1):
    # Calculate Sum of Squares
    y_pred = alpha[0]*x + alpha[1]
    ss_total = np.sum((y - np.mean(y))**2)
    ss_reg = np.sum((y_pred.flatten() - np.mean(y))**2) 
    ss_residual = np.sum((y - y_pred.flatten())**2) 
    
    # Calculate Mean Squares
    ms_reg = ss_reg / k 
    ms_residual = ss_residual / (n - k - 1) 
    
    # Calculate F-statistic
    f_stat = ms_reg / ms_residual
    
    # Calculate p-value using F-distribution
    p_val = 1 - stats.f.cdf(f_stat, k, n - k - 1)
    return p_val, f_stat

# Calculate statistics
alpha = least_squares(x, y)
r2 = r_squared(x, y, alpha)
n = len(x)
p, f_stat = p_value(n)

print(f"Sample size: {n}")
print(f"Correlation strength (R²): {float(r2):.4f}")
print(f"Statistical significance (p-value): {float(p):.16f}") 
print(f"Slope: {float(alpha[0][0]):.4f} pounds/inch")
print(f"Intercept: {float(alpha[1][0]):.4f} pounds")



# plot the results
# plt.figure(figsize = (10,8))
# plt.plot(x, y, 'b.')
# plt.plot(x, alpha[0]*x + alpha[1], 'r')
# plt.xlabel('Height(Inches)')
# plt.ylabel('Weight(Pounds)')
# plt.show()

