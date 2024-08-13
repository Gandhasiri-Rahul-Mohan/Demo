import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px

# Function to load data from Excel
def load_data(file_path):
    return pd.read_excel(file_path)

# Function to perform initial data exploration
def explore_data(df):
    print(df.head())
    print(df.tail())
    print(df.info())
    print(df.isnull().sum())

# Function to perform KNN imputation
def knn_imputation(df, features):
    imputer = KNNImputer(n_neighbors=5)
    imputed_data = imputer.fit_transform(df[features])
    df[features] = pd.DataFrame(imputed_data, columns=features)
    return df

# Function to calculate and plot the correlation matrix
def plot_correlation_matrix(df, features):
    correlation_matrix = df[features].corr()
    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.show()

# Function to fit and summarize an OLS regression model
def fit_ols_model(df, independent_vars, dependent_var):
    X = df[independent_vars]
    Y = df[dependent_var]
    X = sm.add_constant(X)
    model = sm.OLS(Y, X).fit()
    print(model.summary())
    return model

# Function to split the data and train the Linear Regression model
def train_linear_model(df, independent_vars, dependent_var, test_size=0.15):
    X = df[independent_vars]
    Y = df[dependent_var]
    X = sm.add_constant(X)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size)
    model = LinearRegression()
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    return Y_test, y_pred

# Function to evaluate the model
def evaluate_model(Y_test, y_pred):
    mse = mean_squared_error(Y_test, y_pred)
    r2 = r2_score(Y_test, y_pred)
    rmse = np.sqrt(mse).round(2)
    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")
    print(f"Root Mean Square Error: {rmse}")
    return mse, r2, rmse

# Function to plot actual vs predicted values
def plot_actual_vs_predicted(Y_test, y_pred):
    fig = px.scatter(x=Y_test, y=y_pred, 
                     labels={'x': 'Actual Sale Count', 'y': 'Predicted Sale Count'}, 
                     title='Actual vs. Predicted Sale Count')
    fig.add_shape(type='line',
                  x0=Y_test.min(), y0=Y_test.min(),
                  x1=Y_test.max(), y1=Y_test.max(),
                  line=dict(color='Red', dash='dash'))
    fig.show()

# Main function to execute the workflow
def main(file_path):
    df = load_data()
    explore_data(df)
    
    features_for_imputation = ['Supplier Discount (%)', 'Sale Count']
    df = knn_imputation(df, features_for_imputation)
    
    features_for_correlation = ['Sale Count', 'Stock in Liters', 'Price (INR per Liter)', 
                                'Production Count', 'Supplier Discount (%)', 'Forex Rate (INR to USD)']
    plot_correlation_matrix(df, features_for_correlation)
    
    independent_vars = df.columns.drop(['Date of Procurement', 'Production Count'])
    dependent_var = 'Production Count'
    
    fit_ols_model(df, independent_vars, dependent_var)
    
    Y_test, y_pred = train_linear_model(df, independent_vars, dependent_var)
    
    evaluate_model(Y_test, y_pred)
    
    plot_actual_vs_predicted(Y_test, y_pred)

# Run the main function
# main("Sample Data 1.xlsx")
