# %%

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# %%
# 1 part a

def split_data(df, target_column, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42):
    """
    Splits the dataset into training, validation, and test sets.
    """
    assert train_size + val_size + test_size == 1, "Splits must sum to 1"

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Split into training and temp set (for validation and test)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(1 - train_size), random_state=random_state)
    
    # Split temp set into validation and test sets
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_size/(test_size + val_size), random_state=random_state)

    return X_train, X_val, X_test, y_train, y_val, y_test


df = pd.read_csv("./Top_spotify_songs.csv")

X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, target_column="popularity")

print("Train set size: "+ str(len(X_train)))
print("Validation set size: " + str(len(X_val)))
print("Test set size: "+ str(len(X_test)))


# %%
# part 1 b

X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, target_column="popularity")

# Print the shapes of the splits
print("X_train shape: " + str(X_train.shape)+", y_train shape: " + str(y_train.shape))
print(f"X_val shape: "+str(X_val.shape) + ", y_val shape: " + str(y_val.shape))
print(f"X_test shape:"+str(X_test.shape) + ", y_test shape: " + str(y_test.shape))

# %%
#part 1 c



def create_correlation_matrix(df, target_column):
    numeric_cols = df.select_dtypes(include=['number']).columns
    df_numeric = df[numeric_cols]
    """
    Creates and saves a correlation matrix between features and the target column.
    """
    correlation_matrix = df_numeric.corr()

    corr_with_target = correlation_matrix[target_column].abs().sort_values(ascending=False)
    top_features = corr_with_target.head(15).index  
    
    reduced_corr_matrix = correlation_matrix.loc[top_features, top_features]

    # Set up the plot size and style
    plt.figure(figsize=(12, 10))
    sns.heatmap(reduced_corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)

    plt.title(f"Correlation Matrix for {target_column}")
    plt.savefig(f"correlation_matrix_{target_column}.png", bbox_inches='tight')
    plt.close()

# Create and save correlation matrix
create_correlation_matrix(df, target_column="popularity")

# %%
# 1 part d

# Select only numeric columns
df_numeric = df.select_dtypes(include=['number'])

# Calculate correlation matrix
corr_matrix = df_numeric.corr()

# Unstack and sort correlation values
corr_matrix_unstacked = corr_matrix.abs().unstack()
sorted_corr = corr_matrix_unstacked.sort_values(ascending=False)

# Filter out self-correlations (the diagonal)
sorted_corr = sorted_corr[sorted_corr < 1]

# Get the top N highly correlated feature pairs (you can change the number as needed)
top_n = 5
top_corr_pairs = sorted_corr.head(top_n)

# Create scatterplots for each pair with the highest correlation
for (feature1, feature2), corr_value in top_corr_pairs.items():
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=df[feature1], y=df[feature2])
    plt.title(f"Scatter plot between {feature1} and {feature2} (Correlation: {corr_value:.2f})")
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.savefig(f"scatterplot_{feature1}_{feature2}.png")  # Save the plot as PNG
    plt.close()  # Close the plot to avoid memory issues

print("Scatterplots saved.")

# %%
#1 part e

# Select target column
target_column = 'popularity'

# 1. Select Features for Model 1: Based on high correlation with target
features_model_1 = ['danceability', 'energy', 'duration_ms']

# 2. Select Features for Model 2: Based on high correlation among features
features_model_2 = ['key', 'loudness', 'speechiness']

# 3. Select Features for Model 3: Random selection of features
features_model_3 = ['acousticness', 'instrumentalness', 'tempo']

def train_model(features):
    # Split data into X (features) and y (target)
    X = df[features]
    y = df[target_column]
    
    # Split data into train and test sets (70% train, 30% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Initialize the linear regression model
    model = LinearRegression()
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = model.predict(X_test)
    
    # Evaluate the model's performance
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, mse, r2

# Train and evaluate 3 models with different feature sets
model_1, mse_1, r2_1 = train_model(features_model_1)
model_2, mse_2, r2_2 = train_model(features_model_2)
model_3, mse_3, r2_3 = train_model(features_model_3)

print(f"Model 1 (Features: {features_model_1})")
print(f"Mean Squared Error: {mse_1:.2f}, R² Score: {r2_1:.2f}")

print(f"\nModel 2 (Features: {features_model_2})")
print(f"Mean Squared Error: {mse_2:.2f}, R² Score: {r2_2:.2f}")

print(f"\nModel 3 (Features: {features_model_3})")
print(f"Mean Squared Error: {mse_3:.2f}, R² Score: {r2_3:.2f}")

# %%
# 1 part f


# Define the target column
target_column = 'popularity' 
features_model_1 = ['danceability', 'energy']
features_model_2 = ['danceability', 'energy', 'loudness']
features_model_3 = ['danceability', 'energy', 'speechiness']

# Split the data into train, validation, and test sets
X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, target_column)

def train_model(features):
    # Select features
    X_train_features = X_train[features]
    
    # Initialize the linear regression model
    model = LinearRegression()
    
    # Train the model
    model.fit(X_train_features, y_train)
    
    # Predict on the training set
    y_train_pred = model.predict(X_train_features)
    
    # Evaluate the model's performance
    mse = mean_squared_error(y_train, y_train_pred)
    r2 = r2_score(y_train, y_train_pred)
    
    return model, mse, r2

# Train models and get the training MSE and R²
model_1, mse_train_1, r2_train_1 = train_model(features_model_1)
model_2, mse_train_2, r2_train_2 = train_model(features_model_2)
model_3, mse_train_3, r2_train_3 = train_model(features_model_3)

# Generate predictions on the training set for plotting
y_train_pred_1 = model_1.predict(X_train[features_model_1])
y_train_pred_2 = model_2.predict(X_train[features_model_2])
y_train_pred_3 = model_3.predict(X_train[features_model_3])

def new_plot(y_true, y_pred, model, features, filename):
    """Creates an improved regression plot with essential information."""
    plt.figure(figsize=(8, 6))
    
    # Scatter plot
    plt.scatter(y_true, y_pred, alpha=0.5)
    
    # Perfect prediction line
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 
             color='red', linestyle='--')
    
    # Add regression equation
    equation = f"y = {model.intercept_:.2f}"
    for i, feature in enumerate(features):
        coefficient = model.coef_[i]
        equation += f" + {coefficient:.2f}*{feature}"
    
    # Add R² value
    r2 = r2_score(y_true, y_pred)
    
    # Add text annotations
    plt.annotate(f"Equation: {equation}", xy=(0.05, 0.95), xycoords='axes fraction')
    plt.annotate(f"R² = {r2:.4f}", xy=(0.05, 0.90), xycoords='axes fraction')
    
    # Labels and title
    plt.xlabel('Actual Popularity')
    plt.ylabel('Predicted Popularity')
    plt.title(f'Model: {", ".join(features)}')
    
    plt.savefig(filename)
    plt.close()

# Plot the training data for each model
new_plot(y_train, y_train_pred_1, model_1, features_model_1, "model_1_training_plot.png")
new_plot(y_train, y_train_pred_2, model_2, features_model_2, "model_2_training_plot.png")
new_plot(y_train, y_train_pred_3, model_3, features_model_3, "model_3_training_plot.png")

# Create a table of training errors
error_table = pd.DataFrame({
    'Model': ['Model 1', 'Model 2', 'Model 3'],
    'Training Error (MSE)': [mse_train_1, mse_train_2, mse_train_3],
    'Training R2': [r2_train_1, r2_train_2, r2_train_3]
})

print(error_table)

# Model 1 performance on validation set
y_val_pred_1 = model_1.predict(X_val[features_model_1])
mse_val_1 = mean_squared_error(y_val, y_val_pred_1)
r2_val_1 = r2_score(y_val, y_val_pred_1)

# Model 2 performance on validation set
y_val_pred_2 = model_2.predict(X_val[features_model_2])
mse_val_2 = mean_squared_error(y_val, y_val_pred_2)
r2_val_2 = r2_score(y_val, y_val_pred_2)

# Model 3 performance on validation set
y_val_pred_3 = model_3.predict(X_val[features_model_3])
mse_val_3 = mean_squared_error(y_val, y_val_pred_3)
r2_val_3 = r2_score(y_val, y_val_pred_3)

# Add validation errors to the table
error_table['Validation Error (MSE)'] = [mse_val_1, mse_val_2, mse_val_3]
error_table['Validation R2'] = [r2_val_1, r2_val_2, r2_val_3]

print(error_table)

# Find the model with the lowest validation error
best_model_index = error_table['Validation Error (MSE)'].idxmin()
best_model = error_table.iloc[best_model_index]

print(f"Best model based on validation MSE: {best_model['Model']}")

# Evaluate the best model on the test set and report the test error
if best_model['Model'] == 'Model 1':
    y_test_pred = model_1.predict(X_test[features_model_1])
elif best_model['Model'] == 'Model 2':
    y_test_pred = model_2.predict(X_test[features_model_2])
else:
    y_test_pred = model_3.predict(X_test[features_model_3])

mse_test = mean_squared_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)

print(f"Test Error (MSE) for {best_model['Model']}: {mse_test}")
print(f"Test R2 for {best_model['Model']}: {r2_test}")



