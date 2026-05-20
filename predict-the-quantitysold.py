#To predict the column **Quantity_Sold_(kilo)** for a dataset of supermarket sales.
#Preprocess the Datasets

#Importing the necessary libraries
# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

#Load both *labeled_data.csv* and *unlabeled_data.csv*.
# %%
labeled_df = pd.read_csv('labeled_data.csv')
unlabeled_df = pd.read_csv('unlabeled_data.csv')

#Save ID separately for the unlabeled dataset since we need it for the submission file.
unlabeled_ids = unlabeled_df["ID"]

#Drop ID from features since it is not a useful feature for prediction and it may cause overfitting if we keep it in the dataset. The ID column is just a unique identifier for each row and does not contain any meaningful information that can help the model to learn the relationship between the features and the target variable.
labeled_df = labeled_df.drop(columns=["ID"], errors="ignore")
unlabeled_df = unlabeled_df.drop(columns=["ID"], errors="ignore")

#Basic cleaning from the labeled dataset and the unlabeled dataset.
# %%
#Check whether the Item Name_x and Item Name_y columns are the same. if they are the same, we can drop one of them from both datasets.
if labeled_df["Item Name_x"].equals(labeled_df["Item Name_y"]):
    labeled_df = labeled_df.drop(columns=["Item Name_y"])

if unlabeled_df["Item Name_x"].equals(unlabeled_df["Item Name_y"]):
    unlabeled_df = unlabeled_df.drop(columns=["Item Name_y"])

#Fixing the negative values in the "Unit Selling Price (RMB/kg)" column by taking the absolute value.
# %%
labeled_df["Unit Selling Price (RMB/kg)"] = labeled_df["Unit Selling Price (RMB/kg)"].abs()
unlabeled_df["Unit Selling Price (RMB/kg)"] = unlabeled_df["Unit Selling Price (RMB/kg)"].abs()

# Remove invalid targets from the labeled dataset by filtering out rows where "Quantity_Sold_(kilo)" is negative.
# %%
labeled_df = labeled_df[labeled_df["Quantity_Sold_(kilo)"] >= 0]
#Split *labeled_data.csv* into training and validation sets.
# %%  
x= labeled_df.drop(columns=['Quantity_Sold_(kilo)'])
y= labeled_df['Quantity_Sold_(kilo)']

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

#Explore the datasets to understand the distribution of the features and the target variable, and to identify any potential issues such as missing values or outliers. This can help us to make informed decisions about how to preprocess the data and which features to engineer.
# %%
print(x_train.describe())
print(y_train.describe())

#Fix the missing values in the wholesale price column by filling them with the mean value of the column.
# %%
mean_wholesale_price = x_train["Wholesale Price (RMB/kg)"].mean()
x_train["Wholesale Price (RMB/kg)"] = x_train["Wholesale Price (RMB/kg)"].fillna(mean_wholesale_price)
x_val["Wholesale Price (RMB/kg)"] = x_val["Wholesale Price (RMB/kg)"].fillna(mean_wholesale_price)  
unlabeled_df["Wholesale Price (RMB/kg)"] = unlabeled_df["Wholesale Price (RMB/kg)"].fillna(mean_wholesale_price)

#Check for outliers in the "Unit Selling Price (RMB/kg)" and "Quantity_Sold_(kilo)" columns using box plots. If there are any outliers, we can consider removing them or transforming the data to reduce their impact on the model.
# %%
plt.figure()
plt.boxplot(x_train["Unit Selling Price (RMB/kg)"])
plt.title(f"Outliers: Unit Selling Price (RMB/kg)")
plt.show()

plt.figure()
plt.boxplot(y_train)
plt.title("Outliers: Quantity_Sold_(kilo)")
plt.show()

#Feature engineering: Create a new feature called "Price Difference" by subtracting the "Wholesale Price (RMB/kg)" from the "Unit Selling Price (RMB/kg)". 
#From this we can get the difference between the selling price and the wholesale price, which means if the sale price is too high then the quantity sold will be low, and if the sale price is too low then the quantity sold will be high. This feature can help the model to learn the relationship between the price and the quantity sold.
# %%
x_train["Price Difference"] = x_train["Unit Selling Price (RMB/kg)"] - x_train["Wholesale Price (RMB/kg)"]
x_val["Price Difference"] = x_val["Unit Selling Price (RMB/kg)"] - x_val["Wholesale Price (RMB/kg)"]
unlabeled_df["Price Difference"] = unlabeled_df["Unit Selling Price (RMB/kg)"] - unlabeled_df["Wholesale Price (RMB/kg)"]

#Create a new feature called "Profit ratio" by dividing the "Unit Selling Price (RMB/kg)" by the "Wholesale Price (RMB/kg)". 
#From this we can get the profit ratio, which means how many times the selling price is compared to the wholesale price.
# %%
x_train["Profit Ratio"] = x_train["Unit Selling Price (RMB/kg)"] / (x_train["Wholesale Price (RMB/kg)"] + 1e-5)
x_val["Profit Ratio"] = x_val["Unit Selling Price (RMB/kg)"] / (x_val["Wholesale Price (RMB/kg)"] + 1e-5)
unlabeled_df["Profit Ratio"] = unlabeled_df["Unit Selling Price (RMB/kg)"] / (unlabeled_df["Wholesale Price (RMB/kg)"] + 1e-5)

#Create a loss-adjusted margin feature using the loss rate.
#This estimates how much margin remains after accounting for the percentage of goods that may be lost.
# %%
loss_rate_train = 1 - (x_train["Loss Rate (%)"] / 100)
loss_rate_val = 1 - (x_val["Loss Rate (%)"] / 100)
loss_rate_unlabeled = 1 - (unlabeled_df["Loss Rate (%)"] / 100)

x_train["Loss Adjusted Margin"] = (x_train["Unit Selling Price (RMB/kg)"] * loss_rate_train) - x_train["Wholesale Price (RMB/kg)"]
x_val["Loss Adjusted Margin"] = (x_val["Unit Selling Price (RMB/kg)"] * loss_rate_val) - x_val["Wholesale Price (RMB/kg)"]
unlabeled_df["Loss Adjusted Margin"] = (unlabeled_df["Unit Selling Price (RMB/kg)"] * loss_rate_unlabeled) - unlabeled_df["Wholesale Price (RMB/kg)"]

#Encode the Item Name_x column using target encoding, which replaces each category with the mean of the target variable for that category. This can help the model to learn the relationship between the item name and the quantity sold.
# %%
target_mean = x_train.join(y_train).groupby("Item Name_x")["Quantity_Sold_(kilo)"].mean()

x_train["Item_encoded"] = x_train["Item Name_x"].map(target_mean)
x_val["Item_encoded"] = x_val["Item Name_x"].map(target_mean)
unlabeled_df["Item_encoded"] = unlabeled_df["Item Name_x"].map(target_mean)

print(x_train["Item_encoded"].isnull().sum())  
print(x_val["Item_encoded"].isnull().sum())  
print(unlabeled_df["Item_encoded"].isnull().sum()) 

#Fixing the missing values in the "Item_encoded" column by filling them with the global mean of the target variable. This can help the model to learn the relationship between the item name and the quantity sold, even for the items that are not present in the training set.
#%%
global_mean = y_train.mean()

x_val["Item_encoded"] = x_val["Item_encoded"].fillna(global_mean)
unlabeled_df["Item_encoded"] = unlabeled_df["Item_encoded"].fillna(global_mean)

#Drop the original "Item Name_x" column from the datasets since we have already encoded it and it is no longer needed. This can help to reduce the dimensionality of the dataset and prevent overfitting.
#%%
x_train = x_train.drop(columns=["Item Name_x"], errors="ignore")
x_val = x_val.drop(columns=["Item Name_x"], errors="ignore")
unlabeled_df = unlabeled_df.drop(columns=["Item Name_x"], errors="ignore")

#Also Category Name column is a categorical feature, so we need to encode it with Target Encoding. This can help the model to learn the relationship between the category name and the quantity sold, which can be useful for making predictions on the validation and unlabeled datasets.
# %%
category_mean = x_train.join(y_train).groupby("Category Name")["Quantity_Sold_(kilo)"].mean()

x_train["Category_encoded"] = x_train["Category Name"].map(category_mean)
x_val["Category_encoded"] = x_val["Category Name"].map(category_mean).fillna(y_train.mean())
unlabeled_df["Category_encoded"] = unlabeled_df["Category Name"].map(category_mean).fillna(y_train.mean())

print(x_train["Category_encoded"].isnull().sum())  
print(x_val["Category_encoded"].isnull().sum())  
print(unlabeled_df["Category_encoded"].isnull().sum()) 

x_train = x_train.drop(columns=["Category Name"], errors="ignore")
x_val = x_val.drop(columns=["Category Name"], errors="ignore")
unlabeled_df = unlabeled_df.drop(columns=["Category Name"], errors="ignore")

#Also Weekday column is a categorical feature, so we need to encode it as well with one-hot encoding.
# %%
x_train = pd.get_dummies(x_train, columns=["Weekday"])
x_val = pd.get_dummies(x_val, columns=["Weekday"])
unlabeled_df = pd.get_dummies(unlabeled_df, columns=["Weekday"])

#After one-hot encoding, different datasets may have different columns. To fix this, we need to reindex the columns of the validation and unlabeled datasets to match the columns of the training dataset, filling any missing columns with zeros. This can help to ensure that the model can make predictions on the validation and unlabeled datasets without any issues.
x_val = x_val.reindex(columns=x_train.columns, fill_value=0)
unlabeled_df = unlabeled_df.reindex(columns=x_train.columns, fill_value=0)

#Training the model
#Using a simple linear regression model to predict the quantity sold based on the features we have engineered and encoded.
# %%
model1 = LinearRegression()
model1.fit(x_train, y_train)
y_pred = model1.predict(x_val)
mse = mean_squared_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)
print("Mean Squared Error (Linear Regression):", mse)
print("R^2 Score (Linear Regression):", r2)
#The Linear Regression model achieved a higher MSE, which indicates that it may not be the best model for this dataset. This could be due to the fact that the relationship between the features and the target variable is not linear. It maybe non linear, which is why the Random Forest model performed better.
#Using a Random Forest Regressor model to predict the quantity sold based on the features we have engineered and encoded.
# %%
model2 = RandomForestRegressor(
    n_estimators=700,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features="log2",
    random_state=42,
    n_jobs=-1
)

model2.fit(x_train, y_train)

# ── Results ──────────────────────────────────────────
y_train_pred = model2.predict(x_train)
train_r2  = r2_score(y_train, y_train_pred)
train_mse = mean_squared_error(y_train, y_train_pred)

y_pred   = model2.predict(x_val)
val_r2   = r2_score(y_val, y_pred)
val_mse  = mean_squared_error(y_val, y_pred)

baseline_mse = mean_squared_error(y_val, np.full(len(y_val), y_train.mean()))

print("=== Results ===")
print(f"Baseline MSE : {baseline_mse:.4f}")
print(f"Train R²     : {train_r2:.4f}  |  Train MSE : {train_mse:.4f}")
print(f"Val R²       : {val_r2:.4f}  |  Val MSE   : {val_mse:.4f}")

importances = pd.Series(model2.feature_importances_, index=x_train.columns)
print("\n=== Top 10 Features ===")
print(importances.sort_values(ascending=False).head(10))
# ─────────────────────────────────────────────────────
