import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load dataset
df = pd.read_csv("final_scout_not_dummy.csv")

# Select better features (ADD MORE HERE)
df = df[[
    "make_model",
    "body_type",
    "hp_kW",
    "age",
    "km",
    "Gears",
    "Gearing_Type",
    "Fuel",
    "Previous_Owners",
    "Displacement_cc",
    "Weight_kg",
    "Drive_chain",
    "price"
]]

# Remove missing values
df = df.dropna()

# Convert categorical → numeric
df = pd.get_dummies(df)

X = df.drop("price", axis=1)
y = df["price"]

# Better model
model = RandomForestRegressor(n_estimators=200, max_depth=10)
model.fit(X, y)

# Save model
pickle.dump((model, X.columns), open("Auto_Price_Pred_Model.pkl", "wb"))

print("Improved model created!")