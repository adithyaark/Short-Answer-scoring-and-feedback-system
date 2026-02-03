from sklearn.ensemble import RandomForestRegressor
import joblib
import numpy as np

X = np.random.rand(100, 10)
y = np.random.rand(100)

model = RandomForestRegressor()
model.fit(X, y)

joblib.dump(model, "random_forest_model.pkl")
