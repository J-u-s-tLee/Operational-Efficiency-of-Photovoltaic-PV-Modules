from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import Data

data = Data.load_data("pv_module_efficiency_dataset.csv")
X_train, X_test, y_train, y_test = Data.split_data(data, "expected_efficiency", "efficiency_level", 0.3, 42, "Linear Regression")

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
r2 = r2_score(y_test, y_pred)
print("R-squared Score:", r2)