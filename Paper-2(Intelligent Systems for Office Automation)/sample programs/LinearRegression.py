import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression   
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# Load the dataset
df = pd.read_csv('E:\HCL\Paper-2\data.csv')


x = df[ ['Time on Website' , 'Time on App' , 'Length of Membership'] ]
y = df['Yearly Amount Spent']


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(x_train, y_train)

model.score(x_test,y_test)
y_pred = model.predict(x_test)

print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R^2 Score:", r2_score(y_test, y_pred))

plt.scatter(y_test, y_pred)
plt.xlabel("Actual Yearly Amount Spent")
plt.ylabel("Predicted Yearly Amount Spent")
plt.title("Actual vs Predicted Yearly Amount Spent")
plt.show()