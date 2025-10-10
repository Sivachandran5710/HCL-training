# %% 
# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# %% 
# --- Load Iris dataset ---
wine = load_wine()

# Convert into DataFrame
df = pd.DataFrame(wine.data, columns=wine.feature_names)

# Add target column
df['target'] = wine.target

# Add flower_name column using lambda
df['flower_name'] = df['target'].apply(lambda x: wine.target_names[x])

# Display first rows
print("Head of dataset:\n", df.head())
print("\nColumn names:", df.columns)

# %% 
# --- Separate by flower type ---
df0 = df[df.target == 0]   # Setosa
df1 = df[df.target == 1]   # Versicolor
df2 = df[df.target == 2]   # Virginica

# --- Scatter plot (Sepal length vs Sepal width) ---
plt.figure(figsize=(8,6))
plt.scatter(df0['sepal length (cm)'], df0['sepal width (cm)'], color='red', label='Setosa')
plt.scatter(df1['sepal length (cm)'], df1['sepal width (cm)'], color='green', label='Versicolor')
plt.scatter(df2['sepal length (cm)'], df2['sepal width (cm)'], color='blue', label='Virginica')
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.title("Iris Sepal Length vs Width")
plt.legend()
plt.show()

# %% 
# --- Train-test split ---
X = df.drop(['target', 'flower_name'], axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# %% 
# --- Train SVM model ---
model = SVC(C=0.1)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
score = model.score(X_test, y_test)
print("\nModel Accuracy:", score)

# %%
