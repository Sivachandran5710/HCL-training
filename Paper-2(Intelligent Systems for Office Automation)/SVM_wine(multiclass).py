# %% 
# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# %% 
# --- Load Wine dataset ---
wine = load_wine()

# Convert into DataFrame
df = pd.DataFrame(wine.data, columns=wine.feature_names)

# Add target column
df['target'] = wine.target

# Add wine class name column using lambda
df['wine_class'] = df['target'].apply(lambda x: wine.target_names[x])

# Display first rows
print("Head of dataset:\n", df.head())
print("\nColumn names:", df.columns)

# %% 
# --- Separate by wine type ---
df0 = df[df.target == 0]   # class_0
df1 = df[df.target == 1]   # class_1
df2 = df[df.target == 2]   # class_2

# --- Scatter plot (Alcohol vs Color Intensity) ---
plt.figure(figsize=(8,6))
plt.scatter(df0['alcohol'], df0['color_intensity'], color='red', label=wine.target_names[0])
plt.scatter(df1['alcohol'], df1['color_intensity'], color='green', label=wine.target_names[1])
plt.scatter(df2['alcohol'], df2['color_intensity'], color='blue', label=wine.target_names[2])
plt.xlabel("Alcohol")
plt.ylabel("Color Intensity")
plt.title("Wine Dataset: Alcohol vs Color Intensity")
plt.legend()
plt.show()

# %% 
# --- Train-test split ---
X = df.drop(['target', 'wine_class'], axis=1)
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
