# %%
# Import necessary libraries
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# Load the Iris dataset
iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

# Display basic information
print("Feature names:", iris.feature_names)
print("Target names:", iris.target_names)
print("\nFirst 5 rows of the DataFrame:\n", iris.frame.head())

# %%
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("\nTraining set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

# %%
# Initialize and train the Decision Tree model
# `random_state` ensures reproducibility
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# %%
# Make predictions on the test set
y_pred = dt_model.predict(X_test)

# %%
# Evaluate the model's performance
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=iris.target_names))

# %%
# Visualize the Decision Tree
plt.figure(figsize=(15, 10))
plot_tree(dt_model,
          feature_names=iris.feature_names,
          class_names=iris.target_names,
          filled=True,
          rounded=True,
          fontsize=10)
plt.title("Decision Tree for Iris Dataset")
plt.show()

# %%
# Optional: Plot Confusion Matrix for better visualization
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title('Decision Tree Confusion Matrix for Iris Dataset')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# %%
