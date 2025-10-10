# %%
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# Load the diabetes dataset
df = pd.read_csv("diabetes.csv")

# Display basic information
print("DataFrame shape:", df.shape)
print("\nFirst 5 rows of the DataFrame:\n", df.head())
print("\nUnique Target classes:", df.Outcome.unique())
target_names = ['No Diabetes', 'Diabetes']

# %%
# Define features (X) and target (y)
X = df.drop("Outcome", axis=1)   
y = df['Outcome']

# %%
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("\nTraining set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

# %%
# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %%
# Initialize and train the KNN model
knn_model = KNeighborsClassifier(n_neighbors=10)
knn_model.fit(X_train_scaled, y_train)

# %%
# Make predictions on the test set
y_pred = knn_model.predict(X_test_scaled)

# %%
# Evaluate the model's performance
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
# Use the custom target_names list
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=target_names))

# %%
# Plot Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names)
plt.title('KNN Confusion Matrix for Diabetes Dataset')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# %%
