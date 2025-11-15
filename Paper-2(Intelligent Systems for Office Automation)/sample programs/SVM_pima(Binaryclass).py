# %%
# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns

# %%
df = pd.read_csv("diabetes.csv")

print("Head of dataset:\n", df.head())
print("\nColumn names:", df.columns)

# %%
# Features and Target
X = df.drop("Outcome", axis=1)   
y = df["Outcome"]                

# %%
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# %%
# Train SVM binary classifier
model = SVC(C=1.0, kernel="linear")  
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# %%
# Evaluation
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))

# %%
# Visualization: Confusion Matrix Heatmap
plt.figure(figsize=(5,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", xticklabels=["No Diabetes","Diabetes"], yticklabels=["No Diabetes","Diabetes"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Pima Indians Diabetes (SVM)")
plt.show()
