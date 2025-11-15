from sklearn.datasets import load_iris
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Load the dataset
iris = load_iris()

# Create a DataFrame for easier manipulation
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['target_names'] = iris.target_names[iris.target]

# Display initial information (First 5 rows and data types)
print("--- Initial Data Sample ---")
print(df.head())

print("\n--- Data Types and Missing Values ---")
print(df.info())


# Separate features (X) and the categorical target (y)
X = df.drop(columns=['target', 'target_names'])
y_categorical = df['target_names']

# Initialize the LabelEncoder
le = LabelEncoder()

# Fit and transform the categorical target variable
y_encoded = le.fit_transform(y_categorical)

print("\n--- Encoded Target Variable (y) ---")
print(f"Original Categories: {le.classes_}")
print(f"Encoded Values (First 5): {y_encoded[:5]}")


# Separate data into training and testing sets (best practice)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.3, random_state=42
)

# Initialize the StandardScaler
scaler = StandardScaler()

# 1. Fit the scaler on the TRAINING data (important: never fit on test data)
scaler.fit(X_train)

# 2. Transform both the training and testing data
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n--- Feature Scaling (Standardization) ---")
print("Original Training Data (First Row):")
print(X_train.iloc[0].values)
print("\nScaled Training Data (First Row):")
print(X_train_scaled[0])
print(f"\nScaled Training Data Mean (should be near 0): {X_train_scaled.mean():.4f}")
print(f"Scaled Training Data Std Dev (should be near 1): {X_train_scaled.std():.4f}")