import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
# Generate random data
data = np.random.rand(100, 3)  # 100 samples, 3 features
# Convert to DataFrame
df = pd.DataFrame(data, columns=['feature1', 'feature2', 'target'])
# Display shape and description of the DataFrame
print("DataFrame shape:", df.shape)
print("DataFrame description:\n", df.describe())
# Prepare the data
features = ['feature1', 'feature2']
target = 'target'
x = df[features]
y= df[target]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print("Test set shape:", x_test.shape)
model = LogisticRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print("Predicted values:", y_pred)
y_pred_proba = model.predict_proba(x_test)
print("Predicted probabilities:", y_pred_proba)
print("Accuracy:", model.score(x_test, y_test))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
