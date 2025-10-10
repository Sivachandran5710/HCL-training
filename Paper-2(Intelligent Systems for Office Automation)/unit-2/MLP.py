# -----------------------------
# Import Libraries
# -----------------------------
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# -----------------------------
# Step 1: Load MNIST Dataset
# -----------------------------
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print("Training data shape:", X_train.shape)
print("Test data shape:", X_test.shape)

# -----------------------------
# Step 2: Preprocess Data
# -----------------------------
# Normalize pixel values to [0,1]
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# One-hot encode labels
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# -----------------------------
# Step 3: Build the MLP Model
# -----------------------------
model = Sequential([
    Flatten(input_shape=(28, 28)),          # Flatten 28x28 images to 784 vector
    Dense(128, activation='relu'),          # Hidden layer with 128 neurons
    Dense(64, activation='relu'),           # Hidden layer with 64 neurons
    Dense(10, activation='softmax')         # Output layer for 10 classes
])

# -----------------------------
# Step 4: Compile the Model
# -----------------------------
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# -----------------------------
# Step 5: Train the Model
# -----------------------------
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=10,
    batch_size=32
)

# -----------------------------
# Step 6: Evaluate and save the Model
# -----------------------------
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {test_acc*100:.2f}%")

# Save the model
model.save("mnist_mlp_model.h5")  # HDF5 format

# Later, load the model
from tensorflow.keras.models import load_model
loaded_model = load_model("mnist_mlp_model.h5")

# Evaluate to verify
test_loss, test_acc = loaded_model.evaluate(X_test, y_test)
print(f"Loaded Model Test Accuracy: {test_acc*100:.2f}%")

# -----------------------------
# Step 7: Plot Training History
# -----------------------------
plt.figure(figsize=(12,4))

# Accuracy
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()



