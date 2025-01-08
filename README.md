# Install TensorFlow
pip install tensorflow

# Install matplotlib for visualizing results
pip install matplotlib

# Install numpy for numerical operations
pip install numpy

# Install pandas (optional, but useful for data manipulation)
pip install pandas

import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Normalize the images to a range of 0-1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Reshape the images to add an extra dimension for channels (grayscale images)
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

# Build the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10)  # 10 output units for 10 digits (0-9)
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=5)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f"Test accuracy: {test_acc}")

# Predict and display a sample result
predictions = model.predict(test_images)

# Display the first image from the test set and its prediction
plt.imshow(test_images[0].reshape(28, 28), cmap='gray')
plt.title(f"Predicted: {tf.argmax(predictions[0]).numpy()}")
plt.show()
