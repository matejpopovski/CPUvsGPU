import tensorflow as tf
mnist = tf.keras.datasets.mnist
import matplotlib.pyplot as plt
import numpy as np

# Verify TensorFlow installation
print(f"TensorFlow version: {tf.__version__}")

# Load MNIST dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Display the first image in the dataset
plt.imshow(train_images[0], cmap=plt.cm.binary)
plt.title(f"Label: {train_labels[0]}")
plt.show()

# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),  # Flatten the 28x28 images
    tf.keras.layers.Dense(128, activation='relu'),  # Hidden layer with 128 neurons
    tf.keras.layers.Dense(10, activation='softmax') # Output layer with 10 neurons (one for each digit)
])

# Compile the modell
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=5, validation_split=0.1)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc}")

# Make predictions
predictions = model.predict(test_images)

# Display the prediction for the first image
print(f"Prediction for the first image: {np.argmax(predictions[0])}")

# Save the model
model.save('digit_recognition_model.h5')

# Load the model
loaded_model = tf.keras.models.load_model('digit_recognition_model.h5')
