import os
import cv2 # computer vision
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# mnist = tf.keras.datasets.mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# x_train = tf.keras.utils.normalize(x_train, axis=1)
# x_test = tf.keras.utils.normalize(x_test, axis=1)

# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
# model.add(tf.keras.layers.Dense(128, activation='relu'))
# model.add(tf.keras.layers.Dense(10, activation='softmax'))

# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# model.fit(x_train, y_train, epochs=10)

# model.save('handwritten.keras')

# model = tf.keras.models.load_model('handwritten.keras')

# #loss, accuracy = model.evaluate(x_test, y_test)

# image_number = 1
# while os.path.isfile(f"digits/digit{image_number}.png"):
#     try:
#         img = cv2.imread(f"digits/digit{image_number}.png")[:,:,0]
#         img = np.invert(np.array([img]))
#         prediction = model.predict(img)
#         print(f"This digit is probably a {np.argmax(prediction)}")
#         plt.imshow(img[0], cmap=plt.cm.binary)
#         plt.show()
#     except:
#         print("Error.")
#     finally:
#         image_number += 1



# # Load MNIST dataset
# mnist = tf.keras.datasets.mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# # Normalize the data to the range [0, 1]
# x_train = tf.keras.utils.normalize(x_train, axis=1)
# x_test = tf.keras.utils.normalize(x_test, axis=1)

# # Define the model
# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))  # Flatten the 28x28 images into 1D vectors
# model.add(tf.keras.layers.Dense(128, activation='relu'))  # First Dense layer with 128 neurons
# model.add(tf.keras.layers.Dense(10, activation='softmax'))  # Output layer with 10 neurons for the 10 digits (0-9)

# # Compile the model
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# # Train the model for 10 epochs (you can increase this number for better results)
# history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# # Save the retrained model
# model.save('handwritten.keras')

# # Evaluate the model
# loss, accuracy = model.evaluate(x_test, y_test)
# print(f"Test accuracy: {accuracy:.4f}")

# # Optionally plot the training history for visualization
# plt.plot(history.history['accuracy'], label='Training Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()

model = tf.keras.models.load_model('handwritten.keras')

image_number = 1
while os.path.isfile(f"digits/digit{image_number}.png"):
    try:
        # Read the image
        img = cv2.imread(f"digits/digit{image_number}.png", cv2.IMREAD_GRAYSCALE)  # Ensure it's grayscale
        img = cv2.resize(img, (28, 28))  # Resize to 28x28 pixels
        img = np.invert(img)  # Invert colors to match MNIST
        
        # Normalize the image to the same range [0, 1] as training data
        img = img / 255.0

        # Reshape image to match the input shape of the model
        img = img.reshape(1, 28, 28)

        # Predict the digit
        prediction = model.predict(img)
        print(f"This digit is probably a {np.argmax(prediction)}")

        # Display the image
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        image_number += 1