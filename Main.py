#adding more documention for better code readability : )
# Import necessary libraries
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Load the MNIST dataset, this dataset contains all the data necessary for testing and training.
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Preprocess the data (normalize the images)
x_train = x_train / 255.0  # Normalize the training images to [0, 1]
x_test = x_test / 255.0    # Normalize the test images to [0, 1]

# Reshape the data to include a channel (for grayscale images)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Convert labels to categorical (one-hot encoding)
y_train = tf.keras.utils.to_categorical(y_train, 10) #optional(you can directly import keras.utils to reduce the size of the file)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Build the CNN model
model = models.Sequential([ #every layer is sequentially placed over one another so the outputs of one layer flows onto the next one.
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)), #the 32 represent the number of filters being applied to the layer of neurons which includes edges, textures etc, Using RELU (rectified linear unit) activation to avoid non-linearity.
    layers.MaxPooling2D((2, 2)),#converting into 2 by 2 windows 
    layers.Conv2D(64, (3, 3), activation='relu'),#again applying 64 more filters to improve accuracy with relu activation to avoid non linearity
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),#the output usually needs to be flattened into 1D vector to be used by the Dense function
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compiling the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc * 100:.2f}%")

# Plot training history (loss and accuracy)
plt.figure(figsize=(12, 4))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# Make predictions on some test images
predictions = model.predict(x_test[:5])

# Visualize the predictions
for i in range(5):
    plt.imshow(x_test[i].reshape(28, 28), cmap=plt.cm.binary)
    plt.title(f"Predicted: {predictions[i].argmax()} | True: {y_test[i].argmax()}")
    plt.show()
