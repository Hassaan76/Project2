import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras import layers, models, optimizers
from sklearn.model_selection import train_test_split


IMG_WIDTH, IMG_HEIGHT = 500, 500
BATCH_SIZE = 32
NUM_CLASSES = 3  


train_dir = r"C:\Users\hassa\Documents\GitHub\Projects\Project2\train"
valid_dir = r"C:\Users\hassa\Documents\GitHub\Projects\Project2\valid"


def load_images_from_directory(directory, target_size, class_labels):
    images = []
    labels = []
    for label, class_name in enumerate(class_labels):
        class_dir = os.path.join(directory, class_name)
        for file in os.listdir(class_dir):
            img_path = os.path.join(class_dir, file)
            img = load_img(img_path, target_size=target_size)
            img_array = img_to_array(img) / 255.0  
            images.append(img_array)
            labels.append(label)
    return np.array(images), np.array(labels)


class_labels = ["Crack", "Missing-head", "Paint-off"]


train_images, train_labels = load_images_from_directory(train_dir, (IMG_WIDTH, IMG_HEIGHT), class_labels)
valid_images, valid_labels = load_images_from_directory(valid_dir, (IMG_WIDTH, IMG_HEIGHT), class_labels)


train_labels = tf.keras.utils.to_categorical(train_labels, NUM_CLASSES)
valid_labels = tf.keras.utils.to_categorical(valid_labels, NUM_CLASSES)


def build_model(activation_fn, dense_neurons, learning_rate):
    model = models.Sequential()

    
    model.add(layers.Conv2D(32, (5, 5), activation=activation_fn, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)))
    model.add(layers.MaxPooling2D((3, 3)))

    model.add(layers.Conv2D(64, (5, 5), activation=activation_fn))
    model.add(layers.MaxPooling2D((3, 3)))

    model.add(layers.Conv2D(128, (5, 5), activation=activation_fn))
    model.add(layers.MaxPooling2D((3, 3)))

    model.add(layers.Conv2D(128, (5, 5), activation=activation_fn))
    model.add(layers.MaxPooling2D((3, 3)))

    
    model.add(layers.Flatten())
    model.add(layers.Dense(dense_neurons, activation=activation_fn))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(dense_neurons, activation=activation_fn))
    model.add(layers.Dropout(0.5))

    
    model.add(layers.Dense(NUM_CLASSES, activation="softmax"))

    
    opt = optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
    return model



activation_fn = "relu"
dense_neurons = 128
learning_rate = 0.001

model = build_model(activation_fn, dense_neurons, learning_rate)

history = model.fit(
    train_images,
    train_labels,
    batch_size=BATCH_SIZE,
    epochs=50,
    validation_data=(valid_images, valid_labels),
    verbose=1,
)


def plot_training_history(history):
    epochs = range(len(history.history["accuracy"]))

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history.history["accuracy"], label="Training Accuracy", color="blue")
    plt.plot(epochs, history.history["val_accuracy"], label="Validation Accuracy", color="orange")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history.history["loss"], label="Training Loss", color="blue")
    plt.plot(epochs, history.history["val_loss"], label="Validation Loss", color="orange")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_training_history(history)


model.save('hassaan_model.h5')

def load_and_predict_images(img_paths, model):
    predicted_classes = []
    probabilities = []
    all_predictions = []
    img_arrays = []

    for img_path in img_paths:
        img = load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        img_arrays.append(img)


        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        
        
        predicted_classes.append(predicted_class)
        probabilities.append(predictions[0][predicted_class])
        all_predictions.append(predictions[0])

    return predicted_classes, probabilities, all_predictions, img_arrays


def visualize_predictions(img_paths, model, true_labels, class_labels):
    predicted_classes, probabilities, all_predictions, img_arrays = load_and_predict_images(img_paths, model)
    
    plt.figure(figsize=(15, 5))
    
    for i, img_path in enumerate(img_paths):
        plt.subplot(1, len(img_paths), i + 1)
        plt.imshow(img_arrays[i])
        plt.axis('off')
        
        
        true_label = true_labels[i]
        predicted_label = class_labels[predicted_classes[i]]
        
        
        plt.title(
            f"True: {true_label}\nPredicted: {predicted_label}",
            fontsize=10,
            color='blue'
        )
        
        
        for j, class_name in enumerate(class_labels):
            plt.text(
                0, 490 + j * 15,
                f"{class_name}: {all_predictions[i][j] * 100:.1f}%",
                fontsize=8,
                color='green'
            )
    
    plt.tight_layout()
    plt.show()


img_paths = [
    r"C:\Users\hassa\Documents\GitHub\Projects\Project2\test\crack\test_crack.jpg",
    r"C:\Users\hassa\Documents\GitHub\Projects\Project2\test\missing-head\test_missinghead.jpg",
    r"C:\Users\hassa\Documents\GitHub\Projects\Project2\test\paint-off\test_paintoff.jpg"
]


true_labels = ['crack', 'missing-head', 'paint-off']
class_labels = ['crack', 'missing-head', 'paint-off']


visualize_predictions(img_paths, model, true_labels, class_labels)