import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array


IMG_WIDTH, IMG_HEIGHT = 500, 500


model = load_model('hassaan_model.h5')


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
        all_predictions.append(predictions[0])  # All class probabilities

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
