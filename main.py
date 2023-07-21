import tensorflow as tf
import scipy.ndimage as ndi
import numpy as np
import matplotlib.pyplot as plt
import wandb
from PIL import ImageOps
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Nadam
from keras import backend as K
from tensorflow.keras.metrics import MeanIoU
from src.models.unet import U_Net
from src.dataset_util.preprocessing import load_data
import yaml

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def plot_and_log(figure_name, x_data, y_data, x_label, y_label, title, legend):
    plt.plot(x_data, y_data, 'y', label=legend)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()
    wandb.log({figure_name: plt})
    plt.close()


# Set up Weights and Biases
project_name = "Renal Segmentation using Convolutional Neural Network"
entity = "sohamt09"
wandb_api_key = "4a711c08cbaa05faa44dd74aa6c768df75a836aa"
wandb.login(key=wandb_api_key)
wandb.init(project=project_name, entity=entity)

# Constants
BATCH_SIZE = 2
NUM_CLASSES = 3
EPOCHS = 5

# Load data
x_train, x_valid, y_train, y_valid = load_data("Dataset/")

# Model definition
input_shape = (256, 256, 1)
model = U_Net(input_shape)

# Custom metrics
def dice_coefficient(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[0, 1])
    union = K.sum(y_true, axis=[0, 1]) + K.sum(y_pred, axis=[0, 1])
    return K.mean((2. * intersection + smooth) / (union + smooth), axis=0)

def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

def jacard_coef(y_true, y_pred):
    intersection = K.sum(y_true * y_pred)
    return (intersection + 1.0) / (K.sum(y_true) + K.sum(y_pred) - intersection + 1.0)

def precision(y_true, y_pred):
    return

model.compile(optimizer=Nadam(learning_rate=1e-4),
            loss='sparse_categorical_crossentropy',
            metrics=[jacard_coef, dice_coefficient, 'accuracy'])

history = model.fit(x=x_train,
                    y=y_train,
                    batch_size=BATCH_SIZE,
                    steps_per_epoch=len(x_train) // BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=(x_valid, y_valid))

# Evaluation
model.evaluate(x_valid, y_valid)
predicted_masks = model.predict(x_valid)

# Calculate IoU for each class
IOU = MeanIoU(num_classes=NUM_CLASSES)
IOU.update_state(y_valid, np.argmax(predicted_masks, axis=3))
iou_results = IOU.result().numpy()

class1_IoU = iou_results[0, 0] / (iou_results[0, 0] + iou_results[0, 1] + iou_results[0, 2] + iou_results[1, 0] + iou_results[2, 0])
class2_IoU = iou_results[1, 1] / (iou_results[1, 1] + iou_results[1, 0] + iou_results[1, 2] + iou_results[0, 1] + iou_results[2, 1])
class3_IoU = iou_results[2, 2] / (iou_results[2, 2] + iou_results[2, 0] + iou_results[2, 1] + iou_results[0, 2] + iou_results[1, 2])

print("Mean IoU = ", iou_results)
print("IoU for class 1 is:", class1_IoU)
print("IoU for class 2 is:", class2_IoU)
print("IoU for class 3 is:", class3_IoU)

# Plot and log training and validation loss
epochs = range(1, EPOCHS + 1)
plot_and_log("Training and Validation Loss", epochs, history.history['loss'], 'Epochs', 'Loss', 'Training and validation loss', 'Training loss')

# Plot and log training and validation Dice
plot_and_log("Training and Validation Dice", epochs, history.history['dice_coefficient'], 'Epochs', 'Dice', 'Training and validation Dice', 'Training Dice')

# Plot and log training and validation Accuracy
plot_and_log("Training and Validation Accuracy", epochs, history.history['accuracy'], 'Epochs', 'Accuracy', 'Training and validation Accuracy', 'Training Accuracy')


def display_mask(j):
    """Quick utility to display a model's prediction."""
    mask = np.argmax(predicted_masks[j], axis=-1)
    mask = np.expand_dims(mask, axis=-1)
    img = ImageOps.autocontrast(keras.preprocessing.image.array_to_img(mask))
    #display(img)
    plt.imshow(ndi.rotate(np.squeeze(mask),-90), cmap='gray')
    plt.show()


# Display the prediction for the first validation sample (i=0)
display_mask(0)
