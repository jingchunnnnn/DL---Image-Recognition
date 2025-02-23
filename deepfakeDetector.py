# Link to Kaggle Dataset: https://www.kaggle.com/datasets/uditsharma72/real-vs-fake-faces

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import splitfolders
import cv2
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras import layers, models
import os
import random
from PIL import Image

# Split the images in the kaggle dataset into 3 different subfolders
input_folder = "/Users/zhoujingchun/Desktop/Faces"
output_folder = "/Users/zhoujingchun/Desktop/DeepfakeRecognition"

split_ratio = (0.8, 0.1, 0.1)

splitfolders.ratio(
    input_folder, 
    output = output_folder,
    seed = 500,
    ratio = split_ratio,
    group_prefix = None
)


img_size = (224, 224)
batch_size = 32

train_datagen = ImageDataGenerator(
    preprocessing_function = preprocess_input,
    rotation_range = 20,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
    fill_mode = 'nearest'
)


# Data augmentation (rescale)
test_datagen = ImageDataGenerator(preprocessing_function = preprocess_input)
valid_datagen = ImageDataGenerator(preprocessing_function = preprocess_input)


train_dir = os.path.join(output_folder, 'train')
val_dir = os.path.join(output_folder, 'val')
test_dir = os.path.join(output_folder, 'test')

train_data = train_datagen.flow_from_directory(
    train_dir, 
    target_size = img_size,
    batch_size = batch_size,
    class_mode = 'categorical'
)

valid_data = valid_datagen.flow_from_directory(
    val_dir,
    target_size = img_size,
    batch_size = batch_size,
    class_mode = 'categorical'
)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size = img_size,
    batch_size = batch_size,
    class_mode = 'categorical'
)

# Print random img
images, labels = next(valid_data)
i = random.randint(0, images.shape[0] - 1)

plt.imshow(images[i])
plt.show()


from keras.applications.resnet import ResNet50
base_model = ResNet50(
    weights = 'imagenet',
    include_top = False,
    input_shape = (img_size[0], img_size[1], 3)
)

base_model.trainable = False


model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation = 'relu'),
    layers.Dropout(0.5),
    layers.Dense(2, activation = 'softmax')
])

model.compile(
    optimizer = 'adam',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)



# Training for 100 epochs
model.fit(
    train_data, 
    epochs = 100, 
    validation_data = valid_data
)


# Evaluate overall accuracy
test_loss, test_accuracy = model.evaluate(test_data)
print(f'Test Accuracy: {test_accuracy * 100: .2f}%')


class_names = {
    0: 'Fake',
    1: 'Real'
}


def predict_img(image, model):
    test_img = cv2.imread(image)
    test_img = cv2.resize(test_img, (224, 224))
    test_img = np.expand_dims(test_img, axis = 0)
    result = model.predict(test_img)
    r = np.argmax(result)
    if r :
        return 'Yay! This photo is real.'
    return 'Watch out! This photo is a fake!'

def detector(img):
    imgArray = np.array(Image.open(img))
    plt.imshow(imgArray)
    plt.show()
    return predict_img(img, model)


# Testing the trained model

img1 = '/Users/zhoujingchun/Desktop/DeepfakeRecognition/test/fake/mid_364_1111.jpg'
detector(img1)


#fake
img2 = '/Users/zhoujingchun/Desktop/zelenskyy.jpg'
detector(img2)


#fake
img3 = '/Users/zhoujingchun/Desktop/zucc.jpg'
detector(img3)


#real
img4 = '/Users/zhoujingchun/Desktop/zuck.jpeg'
detector(img4)


# for fun
img5 = '/Users/zhoujingchun/Desktop/bob.webp'
detector(img5)




