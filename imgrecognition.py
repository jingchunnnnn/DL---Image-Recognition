# Link to dataset: https://www.kaggle.com/datasets/mdwaquarazam/agricultural-crops-image-classification

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
input_folder = "/Users/zhoujingchun/Desktop/Agricultural-crops"
output_folder = "/Users/zhoujingchun/Desktop/ImageRecognition"

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


# In[45]:


model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation = 'relu'),
    layers.Dropout(0.5),
    layers.Dense(30, activation = 'softmax')
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
    0: 'Cherry', 
    1: 'Coffee-plant', 
    2: 'Cucumber', 
    3: 'Fox_nut(Makhana)', 
    4: 'Lemon', 
    5: 'Olive-tree', 
    6: 'Pearl_millet(bajra)', 
    7: 'Tobacco-plant', 
    8: 'almond', 
    9: 'banana', 
    10: 'cardamom', 
    11: 'chilli', 
    12: 'clove', 
    13: 'coconut', 
    14: 'cotton', 
    15: 'gram', 
    16: 'jowar', 
    17: 'jute', 
    18: 'maize', 
    19: 'mustard-oil', 
    20: 'papaya', 
    21: 'pineapple', 
    22: 'rice', 
    23: 'soybean', 
    24: 'sugarcane', 
    25: 'sunflower', 
    26: 'tea', 
    27: 'tomato', 
    28: 'vigna-radiati(Mung)', 
    29: 'wheat'
}


def predict_img(image, model):
    test_img = cv2.imread(image)
    test_img = cv2.resize(test_img, (224, 224))
    test_img = np.expand_dims(test_img, axis = 0)
    result = model.predict(test_img)
    r = np.argmax(result)
    print('Category: ' + class_names[r])

def categorize(img):
    imgArray = np.array(Image.open(img))
    plt.imshow(imgArray)
    plt.show()
    return predict_img(img, model)


# Print the image and let the model recognise its category
    
testImg = '/Users/zhoujingchun/Desktop/ImageRecognition/test/maize/image (6).jpg'
categorize(testImg)

# Testing using an image from Google which was not used to train the model
googleImg = '/Users/zhoujingchun/Desktop/test-img-from-google.webp'
categorize(googleImg)




