import os
import caer
import canaro
import numpy as np
import cv2 as cv
import gc
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import LearningRateScheduler
import sklearn.model_selection as skm

IMG_SIZE = (80, 80)
channels = 1
char_path = r'../input/the-simpsons-characters-dataset/simpsons_dataset'

char_dict = {}
for char in os.listdir(char_path):
    char_dict[char] = len(os.listdir(os.path.join(char_path,char)))

# Sort in descending order
char_dict = caer.sort_dict(char_dict, descending=True)
char_dict

characters = []
count = 0
for i in char_dict:
    characters.append(i[0])
    count += 1
    if count >= 10:
        break
characters

#creating training data
train = caer.preprocess_from_dir(char_path, characters, channels=channels, IMG_SIZE=IMG_SIZE, isShuffle=True, verbose=0)

plt.figure(figsize=(30, 30))
plt.imshow(train[0][0], cmap='gray')
plt.show()

featureSet, labels = caer.sep_train(train, IMG_SIZE=IMG_SIZE)

featureSet, labels = caer.sep_train(train, IMG_SIZE=IMG_SIZE)

# Normalize the featureSet 
featureSet = caer.normalize(featureSet)
# Converting numerical labels to binary class vectors whatever that means lmao
labels = to_categorical(labels, len(characters))

x_train, x_val, y_train, y_val = caer.train_val_split(featureSet, labels, val_ratio=.2)

split_data = skm.train_test_split(featureSet, labels, test_size=.2)
x_train, x_val, y_train, y_val = (np.array(item) for item in split_data)


# clear variables
del train
del featureSet
del labels 
gc.collect()

BATCH_SIZE = 32
EPOCHS = 10

datagen = canaro.generators.imageDataGenerator()
train_gen = datagen.flow(x_train, y_train, batch_size=BATCH_SIZE)

model = canaro.models.createSimpsonsModel(IMG_SIZE=IMG_SIZE, channels=channels, output_dim=len(characters), loss='binary_crossentropy', decay=1e-6, learning_rate=0.001, momentum=0.9,nesterov=True)

model.summary()

callbacks_list = [LearningRateScheduler(canaro.lr_schedule)]

training = model.fit(train_gen,steps_per_epoch=len(x_train)//BATCH_SIZE,epochs=EPOCHS,validation_data=(x_val,y_val),validation_steps=len(y_val)//BATCH_SIZE,callbacks = callbacks_list)

test_path = r'../input/test-data-2/PRI_154097326-2.jpg'
img = cv.imread(test_path)
plt.imshow(img)
plt.show()


def prepare(image):
    image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    image = cv.resize(image, IMG_SIZE)
    image = caer.reshape(image, IMG_SIZE, 1)
    return image
    
predictions = model.predict(prepare(img))

print(characters[np.argmax(predictions[0])])
