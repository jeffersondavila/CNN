import os
import requests
import numpy as np
import pandas as pd
from keras.models import Model
import matplotlib.pyplot as plt
from keras.optimizers import SGD
from keras.layers import Dropout, Dense
from keras.applications import mobilenet
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator

def read_and_process_labels(file_path):
    with open(file_path, 'r') as file:
        labels = file.readlines()
        labels = [label.strip() for label in labels if label.strip()]
        return labels[:STYLE_NUM_LABELS]

def save_model(model, weights_filename):
    if not os.path.exists(weights_filename):
        model.save(weights_filename)
        print("Pesos del modelo guardados")

STYLE_NUM_LABELS = 2
SEED = 42

IMAGES_DIR = 'C:/Users/jeffe/PycharmProjects/RedNeuronalSimple/data/celeba/celeba_style/images/'


style_labels = read_and_process_labels('./data/celeba/celeba_style/style_names.txt')


train_valid_frame = pd.read_csv('./data/celeba/celeba_style/train.txt', sep=" ", header=None)
train_valid_frame.columns = ['files', 'lab_idx']
train_valid_frame['labels'] = train_valid_frame['lab_idx'].map({i: j for i, j in enumerate(style_labels)})

train_frame, valid_frame = train_test_split(
    train_valid_frame, test_size=0.2, random_state=SEED, stratify=train_valid_frame['labels'])


test_frame = pd.read_csv('./data/celeba/celeba_style/test.txt', sep=" ", header=None)
test_frame.columns = ['files', 'lab_idx']
test_frame['labels'] = test_frame['lab_idx'].map({i: j for i, j in enumerate(style_labels)})

train_frame['files'] = IMAGES_DIR + train_frame['files']
valid_frame['files'] = IMAGES_DIR + valid_frame['files']
test_frame['files'] = IMAGES_DIR + test_frame['files']

base_model = mobilenet.MobileNet(input_shape=(224, 224, 3), alpha=1, include_top=False, pooling='avg', weights='imagenet')

for layer in base_model.layers:
    layer.trainable = False

x = Dropout(0.75)(base_model.output)
x = Dense(1, activation='sigmoid', name='flickr_out')(x)

model = Model(base_model.input, x)

train_datagen = ImageDataGenerator(preprocessing_function=mobilenet.preprocess_input)
valid_datagen = ImageDataGenerator(preprocessing_function=mobilenet.preprocess_input)
test_datagen = ImageDataGenerator(preprocessing_function=mobilenet.preprocess_input)

train_iter = train_datagen.flow_from_dataframe(train_frame,
                                               x_col='files',
                                               y_col='labels',
                                               target_size=(224,224),
                                               class_mode='binary',
                                               batch_size=32,
                                               shuffle=False)

valid_iter = valid_datagen.flow_from_dataframe(valid_frame,
                                               x_col='files',
                                               y_col='labels',
                                               target_size=(224,224),
                                               class_mode='binary',
                                               batch_size=32,
                                               shuffle=False)

test_iter = test_datagen.flow_from_dataframe(test_frame,
                                             x_col='files',
                                             y_col='labels',
                                             target_size=(224,224),
                                             class_mode='binary',
                                             batch_size=32,
                                             shuffle=False)

sgd = SGD(learning_rate=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

history = model.fit(train_iter,
                    steps_per_epoch=train_frame.shape[0] // train_iter.batch_size,
                    epochs=20,
                    validation_data=valid_iter,
                    validation_steps=valid_frame.shape[0] // valid_iter.batch_size)

score = model.evaluate(valid_iter, steps=valid_frame.shape[0] // valid_iter.batch_size)
print("loss en validación: {} \naccuracy en validación: {}".format(score[0],score[1]))
score = model.evaluate(test_iter, steps=test_frame.shape[0] // test_iter.batch_size)
print("loss en test: {} \naccuracy en test: {}".format(score[0],score[1]))

save_model(model, 'model.h5')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['train', 'valid'])
plt.title('Cross Entropy')
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['train', 'valid'])
plt.title('Accuracy')
plt.show()