import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import KFold

(train_X, train_y), (test_X, test_y) = mnist.load_data()

num_trainX = train_X.shape[0]
trainX_img_len = train_X.shape[1]
trainX_img_width = train_X.shape[2]
num_testX = test_X.shape[0]
testX_img_len = test_X.shape[1]
testX_img_width = test_X.shape[2]

# reshape dataset to single color channel
train_X = train_X.reshape(num_trainX, trainX_img_len, trainX_img_width, 1)
test_X = test_X.reshape(num_testX, testX_img_len, testX_img_width, 1)

# one hot encoding for targets
train_y = tf.one_hot(train_y, 10)
test_y = tf.one_hot(test_y, 10)

# normalize pixel values from [0, 255] to [0, 1]
train_X = train_X.astype('float32')
test_X = test_X.astype('float32')
train_X = train_X / 255.0
test_X = test_X / 255.0

accuracies = list()

# 5-fold cross-validation
k_fold = KFold(5, shuffle=True, random_state=1)

# splits
for train_idx, test_idx in k_fold.split(train_X):
    # convolutional neural network model
    model = keras.Sequential()
    model.add(layers.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_uniform', input_shape=(trainX_img_len, trainX_img_width, 1)))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # splitting data for train and test
    train_X_split, test_X_split = train_X[train_idx], train_X[test_idx]
    train_y_split, test_y_split = train_y[train_idx], train_y[test_idx]

    # fit model
    current = model.fit(train_X_split, train_y_split, epochs=10, batch_size=32, validation_data=(test_X_split, test_y_split), verbose=0)

    # evaluate model
    _, accuracy = model.evaluate(test_X_split, test_y_split, verbose=0)
    accuracies.append(accuracy)

print('Accuracy: mean=%.3f std=%.3f, n_folds=%d' % (mean(scores)*100, std(scores)*100, len(scores)))

