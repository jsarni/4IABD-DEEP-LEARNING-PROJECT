from cifar10.models.ModelTester import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.datasets import cifar10

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import matplotlib.pyplot as plt
import numpy as np



(train_data, train_labels), (val_data, val_labels) = cifar10.load_data()

linClassifier = Sequential()
linClassifier.add(Flatten())
linClassifier.add(Dense(units = 10, kernel_initializer='uniform', activation='relu', input_dim=10))

print('Training data shape: ', train_data.shape)
print('Training labels shape: ', train_labels.shape)
print('Test data shape: ', val_data.shape)
print('Test labels shape: ', val_labels.shape)
#print('Features : '),  train_data.


classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
samples_per_class = 7


def visualisation(dataset, classes, samples_per_class):
    num_classes = len(classes)
    for y, cls in enumerate(classes):
      idxs = np.flatnonzero(train_labels == y)
      idxs = np.random.choice(idxs, samples_per_class, replace=False)
      for i, idx in enumerate(idxs):
        plt_idx = i * num_classes + y + 1
        plt.subplot(samples_per_class, num_classes, plt_idx)
        plt.imshow(train_data[idx].reshape((32, 32, 3)).astype('uint8'))
        plt.axis('off')
        if i == 0:
          plt.title(cls)
    plt.show()

visualisation(train_data, classes, samples_per_class)

print("--------------------------it's done--------------------------")



linClassifier.compile(optimizer='adam', loss='binary_crossentropy', metrics='accuracy')
linClassifier.fit(train_data, train_labels, batch_size= 100, epochs=10)
linClassifier.predict(val_data, batch_size=100)
