from tensorflow import keras
from .Models import CNN_2D
from kegra.utils import *


class DeepLearningModels(object):

    def __init__(self, train_data, train_labels, val_data, val_labels, number_of_categories, epochs):
        self.__train_data = train_data
        self.__train_labels = train_labels
        self.__val_data = val_data
        self.__epochs = epochs
        self.__val_labels = val_labels
        self.__number_of_categories = number_of_categories
        self.__input_size = self.__train_data.shape[1]

    def train_model(self):
        print(self.__number_of_categories)
        if self.__number_of_categories !=2:
            print('Number of categories are higher than 2 --> n class one hote encoding is used for ')
            self.__train_labels = keras.utils.to_categorical(self.__train_labels, self.__number_of_categories)
            self.__val_labels = keras.utils.to_categorical(self.__val_labels, self.__number_of_categories)

        self.__train_data = np.reshape(self.__train_data, (len(self.__train_data), 6, 6, 1))
        self.__val_data = np.reshape(self.__val_data , (len(self.__val_data), 6, 6, 1))
        network = CNN_2D.ConvolutionalNeuronalNetwork((6, 6, 1), self.__number_of_categories)
        model = network.compile()
        history = model.fit(self.__train_data, self.__train_labels, validation_data=(self.__val_data, self.__val_labels), epochs=self.__epochs ,
                            batch_size=64)
        return model

    def __reorder_data(self, data):
        data_trans = np.transpose(data)
        sf12, sf1, sf2, sf3, sf4, sp12, sp23, sp34, sp41, se11, se12, se13, se21, se22, se23 = data_trans
        new_data = np.array([sp12, sf1, se13, sf2, sp23, se21, se11, sf12, se21, se22, sp34, sf4, se23, sf3, sp41])
        sorted_data = np.transpose(new_data)
        return sorted_data

    def __expand_data(self, data, labels):
        data = np.expand_dims(data, axis=2)  # Expand for Conv1D
        labels = np.expand_dims(labels, axis=2)  # Expand for Conv1D
        return data, labels
