import tensorflow as tf
from tensorflow import keras

class ConvolutionalNeuronalNetwork():
    """ Convolutional Neuronal network with adam optimizer

    """
    def __init__(self, input_layer_size=(6,6,1), number_of_categories = 2):
        self.__input_layer_size = input_layer_size
        self.__number_of_categories = number_of_categories

    def compile(self):
        print(f'Use Convolutional Layers for the model')
        optimizer = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0,
                                             amsgrad=False)
        end_layer_size = self.__number_of_categories
        if self.__number_of_categories == 2:
            end_layer_size = 1

        model = keras.Sequential([
            keras.layers.GaussianNoise(0.05),
            keras.layers.Conv2D(32, kernel_size=(1, 1), activation='relu', input_shape=self.__input_layer_size, use_bias=True),
            keras.layers.Conv2D(16, kernel_size=(1, 1), activation='relu', use_bias=True),
            keras.layers.Conv2D(16, kernel_size=(1, 1), activation='relu', use_bias=True),
            #keras.layers.Dropout(0.2, noise_shape=None, seed=None),
            keras.layers.Conv2D(8, kernel_size=(1, 1), activation='relu', use_bias=True),
            keras.layers.Conv2D(8, kernel_size=(1, 1), activation='relu', use_bias=True),
            #keras.layers.Dropout(0.1, noise_shape=None, seed=None),
            keras.layers.Flatten(),
            keras.layers.Dense(2* self.__number_of_categories),
            keras.layers.Dense(2 * self.__number_of_categories),

            keras.layers.Dense(end_layer_size, activation='sigmoid', name='Output')
        ])
        if self.__number_of_categories  == 2:
            print('Use binary cross entropy')
            model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        else:
            print('Use cateegorical cross entropy')
            model.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimizer, metrics=['categorical_accuracy'])
        return model