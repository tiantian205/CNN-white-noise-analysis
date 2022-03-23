"""LeNet implementation of CNN. Includes 2 convolution layers, 2 average pool layers, and 2 FC layers
"""
from tensorflow import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Activation, Flatten
from tensorflow.python.keras.layers import Conv2D, AveragePooling2D
from keras.preprocessing.image import ImageDataGenerator


class ModelLe():
    def __init__(self):
        self.model = Sequential()

    def create_model(self):
        # 1st Convolutional Layer
        self.model.add(
            Conv2D(filters=32, input_shape=(28, 28, 1), kernel_size=(5, 5), strides=(1, 1),
                   padding='valid'))
        self.model.add(Activation('relu'))
        # Avg Pooling
        self.model.add(AveragePooling2D(strides=2))

        # 2nd Convolutional Layer
        self.model.add(Conv2D(filters=48, kernel_size=(5, 5), strides=(1, 1), padding='valid'))
        self.model.add(Activation('relu'))
        # Max Pooling
        self.model.add(AveragePooling2D(strides=2))

        # Fully Connected layer
        self.model.add(Flatten())
        # 1st Fully Connected Layer
        self.model.add(Dense(256))
        self.model.add(Activation('relu'))

        # 2nd FC layer
        self.model.add(Dense(84, activation='relu'))

        # Output Layer
        # important to have dense 10, since we have 10 classes
        self.model.add(Dense(10))
        self.model.add(Activation('softmax'))

        self.model.summary()

    def compile_model(self):
        self.model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam',
                           metrics=['accuracy'])

    def train_model(self, X_train, y_train, X_test, y_test, batch_size, epoch):
        gen = ImageDataGenerator(
            rotation_range=8,
            width_shift_range=0.08,
            shear_range=0.3,
            height_shift_range=0.08,
            zoom_range=0.08
        )
        test_gen = ImageDataGenerator()
        train_generator = gen.flow(X_train, y_train, batch_size=batch_size)
        test_generator = test_gen.flow(X_test, y_test, batch_size=batch_size)
        self.model.fit_generator(
            train_generator,
            steps_per_epoch=60000 // batch_size,
            epochs=epoch,
            validation_data=test_generator,
            validation_steps=10000 // batch_size
        )

    def save_model(self, dataset):
        self.model.save("model_alex_{}.h5".format(dataset))
