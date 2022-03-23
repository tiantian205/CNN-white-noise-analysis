# Some modules / packages we will use to help with the process
# If you encounter errors such as Module not found, please make sure to
# install the packages below first
import tensorflow as tf
from tensorflow.python.keras.utils import np_utils
from ModelAlex import ModelAlex
from ModelLe import ModelLe
from keras.models import load_model

if __name__ == '__main__':
    # Load the image data
    # mnist = tf.keras.datasets.mnist
    # (X_train, y_train), (X_test, y_test) = mnist.load_data()

    mnist_fashion = tf.keras.datasets.fashion_mnist
    (X_train, y_train), (X_test, y_test) = mnist_fashion.load_data()

    # reshape into (batch, height, width, channels)
    # we have 60000 training images and 10000 testing images
    X_train = X_train.reshape(60000, 28, 28, 1)
    X_test = X_test.reshape(10000, 28, 28, 1)

    # Normalize to float between 0 and 1
    # Original pixel values are between 0 and 255
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train = X_train / 255
    X_test = X_test / 255

    classes = 10
    y_train = np_utils.to_categorical(y_train, classes)
    y_test = np_utils.to_categorical(y_test, classes)

    # m = ModelAlex()
    m = ModelLe()
    m.create_model()
    m.compile_model()
    BATCH_SIZE = 64
    EPOCHS = 10
    m.train_model(X_train, y_train, X_test, y_test, BATCH_SIZE, EPOCHS)
