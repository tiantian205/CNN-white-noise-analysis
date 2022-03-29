# Some modules / packages we will use to help with the process
# If you encounter errors such as Module not found, please make sure to
# install the packages below first
import tensorflow as tf
from tensorflow.python.keras.utils import np_utils
from ModelAlex import ModelAlex
from ModelLe import ModelLe
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.keras.layers import Lambda
import tensorflow.python.keras.backend as K


def visualize_conv_layer(layer_n, x):
    # print(m.layers[layer_n])
    layer_output = m.layers[layer_n].output
    intermediate_model = tf.keras.models.Model(inputs=m.input, outputs=layer_output)
    ret = intermediate_model.predict(X_test[0].reshape(1, 28, 28, 1))
    count = 0
    for i in range(0, 1000):
        if np.nanargmax(y_test[i]) == x:
            intermediate_prediction = intermediate_model.predict(X_test[i].reshape(1, 28, 28, 1))
            out = Lambda(lambda x: K.mean(x, axis=3)[:, :, :, None])(intermediate_prediction)
            if count == 0:
                ret = out
            else:
                ret = ret + out
            count = count + 1
    ret = tf.divide(ret, tf.constant(float(count), shape=tf.shape(ret)))
    return ret

def visualize_noise_layer(layer_n, x):
    white_noise_images = generate_white_noise_image(0.0, 1.0)
    layer_output = m.layers[layer_n].output
    intermediate_model = tf.keras.models.Model(inputs=m.input, outputs=layer_output)
    ret = intermediate_model.predict(tf.random.uniform(shape=[1, 28, 28, 1]))
    count = 0
    for i in range(0, 1000):
        if np.nanargmax(y_test[i]) == x:
            intermediate_prediction = intermediate_model.predict(
                tf.random.uniform(shape=[1, 28, 28, 1]))
            out = Lambda(lambda x: K.mean(x, axis=3)[:, :, :, None])(intermediate_prediction)
            if count == 0:
                ret = out
            else:
                ret = ret + out
            count = count + 1
    ret = tf.divide(ret, tf.constant(float(count), shape=tf.shape(ret)))
    return ret


def generate_white_noise_image(signal, noise):
    """ generate 1000 white noise images with the given signal percent of signal values and noise
    percent of randomly generate noise values"""
    samples_to_predict = []
    for i in range(0, 1000):    # pick 1000 noise images to generate for time and simplicity
        f = tf.random.uniform(shape=[28, 28, 1])
        k = signal * X_test[i] + noise * f
        tf.reshape(k, (28, 28))
        samples_to_predict.append(k)
    samples_to_predict = np.array(samples_to_predict)
    return samples_to_predict


def generate_classification_image(n):
    """generate the classification image of the given ground truth value n based on the randomly
    generated white noise images"""
    white_noise_images = generate_white_noise_image(0.7, 0.3)
    classification = white_noise_images[0]
    count = 0
    for i in range(0, 1000):  # pick 1000 noise images to generate for time and simplicity
        if np.nanargmax(y_test[i]) == n:
            if count == 0:
                classification = white_noise_images[i]
            else:
                classification = classification + white_noise_images[i]
            count = count + 1
    classification = tf.divide(classification, tf.constant(count, shape=(28, 28, 1)))
    return classification


if __name__ == '__main__':
    """comment the next part and uncomment this part to load the mnist dataset"""
    # mnist = tf.keras.datasets.mnist
    # (X_train, y_train), (X_test, y_test) = mnist.load_data()

    """comment the previous part and uncomment this part to load the fashion mnist dataset"""
    # mnist_fashion = tf.keras.datasets.fashion_mnist
    # (X_train, y_train), (X_test, y_test) = mnist_fashion.load_data()

    """reshape into (batch, height, width, channels)
    we have 60000 training images and 10000 testing images"""
    X_train = X_train.reshape(60000, 28, 28, 1)
    X_test = X_test.reshape(10000, 28, 28, 1)

    """data wrangling with training and testing data"""
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train = X_train / 255
    X_test = X_test / 255

    classes = 10
    y_train = np_utils.to_categorical(y_train, classes)
    y_test = np_utils.to_categorical(y_test, classes)

    """code used to train each CNN with the MNIST and Fashion MNIST datasets"""
    # m = ModelAlex()
    # # m = ModelLe()
    # m.create_model()
    # m.compile_model()
    # BATCH_SIZE = 64
    # EPOCHS = 5
    # m.train_model(X_train, y_train, X_test, y_test, BATCH_SIZE, EPOCHS)
    # m.save_model("Fashion_MNIST")

    """pick which pretrained model to load"""
    # m = load_model("model_le_Fashion_MNIST.h5")

    """generate and display the 10 classification images"""
    """also generates the confusion matrix of the chosen model given the classification images"""
    # w = 10
    # h = 10
    # fig_classification = plt.figure(1, figsize=(10, 5))
    # columns = 5
    # rows = 2
    # confusion_matrix = []
    #
    # for i in range(0, 10):
    #     img = generate_classification_image(i)
    #     fig_classification.add_subplot(rows, columns, i+1)
    #     plt.imshow(img, cmap=plt.get_cmap('cool'))
    #     img = tf.expand_dims(img, axis=0)
    #     p = m.predict(img)
    #     print(p)
    #     confusion_matrix.append(p[0])
    # plt.matshow(confusion_matrix, cmap=plt.get_cmap('summer'))
    # for j in range(0, 10):
    #     for k in range(0, 10):
    #         c = confusion_matrix[k][j]
    #         plt.text(j, k, str(round(c, 3)), va='center', ha='center', size=8)

    """Spike triggered analysis for AlexNet"""
    # fig_first_conv = plt.figure()
    # for x in range(0, 10):
    #     STA_conv1 = visualize_conv_layer(2, x)
    #     fig_first_conv.add_subplot(2, 5, x+1)
    #     STA_conv1 = tf.squeeze(STA_conv1, axis=[0, 3])
    #     plt.imshow(STA_conv1, cmap=plt.get_cmap('cool'))
    #
    # fig_last_conv = plt.figure()
    # for x in range(0, 10):
    #     STA_convLast = visualize_conv_layer(8, x)
    #     fig_last_conv.add_subplot(2, 5, x + 1)
    #     STA_convLast = tf.squeeze(STA_convLast, axis=[0, 3])
    #     plt.imshow(STA_convLast, cmap=plt.get_cmap('cool'))
    #
    # fig_first_conv_noise = plt.figure()
    # for x in range(0, 10):
    #     STA_conv1_noise = visualize_noise_layer(2, x)
    #     fig_first_conv_noise.add_subplot(2, 5, x + 1)
    #     STA_conv1_noise = tf.squeeze(STA_conv1_noise, axis=[0, 3])
    #     plt.imshow(STA_conv1_noise, cmap=plt.get_cmap('cool'))
    #
    # fig_last_conv_noise = plt.figure()
    # for x in range(0, 10):
    #     STA_convLast_noise = visualize_noise_layer(8, x)
    #     fig_last_conv_noise.add_subplot(2, 5, x + 1)
    #     STA_convLast_noise = tf.squeeze(STA_convLast_noise, axis=[0, 3])
    #     plt.imshow(STA_convLast_noise, cmap=plt.get_cmap('cool'))

    """Spike triggered analysis for LeNet"""
    # fig_first_conv = plt.figure()
    # for x in range(0, 10):
    #     STA_conv1 = visualize_conv_layer(2, x)
    #     fig_first_conv.add_subplot(2, 5, x + 1)
    #     STA_conv1 = tf.squeeze(STA_conv1, axis=[0, 3])
    #     plt.imshow(STA_conv1, cmap=plt.get_cmap('cool'))
    #
    # fig_last_conv = plt.figure()
    # for x in range(0, 10):
    #     STA_convLast = visualize_conv_layer(5, x)
    #     fig_last_conv.add_subplot(2, 5, x + 1)
    #     STA_convLast = tf.squeeze(STA_convLast, axis=[0, 3])
    #     plt.imshow(STA_convLast, cmap=plt.get_cmap('cool'))
    #
    # fig_first_conv_noise = plt.figure()
    # for x in range(0, 10):
    #     STA_conv1_noise = visualize_noise_layer(2, x)
    #     fig_first_conv_noise.add_subplot(2, 5, x + 1)
    #     STA_conv1_noise = tf.squeeze(STA_conv1_noise, axis=[0, 3])
    #     plt.imshow(STA_conv1_noise, cmap=plt.get_cmap('cool'))
    #
    # fig_last_conv_noise = plt.figure()
    # for x in range(0, 10):
    #     STA_convLast_noise = visualize_noise_layer(5, x)
    #     fig_last_conv_noise.add_subplot(2, 5, x + 1)
    #     STA_convLast_noise = tf.squeeze(STA_convLast_noise, axis=[0, 3])
    #     plt.imshow(STA_convLast_noise, cmap=plt.get_cmap('cool'))

    plt.show()
