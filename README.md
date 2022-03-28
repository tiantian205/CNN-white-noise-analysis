# CNN-white-noise-analysis
Part 1 and 2:\
I have implemented two different CNN architectures: 
- an AlexNet implementation, featuring 3 convolution layers, 3 max pool layers, and 2 fully connected layers.
- a LeNet implementation, featuring 2 convolution layers, 2 average pool layers, and 2 fully connected layers.

Each CNN is implemented as a object in a python file. Please see a commented code chunk in main.py for instructions on how to train them and save them for future use with ease.\
I am using the tensorflow and keras libraries for this entire project. As such, the MNIST and FashionMNIST datasets are directly imported. Please also see main.py as to how to alternate between the two datasets.

Part 3:\
To generate classification images for both datasets, I first generated 1000 random white noise images using 70% signal and 30% noise. The arbitrary value of 1000 is taken so that I can get output in a relatively short amount of time. \
The function in main.py generate_white_noise_images handles this task. Then I took the average value for each category of the white noise images based on ground truths. This step can be found in the generate_classification_images function. Its argument n dictates what category we are looking for (i.e. classification image for 0 in MNIST)\
Below you can see the classification images for both datasets:
![alt text](part3_imgs/classification_mnist.png "MNIST")
![alt text](part3_imgs/classification_fashion_mnist.png "FashionMNIST")
I then used these classification images and fed them back into my models and obtained the confusion matrices\
Note that the y-axis is the ground truth, with x-axis being the predicted category.\
For the AlexNet CNN:
![alt text](part3_imgs/cm_alex_mnist.png "alex MNIST")
![alt text](part3_imgs/cm_alex_fashion_mnist.png "alex FashionMNIST")
For the LeNet CNN:
![alt text](part3_imgs/cm_lenet_mnist.png "le MNIST")
![alt text](part3_imgs/cm_lenet_fashion_mnist.png "le FashionMNIST")
The exact steps of generating these confusion matrices are within the main function in main.py
