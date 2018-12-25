# ml-cifar10
The data set used for this project is CIFAR-10.  It is a subset of a larger set of 80 million tiny images. For the purpose of this project, we’re using a 10 classifier dataset, which is split into 50’000 training images and 10’000 test images. It is split into 5 training batches and 1 test batch containing 1000 images randomly selected from each class

The training images are also random within each batch but are of the same class. Each image is a 32x32 pixel colored image. The data is stored in arrays of dimension 10,000x3072 for each data batch. 3072 corresponds to 3 channels of Red, Blue and Green values, and 1024 values from 32x32. Each value is a 1 byte unsigned integer (uint8), and stores values from 0-255 (2^7) for the illumination level. 

The batches are split into data and labels. Labels correspond to the classes that the images are classified into and range from 0-9. The dataset also has another file called “batches.meta” that contains the class names corresponding to the label numbers.
The “unpickle” Python module is used to convert the byte stream of data into an object hierarchy to be used in Jupyter. With this, the 5 training batches are concatenated into one “X_train” batch containing 50’000 images, and assigned the test batch into a variable “X_test”. The labels we loaded into “y_train” and “y_test”.

Two different models are used to classify the images.

1] Using an MLP
2] Using a CNN [best for classifying images]