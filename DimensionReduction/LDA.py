from mnist import MNIST
import numpy as np

def my_LDA(X):
    """
    Parameters
    ----------
    X : ndarray
        Input features of shape (N, d), where N is the number of samples and d is the number of features.
    """

if __name__ == "__main__":

    # Get Training data
    mnist_loader = MNIST('MNIST')
    train_data, train_label = mnist_loader.load_training()
    test_data, test_label = mnist_loader.load_testing()
    train_data = np.array(train_data, dtype='float')/255 # norm to [0,1]
    train_label = np.array(train_label, dtype='short')
    test_data = np.array(test_data, dtype='float')/255 # norm to [0,1]
    test_label = np.array(test_label, dtype='short')

    # Add random noise to improve real world performance by introducing random noise into train data
    train_data += np.random.normal(0, 0.0001, train_data.shape)

    print(train_data.shape, train_label.shape, test_data.shape, test_label.shape)

    # load images of '1', '3', '6', '9'
    target_digits = [1, 3, 6, 9]
    digit_index = np.isin(train_label, target_digits)

    X = train_data[digit_index]
    print(X.shape)

    # Find class means
    print(X[:10])
