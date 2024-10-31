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
    print(digit_index)

    X = train_data[digit_index]
    print(X.shape)

    # Find mean of feature of every class
    mean_everything = np.mean(X, axis=0)
    print(mean_everything.shape)

    # Find mean and size of every class
    class_means = []
    class_sizes = []
    for c in target_digits:
        print(f'Finding mean of {c}')
        class_digit_index = (train_label == c) # filter the class index
        target_class = train_data[class_digit_index] # filter the class
        print(target_class)

        class_means.append(np.mean(target_class)) # add mean to list
        c_size = target_class.shape[0]
        class_sizes.append(c_size)

    print(class_means)
    print(class_sizes)
    # Calculate Between-Class scatter matrix
    S_b = ''
    for i in range(target_digits):
        S_b += class_sizes[i] * (class_means[i] - mean_everything) @ (class_means[i] - mean_everything).T
        print(S_b.shape)
