from datetime import timedelta
from time import strftime
import numpy as np
import time
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt


def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""

    return 1.0 / (1.0 + np.exp(-z))


def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - feature selection"""

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    train_data = []
    test_data = []
    train_label = []
    test_label = []
    for i in range(10):
        train_i = mat['train' + str(i)]
        test_i = mat['test' + str(i)]
        for ex in train_i:
            train_data.append(ex)
            train_label.append(i)
        for test_ex in test_i:
            test_data.append(test_ex)
            test_label.append(i)

    train_data = np.array(train_data) / 255
    # train_data = train_data[:, ~np.all(train_data[1:] == train_data[:-1], axis=0)]
    test_data = np.array(test_data) / 255
    # test_data = test_data[:, ~np.all(test_data[1:] == test_data[:-1], axis=0)]
    train_label = np.array(train_label)
    test_label = np.array(test_label)

    train_label = np.reshape(train_label, (len(train_label), 1))
    test_label = np.reshape(test_label, (len(test_label), 1))

    train_with_label = np.hstack([train_data, train_label])
    test_with_label = np.hstack([test_data, test_label])

    np.random.shuffle(train_with_label)
    np.random.shuffle(test_with_label)

    validation_data = train_with_label[-10000:, :-1]
    validation_label = train_with_label[-10000:, -1].reshape(10000, 1)

    train_data = train_with_label[:50000, :-1]
    train_label = train_with_label[:50000, -1].reshape(50000, 1)

    # Split the training sets into two sets of 50000 randomly sampled training examples and 10000 validation examples. 
    # Your code here.

    # Feature selection
    # Your code here.

    print('preprocess done')

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
       %   likelihood error function with regularization) given the parameters
       %   of Neural Networks, the training data, their corresponding training
       %   labels and lambda - regularization hyper-parameter.

       % Input:
       % params: vector of weights of 2 matrices w1 (weights of connections from
       %     input layer to hidden layer) and w2 (weights of connections from
       %     hidden layer to output layer) where all of the weights are contained
       %     in a single vector.
       % n_input: number of node in input layer (not include the bias node)
       % n_hidden: number of node in hidden layer (not include the bias node)
       % n_class: number of node in output layer (number of classes in
       %     classification problem
       % training_data: matrix of training data. Each row of this matrix
       %     represents the feature vector of a particular image
       % training_label: the vector of truth label of training images. Each entry
       %     in the vector represents the truth label of its corresponding image.
       % lambda: regularization hyper-parameter. This value is used for fixing the
       %     overfitting problem.

       % Output:
       % obj_val: a scalar value representing value of error function
       % obj_grad: a SINGLE vector of gradient value of error function
       % NOTE: how to compute obj_grad
       % Use backpropagation algorithm to compute the gradient of error function
       % for each weights in weight matrices.

       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       % reshape 'params' vector into 2 matrices of weight w1 and w2
       % w1: matrix of weights of connections from input layer to hidden layers.
       %     w1(i, j) represents the weight of connection from unit j in input
       %     layer to unit i in hidden layer.
       % w2: matrix of weights of connections from hidden layer to output layers.
       %     w2(i, j) represents the weight of connection from unit j in hidden
       %     layer to unit i in output layer."""

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

    training_data = np.hstack(
        [
            training_data,
            np.ones([training_data.shape[0], 1])
        ]
    )
    hidden1_values = sigmoid(np.matmul(training_data, w1.transpose()))

    hidden1_values = np.hstack([
        hidden1_values,
        np.ones([hidden1_values.shape[0], 1])
    ])

    out_values = sigmoid(np.matmul(hidden1_values, w2.transpose()))

    # One hot encoding
    truth_labels = np.zeros([training_label.shape[0], out_values.shape[1]])
    for i in range(truth_labels.shape[0]):
        truth_labels[i, int(training_label[i])] = 1

    delta_l = out_values - truth_labels
    grad_w2 = np.dot(delta_l.T, hidden1_values)

    grad_w2 = (np.add((lambdaval * w2), grad_w2)) / training_data.shape[0]

    sum_delta_weight2 = np.dot(delta_l, w2[:, :-1])
    one_minus_z_dot_z = (1.0 - hidden1_values[:, :-1]) * hidden1_values[:, :-1]
    lft_part = sum_delta_weight2 * one_minus_z_dot_z
    grad_w1 = np.dot(lft_part.T, training_data)

    grad_w1 = (np.add((lambdaval * w1), grad_w1)) / training_data.shape[0]

    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()), 0)

    tot_err = (-np.add(
        np.dot(
            truth_labels.flatten(),
            np.log(out_values).flatten().T
        ),
        np.dot(
            1 - truth_labels.flatten(),
            np.log(1 - out_values).flatten().T
        )
    ) / training_data.shape[0])

    regularization = lambdaval * np.add(np.sum(w1 ** 2), np.sum(w2 ** 2)) / (2 * training_data.shape[0])

    obj_val = tot_err + regularization

    print(obj_val)

    return obj_val, obj_grad


def nnPredict(w1, w2, data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels"""

    labels = np.empty([data.shape[0], 1])

    data = np.hstack([data, np.ones([data.shape[0], 1])])

    hidden1_values = sigmoid(np.dot(data, w1.transpose()))
    hidden1_values = np.hstack([
        hidden1_values,
        np.ones([hidden1_values.shape[0], 1])
    ])

    out_values = sigmoid(np.dot(hidden1_values, w2.T))

    for index in range(0, out_values.shape[0]):
        labels[index] = np.argmax(out_values[index])

    return labels


"""**************Neural Network Script Starts here********************************"""


def runScript(reg_param, hidden):
    train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

    print('Start Time: ' + strftime("%Y-%m-%d %H:%M:%S"))
    start_time = time.time()
    #  Train Neural Network

    # set the number of nodes in input unit (not including bias unit)
    n_input = train_data.shape[1]

    # set the number of nodes in hidden unit (not including bias unit)
    n_hidden = hidden

    # set the number of nodes in output unit
    n_class = 10

    # initialize the weights into some random matrices
    initial_w1 = initializeWeights(n_input, n_hidden)
    initial_w2 = initializeWeights(n_hidden, n_class)

    # unroll 2 weight matrices into single column vector
    initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)
    # set the regularization hyper-parameter
    lambdaval = reg_param

    args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

    # Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

    opts = {'maxiter': 50}  # Preferred value.

    nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)

    print("Training done")

    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time
    time_del = str(timedelta(seconds=int(round(time_dif))))

    print("Time usage: " + time_del)

    print("lambda:" + str(lambdaval), "hidden:" + str(n_hidden))
    # In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
    # and nnObjGradient. Check documentation for this function before you proceed.
    # nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)

    # Reshape nnParams from 1D vector into w1 and w2 matrices
    w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

    # Test the computed parameters

    predicted_label = nnPredict(w1, w2, train_data)

    # find the accuracy on Training Dataset

    train_acc = str(100 * np.mean((predicted_label == train_label).astype(float)))

    print('\n Training set Accuracy:' + train_acc + '%')

    predicted_label = nnPredict(w1, w2, validation_data)

    # find the accuracy on Validation Dataset

    validation_acc = str(100 * np.mean((predicted_label == validation_label).astype(float)))

    print('\n Validation set Accuracy:' + validation_acc + '%')

    predicted_label = nnPredict(w1, w2, test_data)

    # find the accuracy on Validation Dataset
    test_acc = str(100 * np.mean((predicted_label == test_label).astype(float)))

    print('\n Test set Accuracy:' + test_acc + '%')

    return train_acc, validation_acc, test_acc, time_del
