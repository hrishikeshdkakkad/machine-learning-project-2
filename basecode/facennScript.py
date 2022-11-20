'''
Comparing single layer MLP with deep MLP (using TensorFlow)
'''
import time
from datetime import timedelta

import numpy as np
import pickle
from scipy.optimize import minimize
from math import sqrt


# Do not change this
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


# Replace this with your sigmoid implementation
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# Replace this with your nnObjFunction implementation
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


# Replace this with your nnPredict implementation
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

    hidden1_values = sigmoid(np.dot(data, w1.T))
    hidden1_values = np.hstack([
        hidden1_values,
        np.ones([hidden1_values.shape[0], 1])
    ])

    out_values = sigmoid(np.dot(hidden1_values, w2.T))

    for index in range(0, out_values.shape[0]):
        labels[index] = np.argmax(out_values[index])

    return labels


# Do not change this
def preprocess():
    pickle_obj = pickle.load(file=open('face_all.pickle', 'rb'))
    features = pickle_obj['Features']
    labels = pickle_obj['Labels']
    train_x = features[0:21100] / 255
    valid_x = features[21100:23765] / 255
    test_x = features[23765:] / 255

    labels = labels[0]
    train_y = labels[0:21100]
    valid_y = labels[21100:23765]
    test_y = labels[23765:]
    return train_x, train_y, valid_x, valid_y, test_x, test_y


"""**************Neural Network Script Starts here********************************"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

train_label = train_label.reshape(train_label.shape[0], 1)
validation_label = validation_label.reshape(validation_label.shape[0], 1)
test_label = test_label.reshape(test_label.shape[0], 1)
#  Train Neural Network
# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]
# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 4
# set the number of nodes in output unit
n_class = 2

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden)
initial_w2 = initializeWeights(n_hidden, n_class)
# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)
# set the regularization hyper-parameter
lambdaval = 0
args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

# Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example
opts = {'maxiter': 50}  # Preferred value.

start_time = time.time()
nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)

end_time = time.time()

# Difference between start and end-times.
time_dif = end_time - start_time
time_del = str(timedelta(seconds=int(round(time_dif))))

print("Time usage: " + time_del)

params = nn_params.get('x')
# Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

# Test the computed parameters
predicted_label = nnPredict(w1, w2, train_data)
# find the accuracy on Training Dataset
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')
predicted_label = nnPredict(w1, w2, validation_data)
# find the accuracy on Validation Dataset
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')
predicted_label = nnPredict(w1, w2, test_data)
# find the accuracy on Validation Dataset
print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')
