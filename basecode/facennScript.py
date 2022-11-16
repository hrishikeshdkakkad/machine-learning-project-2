'''
Comparing single layer MLP with deep MLP (using TensorFlow)
'''

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

    delta_l = (truth_labels - out_values)
    grad_w2 = delta_l @ out_values

    grad_w2 = (np.add((lambdaval * w2[:, :-1]), grad_w2)) / training_data.shape[0]

    sum_delta_weight2 = np.sum(delta_l @ w2)
    # (sum_delta_weight2 * training_data[:, :-1]
    one_minus_z = (1.0 - hidden1_values[:, :-1])
    sum_dot_input = sum_delta_weight2 * training_data[:, :-1]
    grad_w1 = (one_minus_z * hidden1_values[:, :-1]) * sum_dot_input
    grad_w1 = (np.add((lambdaval * w1[:, :-1]), grad_w1)) / training_data.shape[0]

    tot_err = np.sum((truth_labels - out_values) ** 2)
    tot_err /= (2.0 * training_data.shape[0])
    sum_w1_w2 = np.sum(w1) ** 2 + np.sum(w2) ** 2
    obj_val = tot_err + lambdaval / 2.0 / training_data.shape[0] * sum_w1_w2

    grad_w1 = np.hstack([grad_w1, w1[:, -1].reshape(w1.shape[0], 1)])
    grad_w2 = np.hstack([grad_w2, w2[:, -1].reshape(w2.shape[0], 1)])

    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()), 0)

    return obj_val, obj_grad


# Replace this with your nnPredict implementation
def nnPredict(w1, w2, data):
    labels = np.empty([data.shape[0], 1])

    data = np.hstack([data, np.ones([data.shape[0], 1])])

    hidden1_values = sigmoid(np.matmul(data, w1.transpose()))
    hidden1_values = np.hstack([
        hidden1_values,
        np.ones([hidden1_values.shape[0], 1])
    ])

    out_values = sigmoid(np.matmul(hidden1_values, w2.transpose()))

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
#  Train Neural Network
# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]
# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 32
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

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
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
