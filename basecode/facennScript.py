'''
Comparing single layer MLP with deep MLP (using TensorFlow)
'''

import numpy as np
import pickle
from scipy.optimize import minimize
from math import sqrt

# Do not change this
def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
                            
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon
    return W



# Replace this with your sigmoid implementation
def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""

    return  1/(1+np.exp(-1.0*z))# your code here
# Replace this with your nnObjFunction implementation

def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
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
    obj_val = 0

    # Your code here
    #
    #
    #
    #
    #
    #//------------------------------Forward Propagation ---------------------------//
    x1=np.concatenate((np.ones((training_data.shape[0],1)), training_data), axis=1)
    y1= np.matmul(x1,w1.T)
    s1= sigmoid(y1)

    x2=np.concatenate((np.ones((training_data.shape[0],1)), s1), axis=1)
    y2=np.matmul(x2,w2.T)
    s2= sigmoid(y2)
    #//------------------------------Forward Propagation Over---------------------------//
    #n_class_labels = Mapping_one_to_n(training_label,n_class)
    n_class_labels = np.zeros((training_label.shape[0],n_class))
    for i in range(training_label.shape[0]):
      n_class_labels[i][int(training_label[i])] = 1

#//------------------------------Backword Propagation ---------------------------//
    #log_likelihood = np.matmul(n_class_labels,(np.log(s2)).T) + np.matmul(1-n_class_labels,(np.log(1-s2)).T)
    size_data= training_data.shape[0]
    class_temp_first = n_class_labels.ravel()
    log_temp_first = (np.log(s2)).ravel()
    class_temp_second = (1-n_class_labels).ravel()
    log_temp_second =  (np.log(1-s2)).ravel()
    log_likelihood = -( np.dot(class_temp_first,log_temp_first.T) + np.dot(class_temp_second,log_temp_second.T))/size_data
    regularization = ((np.sum(w1**2) + np.sum(w2**2))*lambdaval/(2*size_data))
    obj_val = log_likelihood + regularization

    output_delta = (s2 - n_class_labels)
    w2_error = np.dot(output_delta.T , x2)
    
    w2_temp = w2[:,1:w2.shape[1]]
    hidden_delta =  np.dot(w2_temp.transpose(), output_delta.T)*(s1.T*(1 - s1.T)) 

    w1_gradient = (np.dot(hidden_delta , x1) + lambdaval*w1)/size_data
    w2_gradient = (w2_error + lambdaval*w2)/size_data



    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    # obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    obj_grad = np.array([])
    obj_grad = np.concatenate((w1_gradient.flatten(),w2_gradient.flatten()), axis = 0) 
    print("Objective value ",obj_val)
    return (obj_val, obj_grad)
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

    labels = np.array([])
    # Your code here

    x1=np.concatenate((np.ones((data.shape[0],1)), data), axis=1)
    y1= np.matmul(x1,w1.T)
    s1= sigmoid(y1)

    x2=np.concatenate((np.ones((data.shape[0],1)), s1), axis=1)
    y2=np.matmul(x2,w2.T)
    s2= sigmoid(y2)
    labels = np.argmax(s2.T,axis=0)
    labels= labels.reshape(labels.shape[0],1)

    #//--------------
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

    train_y = train_y.reshape(train_y.shape[0],1)
    valid_y = valid_y.reshape(valid_y.shape[0],1)
    test_y = test_y.reshape(test_y.shape[0],1)
    
    return train_x, train_y, valid_x, valid_y, test_x, test_y

"""**************Neural Network Script Starts here********************************"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()
#  Train Neural Network
# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]
# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 256
# set the number of nodes in output unit
n_class = 2

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);
# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)
# set the regularization hyper-parameter
lambdaval = 10
args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example
opts = {'maxiter' :50}    # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)
params = nn_params.get('x')
#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

#Test the computed parameters
predicted_label = nnPredict(w1,w2,train_data)
#find the accuracy on Training Dataset
print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')
predicted_label = nnPredict(w1,w2,validation_data)
#find the accuracy on Validation Dataset
print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')
predicted_label = nnPredict(w1,w2,test_data)
#find the accuracy on Validation Dataset
print('\n Test set Accuracy:' +  str(100*np.mean((predicted_label == test_label).astype(float))) + '%')
