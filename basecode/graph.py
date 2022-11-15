import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np


with open('results.json') as f:
    data = json.load(f)


plot_data = {}

for x in range(len(data)):
    if(data[x]["regularization"] % 10 == 0):
        if data[x]["hidden"] not in plot_data.keys():
            plot_data[data[x]["hidden"]] = [data[x]]
        else:
            plot_data[data[x]["hidden"]].append(data[x])

for key, value in plot_data.items():
    x_axis = []
    y_axis_train_acc = []
    y_axis_validation_acc = []
    y_axis_test_acc = []
    for val in value:
        x_axis.append(val["regularization"])
        y_axis_train_acc.append(float(val["train_acc"]))
        y_axis_validation_acc.append(float(val["validation_acc"]))
        y_axis_test_acc.append(float(val["test_acc"]))
        
    barWidth = 1
    
    fig = plt.figure(figsize=(6, 5))    
    plt.xlabel('Regularization', fontweight ='bold', fontsize = 15)
    plt.ylabel('Acc', fontweight ='bold', fontsize = 15)
    plt.xticks()
    plt.title("Regularization vs Acc for hidden layer " + str(key))
    n = len(x_axis)
    width = 0.25
    plt.bar(np.arange(len(y_axis_train_acc)), y_axis_train_acc, color='blue',
                width=width, edgecolor='black',
                label='y_axis_train_acc')

    plt.bar(np.arange(len(y_axis_validation_acc)) + width, y_axis_validation_acc, color='yellow',
                width=width, edgecolor='black',
                label='y_axis_validation_acc')
    plt.bar(np.arange(len(y_axis_test_acc)) + width*2, y_axis_test_acc, color='green',
                width=width, edgecolor='black',
                label='y_axis_test_acc')
    # plt.xlim(0, 100)
    plt.ylim(0,100)
    plt.legend()
    plt.show()
