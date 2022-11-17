import json

import matplotlib.pyplot as plt
import numpy as np

with open('results.json') as f:
    data = json.load(f)

plot_data = {}
hidden_vs_time = {}

for x in range(len(data)):
    if data[x]["regularization"] not in hidden_vs_time.keys():
        hidden_vs_time[data[x]["regularization"]] = [data[x]]
    else:
        hidden_vs_time[data[x]["regularization"]].append(data[x])
    if data[x]["regularization"] % 10 == 0:
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
    width = 0.25


    fig = plt.figure(figsize=(6, 5))
    plt.xlabel('Regularization', fontweight='bold', fontsize=15)
    plt.ylabel('Acc', fontweight='bold', fontsize=15)
    plt.xticks(np.arange(0, 7), x_axis)
    plt.title("Regularization vs Acc for " + str(key) + " hidden neurons")
    plt.bar(np.arange(len(y_axis_train_acc)), y_axis_train_acc, color='blue',
            width=width, edgecolor='black',
            label='y_axis_train_acc')

    plt.bar(np.arange(len(y_axis_validation_acc)) + width, y_axis_validation_acc, color='yellow',
            width=width, edgecolor='black',
            label='y_axis_validation_acc')
    plt.bar(np.arange(len(y_axis_test_acc)) + width * 2, y_axis_test_acc, color='green',
            width=width, edgecolor='black',
            label='y_axis_test_acc')
    plt.ylim(0, 100)
    plt.yticks(np.arange(0, 101, 5))
    # plt.legend()
    # plt.show()
    plt.savefig("graphs/"+str(key)+"_hidden_neurons.png")
    

ftr = [3600,60,1]

    
    
for key1, value1 in hidden_vs_time.items():
    x_axis_regularization = []
    y_axis_time = []
    for val in value1:
        x_axis_regularization.append(val["hidden"])
        new_time = sum([a*b for a,b in zip(ftr, map(int,val["training_time"].split(':')))])
        y_axis_time.append(new_time)
    barWidth = 1
    plt.bar(x_axis_regularization,y_axis_time, width=barWidth, edgecolor='black',)
    plt.xlabel('Hidden Neurons')
    plt.xticks(np.arange(0, 130, 4))
    plt.ylabel('Time (s)')
    plt.show()

