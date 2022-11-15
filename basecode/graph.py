import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np



with open('results.json') as f:
  data = json.load(f)
  

# print(data)

fig = plt.figure(figsize = (10, 5))

x_axis = []
y_axis_train_acc = []
y_axis_validation_acc = []
y_axis_test_acc = []

x_tick_list = []


for x in range(len(data)):
  if(data[x]["regularization"] % 10 == 0):
    x_tick_list.append(data[x]["regularization"])
    x_axis.append(data[x]["regularization"])
    y_axis_train_acc.append(float(data[x]["train_acc"]))
    y_axis_validation_acc.append(float(data[x]["validation_acc"]))
    y_axis_test_acc.append(float(data[x]["test_acc"]))
  
  
barWidth = 1


br1 = np.arange(len(y_axis_train_acc))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]
 
# Make the plot
plt.bar(br1, y_axis_train_acc, color ='r', width = barWidth,
        edgecolor ='grey', label ='train_acc')
plt.bar(br2, y_axis_validation_acc, color ='g', width = barWidth,
        edgecolor ='grey', label ='validation_acc')
plt.bar(br3, y_axis_test_acc, color ='b', width = barWidth,
        edgecolor ='grey', label ='test_acc')


plt.xlabel('Regularization', fontweight ='bold', fontsize = 15)
plt.ylabel('Acc', fontweight ='bold', fontsize = 15)
plt.xticks([r + barWidth for r in range(len(x_tick_list))],
        x_axis)

  
plt.legend()
plt.show()
  
  



  
# print(x_axis, "x_axis")
# print(y_axis_train_acc, "y_axis_train_acc")
# print(y_axis_validation_acc, "y_axis_validation_acc")
# print(y_axis_test_acc, "y_axis_test_acc")