import nnScript
import json


class ExperimentResult:
    def __init__(self, reg, n_hidden, train_acc, validation_acc, test_acc, train_time):
        self.regularization = reg
        self.hidden = n_hidden
        self.train_acc = train_acc
        self.validation_acc = validation_acc
        self.test_acc = test_acc
        self.training_time = train_time


exp_result_list = []
for i in range(0, 61, 10):
    j = 2
    p = 1

    while p <= 7:
        j = pow(2, p)
        train, valid, test, train_time = nnScript.runScript(i, j)
        exp_result_list.append(ExperimentResult(i, j, train, valid, test, train_time))
        # experiments["Lambda=" + str(i) + " and hidden = " + str(j)] = (nnScript.runScript(i, j))
        p += 1

# max_acc = max(experiments, key=experiments.get)
jsonStr = json.dumps([obj.__dict__ for obj in exp_result_list])
f = open("results.json", "w")
f.write(jsonStr)
f.close()


