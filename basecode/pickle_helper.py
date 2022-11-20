import pickle


def save_model(sel_features, optimal_hidden, w1, w2, optimal_lambda):
    result_map = {"selected features": sel_features, "n_hidden": optimal_hidden, "w1": w1, "w2": w2,
                  "Optimal_lambda": optimal_lambda}
    pickle.dump(result_map, open("nnScriptModel.pickle", "wb"))
