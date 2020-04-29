import numpy as np
import em
import common

X = np.loadtxt("test_incomplete.txt")
X_gold = np.loadtxt("test_complete.txt")

K = 12
# n, d = X.shape
# seed = 0

# TODO: Your code here
def best_run_em(X):
    K = 12
    dict = {}
    for seed in range(5):
        np.random.seed(seed)
        mixture, post = common.init(X, K, seed)
        mixture, post, LL = em.run(X, mixture, post)
        dict[LL] = (mixture, seed)
    return dict[min(dict.keys())]


if __name__ == "__main__":
    mixture, best_seed = best_run_em(X)
    np.random.seed(best_seed)
    X_pred = em.fill_matrix(X, mixture)
    print(common.rmse(X_gold, X_pred))