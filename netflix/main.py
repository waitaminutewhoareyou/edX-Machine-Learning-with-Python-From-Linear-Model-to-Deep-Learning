import numpy as np
import kmeans
import common
import naive_em
import em

X = np.loadtxt("toy_data.txt")

# TODO: Your code here
def run_kmean(X):
    for K in [1,2,3,4]:
        cost_list = []
        for seed in range(5):
            mixture, post = common.init(X, K, seed)
            mixture, post, cost = kmeans.run(X, mixture, post)
            cost_list.append(cost)
            #common.plot(X, mixture, post, "{} means with seed{}".format(K, seed))
        print("The cost of {} cluster is".format(K), min(cost_list))
        best_seed = np.argmin(cost_list)
        for seed_ in [best_seed]:
            mixture, post = common.init(X, K, int(seed_))
            mixture, post, cost = kmeans.run(X, mixture, post)
            common.plot(X, mixture, post, "{} means with seed{}".format(K, seed_))
    return "Done"

def run_naive_em(X):
    for K in [1,2,3,4]:
        likelihood_ls = []
        for seed in range(5):
            mixture, post = common.init(X, K, seed)
            mixture, post, LL = naive_em.run(X,mixture,post)
            likelihood_ls.append(LL)


        print("The likelihood of {} cluster is".format(K), max(likelihood_ls))
        best_seed = np.argmax(likelihood_ls)
        for seed_ in [best_seed]:
            mixture, post = common.init(X, K, int(seed_))
            mixture, post, LL= naive_em.run(X, mixture, post)
            common.plot(X, mixture, post, "{} mixtures with seed{}".format(K, seed_))
    return "Done"

def select_best_bic(X):
    bic_ls = []
    for K in [1,2,3,4]:
        likelihood_ls = []
        bic_ls_seed = []

        for seed in range(5):
            mixture, post = common.init(X, K, seed)
            mixture, post, LL = naive_em.run(X,mixture,post)
            likelihood_ls.append(LL)
            bic_ls_seed.append(common.bic(X, mixture, LL))

        best_seed = np.argmax(bic_ls_seed)

        mixture, post = common.init(X, K, int(best_seed))
        mixture, post, LL = naive_em.run(X, mixture, post)
        bic_ls.append(common.bic(X,mixture,LL))
    print("The best K is {} with bic {}".format(np.argmax(bic_ls)+1, max(bic_ls)))
    return "Done"

def run_em(X):
    for K in [1, 12]:
        likelihood_ls = []
        for seed in range(5):
            mixture, post = common.init(X, K, seed)
            mixture, post, LL = em.run(X, mixture, post)
            likelihood_ls.append(LL)


        print("The likelihood of {} cluster is".format(K), max(likelihood_ls))
    return "Done"


def best_run_em(X):
    K = 12
    dict = {}
    likelihood_ls = []
    for seed in range(5):
        mixture, post = common.init(X, K, seed)
        mixture, post, LL = em.run(X, mixture, post)
        dict[LL] = mixture
    return dict[min(dict.keys())]


if __name__ == "__main__":
    # run_kmean(X)
    # run_naive_em(X)
    # select_best_bic(X)
    X = np.loadtxt("netflix_incomplete.txt")
    X_gold = np.loadtxt("netflix_complete.txt")
    mixture = best_run_em(X)
    X_pred = em.fill_matrix(X, mixture)
    common.rmse(X_gold, X_pred)