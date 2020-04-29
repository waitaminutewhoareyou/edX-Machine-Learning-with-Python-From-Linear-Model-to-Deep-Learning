"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture
from scipy.stats import multivariate_normal



def multivariate_pdf(x, mean,cov):
    return multivariate_normal(mean=mean, cov=cov).pdf([x])

def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    X, mu, var, p = np.array(X), np.array(mixture.mu), np.array(mixture.var), np.array(mixture.p)
    n, d = X.shape
    K, _ = mu.shape
    post = np.zeros((n, K))
    likelihood = 0
    for i in range(n):
        for j in range(K):
            post[i, j] = p[j]*multivariate_pdf(X[i, :], mu[j, :], var[j]*np.eye(d))
        likelihood += np.log(np.sum(post[i, :]))
    for row in post:
        row /= np.sum(row)

    return post, likelihood


def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    X, post = np.array(X), np.array(post)
    n, d = X.shape
    _, K = post.shape
    mu, p, variance = np.zeros((K, d)), [], []
    for k in range(K):
        mu[k, :] = (X.T @ post[:, k])/np.sum(post[:, k])
        p.append(np.mean(post[:, k]))
        variance.append((np.linalg.norm((X - mu[k, :]), axis=1)**2).T @ post[:, k]/(d*np.sum(post[:, k])))

    return GaussianMixture(mu, np.array(variance), np.array(p))
def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    prev_LL = None
    LL = None
    while prev_LL is None or  LL - prev_LL > 1e-6 * abs(LL):
        prev_LL = LL
        post, LL = estep(X, mixture)
        mixture = mstep(X,post)

    return mixture, post, LL

