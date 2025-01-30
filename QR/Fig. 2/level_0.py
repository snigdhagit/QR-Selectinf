import time
import numpy as np
import pandas as pd
import numpy.random as rgt
from scipy.stats import norm
from conquer.linear_model import low_dim, high_dim

# generate covariance matrix
def cov_generate(std, rho):
    p = len(std)
    R = np.abs(np.subtract.outer(np.arange(p), np.arange(p)))
    return np.outer(std, std) * (rho ** R)

# estimate AR covariance matrix
def cov_estmate(X, t = 100):
    ARrho = []
    for s in np.random.sample(t):
        Xr = X[int(s * n)]
        ARrho.append(np.corrcoef(Xr[1:], Xr[:-1])[0, 1])
    ARrho = np.mean(ARrho)
    ARcov = ARrho ** (np.abs(np.subtract.outer(np.arange(p), np.arange(p))))
    return ARcov

def sensitivity_calculate(selected_set, nonzero_set, zero_set):
    selected = np.zeros(p)
    selected[selected_set] = 1
    Ture_positive = np.size([element for element in np.where(selected != 0)[0] if element in nonzero_set])
    False_positive = np.size([element for element in np.where(selected != 0)[0] if element in zero_set])
    False_negative = np.size([element for element in np.where(selected == 0)[0] if element in nonzero_set])
    return Ture_positive / (Ture_positive + 0.5 * False_positive + 0.5 * False_negative)

def recall_calculate(selected_set, nonzero_set, zero_set):
    selected = np.zeros(p)
    selected[selected_set] = 1
    Ture_positive = np.size([element for element in np.where(selected != 0)[0] if element in nonzero_set])
    False_negative = np.size([element for element in np.where(selected == 0)[0] if element in nonzero_set])
    return Ture_positive / (Ture_positive + False_negative)

# set random seed
np.random.seed(2023)

# model setting
reps = 500
tau = 0.7
n, p = 800, 200
mu, Sig = np.zeros(p), cov_generate(np.ones(p), 0.5)
beta = np.zeros(p)
beta[0:5] = .5

# record the results
coverage_naive = []
length_naive = []
recall_naive = []
F1_select_naive = []
F1_infere_naive = []


for i in range(reps):
    print(i)

    # generate sample
    X = rgt.multivariate_normal(mean=mu, cov=Sig, size=n)
    Y = X.dot(beta) + rgt.normal(loc=0, scale=2, size=n) - norm.ppf(tau, loc=0, scale=2)

    # true beta set
    nonzero_set = np.nonzero(beta)[0]
    zero_set = np.array([i for i in range(p) if i not in nonzero_set])

    # ---------------------------- naive ---------------------------
    # selection
    select_h = max(0.05, np.sqrt(tau * (1 - tau)) * (np.log(p) / n) ** 0.25)
    selected_fit = high_dim(X,
                            Y,
                            intercept=False).l1(h=select_h,
                                                tau=tau,
                                                kernel="Gaussian",
                                                Lambda=0.5 * np.sqrt(np.log(p) / n),
                                                standardize=False)
    selected_set = np.nonzero(selected_fit['beta'])[0]
    selected_size = len(selected_set)
    print(selected_set)

    # inference
    # infere_h = ((selected_size + np.log(n)) / n) ** 0.4
    infere_h = select_h
    infere_model = low_dim(X[:, selected_set],
                           Y,
                           intercept=False).norm_ci(h=infere_h,
                                                    tau=tau,
                                                    alpha=0.1,
                                                    kernel="Gaussian",
                                                    standardize=False)

    # confidence interval
    beta_target = np.linalg.pinv(X[:, selected_set]).dot(X.dot(beta))  # target
    lci, uci = infere_model['normal_ci'][:, 0], infere_model['normal_ci'][:, 1]

    # coverage
    coverage = (lci < beta_target) * (uci > beta_target)
    coverage_naive.append(np.mean(coverage))

    # length
    length = uci - lci
    length_naive.append(np.mean(length))

    # recall base on selection
    recall = recall_calculate(selected_set, nonzero_set, zero_set)
    recall_naive.append(recall)

    # F1 score base on selection
    F1_select = sensitivity_calculate(selected_set, nonzero_set, zero_set)
    F1_select_naive.append(F1_select)

    # F1 score base on inference
    selected_infere = np.zeros(p)
    selected_infere[selected_set] = (lci > 0) | (uci < 0)
    F1_infere = sensitivity_calculate(np.nonzero(selected_infere)[0], nonzero_set, zero_set)
    F1_infere_naive.append(F1_infere)

# summary
print(np.mean(recall_naive))
print(np.mean(coverage_naive))
print(np.mean(length_naive))
results = pd.DataFrame(np.column_stack((F1_select_naive, F1_infere_naive, coverage_naive, length_naive, recall_naive)),
                       columns = ['F1_scores_naive', 'F1_infere_naive', 'coverage_naive', 'length_naive', 'recall_naive'])
results.to_csv('results_level_naive.csv', index=False)


