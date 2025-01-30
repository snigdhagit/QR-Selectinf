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

# set random seed
np.random.seed(2023)

# model setting
reps = 500
tau = 0.7
n, p = 800, 200
mu, Sig = np.zeros(p), cov_generate(np.ones(p), 0.5)
beta = np.zeros(p)
beta[0:5] = 1

# record the results
coverage_split = []
length_split = []
F1_select_split = []
F1_infere_split = []

for i in range(reps):
    print(i)

    # generate sample
    X = rgt.multivariate_normal(mean=mu, cov=Sig, size=n)
    Y = X.dot(beta) + rgt.normal(loc=0, scale=2, size=n) - norm.ppf(tau, loc=0, scale=2)
    # Y = X.dot(beta) + expon.rvs(loc=0, scale=2, size=n) - expon.ppf(tau, loc=0, scale=2)
    # Y = X.dot(beta) + poisson.rvs(loc=0, mu=2, size=n) - poisson.ppf(tau, loc=0, mu=2)

    # true beta set
    nonzero_set = np.nonzero(beta)[0]
    zero_set = np.array([i for i in range(p) if i not in nonzero_set])

    # ---------------------------- splitting ---------------------------
    sample_proportion = 2 / 3
    select_n = int(sample_proportion * n)
    infere_n = n - select_n
    index_select = np.random.choice(n, select_n, replace=False)
    index_infere = np.array([i for i in range(n) if i not in index_select])
    X_select, Y_select = X[index_select, :], Y[index_select]
    X_infere, Y_infere = X[index_infere, :], Y[index_infere]

    # selection
    select_h = max(0.05, np.sqrt(tau * (1 - tau)) * (np.log(p) / select_n) ** 0.25)
    selected_fit = high_dim(X_select,
                            Y_select,
                            intercept=False).l1(h=select_h,
                                                tau=tau,
                                                kernel="Gaussian",
                                                Lambda=np.sqrt(np.log(p) / select_n),
                                                standardize=False)
    selected_set = np.nonzero(selected_fit['beta'])[0]
    selected_size = len(selected_set)
    print(selected_set)
    if selected_size == 0:
        continue

    # inference
    # infere_h = ((selected_size + np.log(infere_n)) / infere_n) ** 0.4
    infere_h = select_h
    infere_model = low_dim(X_infere[:, selected_set],
                           Y_infere,
                           intercept=False).norm_ci(h=infere_h,
                                                    tau=tau,
                                                    alpha=0.1,
                                                    kernel="Gaussian",
                                                    standardize=False)

    # confidence interval
    beta_target = np.linalg.pinv(X_infere[:, selected_set]).dot(X_infere.dot(beta))  # target
    lci, uci = infere_model['normal_ci'][:, 0], infere_model['normal_ci'][:, 1]

    # coverage
    coverage = (lci < beta_target) * (uci > beta_target)
    coverage_split.append(np.mean(coverage))
    print(np.mean(coverage))

    # length
    length = uci - lci
    length_split.append(np.mean(length))

    # F1 score base on selection
    F1_select = sensitivity_calculate(selected_set, nonzero_set, zero_set)
    F1_select_split.append(F1_select)

    # F1 score base on inference
    selected_infere = np.zeros(p)
    selected_infere[selected_set] = (lci > 0) | (uci < 0)
    F1_infere = sensitivity_calculate(np.nonzero(selected_infere)[0], nonzero_set, zero_set)
    F1_infere_split.append(F1_infere)

# summary
print(np.mean(F1_select_split))
print(np.mean(coverage_split))

print(sum(np.isfinite(np.array(length_split))))
print(np.mean(length_split))

results = pd.DataFrame(np.column_stack((F1_select_split, F1_infere_split, coverage_split, length_split)),
                       columns = ['F1_scores_split', 'F1_infere_split', 'coverage_split', 'length_split'])
results.to_csv('results_high_split.csv', index=False)


