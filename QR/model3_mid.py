import time
import numpy as np
import pandas as pd
import numpy.random as rgt
from scipy.stats import norm, truncnorm, expon, poisson
from conquer.linear_model import low_dim, high_dim
from selectinf.base import selected_targets
from selectinf.QR_lasso import QR_lasso
from selectinf.randomization import randomization
from selectinf.approx_reference import approximate_grid_inference
from selectinf.exact_reference import exact_grid_inference
from selectinf.regreg_QR.QR_population import *

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
n, p = 800, 201
mu, Sig = np.zeros(p - 1), cov_generate(np.ones(p - 1), 0.5)
c = 0.5
def beta_u(u, p):
    nonzero = np.array([c * u, c * u, c, c, c])
    return np.concatenate((nonzero, np.zeros(p - len(nonzero))), axis=None)

# record the results
coverage_naive = []
coverage_split = []
coverage_exact = []
coverage_apprx = []

coverage_all_naive = []
coverage_all_split = []
coverage_all_exact = []
coverage_all_apprx = []

length_naive = []
length_split = []
length_exact = []
length_apprx = []

F1_select_naive = []
F1_select_split = []
F1_select_exact = []
F1_select_apprx = []

F1_infere_naive = []
F1_infere_split = []
F1_infere_exact = []
F1_infere_apprx = []

time_exact = []
time_apprx = []

for i in range(reps):
    print(i)

    # generate sample
    X_tilde = rgt.multivariate_normal(mean=mu, cov=Sig, size=n)
    X_tilde[:, 0:2] = rgt.uniform(0, 2, n * 2).reshape(n, 2)
    U = rgt.uniform(0, 1, n)
    Y = np.array([2 * c * U[j] + np.dot(X_tilde[j, :], beta_u(U[j], p - 1)) for j in range(n)])
    X = np.c_[np.ones(n), X_tilde]

    # true beta set
    beta = np.append(2 * c * tau, beta_u(tau, p - 1))
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
                                                Lambda=0.6 * np.sqrt(np.log(p) / n),
                                                standardize=False)
    selected_set = np.nonzero(selected_fit['beta'])[0]
    selected_size = len(selected_set)
    print(selected_set)

    # inference
    infere_h = ((selected_size + np.log(n)) / n) ** 0.4
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
    coverage_all_naive.append(np.all(coverage))

    # length
    length = uci - lci
    length_naive.append(np.mean(length))

    # F1 score base on selection
    F1_select = sensitivity_calculate(selected_set, nonzero_set, zero_set)
    F1_select_naive.append(F1_select)

    # F1 score base on inference
    selected_infere = np.zeros(p)
    selected_infere[selected_set] = (lci > 0) | (uci < 0)
    F1_infere = sensitivity_calculate(np.nonzero(selected_infere)[0], nonzero_set, zero_set)
    F1_infere_naive.append(F1_infere)

    # -------------------------- splitting -------------------------
    # splitting
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
                                                Lambda=0.6 * np.sqrt(np.log(p) / select_n),
                                                standardize=False)
    selected_set = np.nonzero(selected_fit['beta'])[0]
    selected_size = len(selected_set)
    print(selected_set)

    # inference
    infere_h = ((selected_size + np.log(infere_n)) / infere_n) ** 0.4
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
    coverage_all_split.append(np.all(coverage))

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

    # ------------------------- randomized -------------------------
    # selection
    randomizer = randomization.isotropic_gaussian(shape=(p,),
                                                  scale=(1 / np.sqrt(n)))
    conv = QR_lasso(X,
                    Y,
                    tau=tau,
                    randomizer=randomizer,
                    Lambda=0.6 * np.sqrt(np.log(p) / n))
    conv.fit()
    conv.setup_inference()
    query_spec = conv.specification
    target_spec, _ = selected_targets(X,
                                      Y,
                                      tau=tau,
                                      solution=conv.observed_soln)

    # nonzero set of penalized estimator
    selected_set = np.nonzero(conv.observed_soln)[0]
    print(selected_set)

    # ------- exact pivot --------
    # inference
    time1 = time.time()
    exact_grid_inf = exact_grid_inference(query_spec, target_spec)
    time2 = time.time()
    time_exact.append(time2 - time1)

    # confidence interval
    beta_target = np.linalg.pinv(X[:, selected_set]).dot(X.dot(beta))  # target
    lci, uci = exact_grid_inf._intervals(level=0.90)

    # coverage
    coverage = (lci < beta_target) * (uci > beta_target)
    coverage_exact.append(np.mean(coverage))
    coverage_all_exact.append(np.all(coverage))
    print(np.mean(coverage))

    # length
    length = uci - lci
    length_exact.append(np.mean(length))

    # F1 score base on selection
    F1_select = sensitivity_calculate(selected_set, nonzero_set, zero_set)
    F1_select_exact.append(F1_select)

    # F1 score base on inference
    selected_infere = np.zeros(p)
    selected_infere[selected_set] = (lci > 0) | (uci < 0)
    F1_infere = sensitivity_calculate(np.nonzero(selected_infere)[0], nonzero_set, zero_set)
    F1_infere_exact.append(F1_infere)

    # ------- approximate pivot --------
    # inference
    time3 = time.time()
    approximate_grid_inf = approximate_grid_inference(query_spec, target_spec)
    time4 = time.time()
    time_apprx.append(time4 - time3)

    # confidence interval
    lci, uci = approximate_grid_inf._intervals(level=0.90)

    # coverage
    coverage = (lci < beta_target) * (uci > beta_target)
    coverage_apprx.append(np.mean(coverage))
    coverage_all_apprx.append(np.all(coverage))

    # length
    length = uci - lci
    length_apprx.append(np.mean(length))

    # F1 score base on selection
    F1_select_apprx.append(F1_select)

    # F1 score base on inference
    selected_infere = np.zeros(p)
    selected_infere[selected_set] = (lci > 0) | (uci < 0)
    F1_infere = sensitivity_calculate(np.nonzero(selected_infere)[0], nonzero_set, zero_set)
    F1_infere_apprx.append(F1_infere)

# summary
summary = pd.DataFrame({'navie': [np.mean(F1_select_naive), np.mean(coverage_naive), np.mean(coverage_all_naive),
                                  np.mean(length_naive), np.mean(F1_infere_naive), np.nan],
                        'splitting':[np.mean(F1_select_split), np.mean(coverage_split), np.mean(coverage_all_split),
                                     np.mean(length_split), np.mean(F1_infere_split), np.nan],
                        'exact': [np.mean(F1_select_exact), np.mean(coverage_exact), np.mean(coverage_all_exact),
                                  np.mean(length_exact), np.mean(F1_infere_exact), np.mean(time_exact)],
                        'approximate': [np.mean(F1_select_apprx), np.mean(coverage_apprx), np.mean(coverage_all_apprx),
                                        np.mean(length_apprx), np.mean(F1_infere_apprx), np.mean(time_apprx)]},
                       index=['F1 score in selection', 'Coverage rate', 'Joint coverage rate','CI length',
                              'F1 score in inference', 'Inference time']).round(4)
results = pd.DataFrame(np.column_stack((F1_select_naive, F1_select_split, F1_select_exact, F1_select_apprx,
                                        coverage_naive, coverage_split, coverage_exact, coverage_apprx,
                                        length_naive, length_split, length_exact, length_apprx,
                                        F1_infere_naive, F1_infere_split, F1_infere_exact, F1_infere_apprx,
                                        time_exact, time_apprx)),
                       columns = ['F1_select_naive', 'F1_select_split', 'F1_select_exact', 'F1_select_apprx',
                                  'coverage_naive', 'coverage_split', 'coverage_exact', 'coverage_apprx',
                                  'length_naive', 'length_split', 'length_exact', 'length_apprx',
                                  'F1_infere_naive', 'F1_infere_split', 'F1_infere_exact', 'F1_infere_apprx',
                                  'time_exact', 'time_apprx'])
summary.to_csv('summary_model3_2.csv', index=False)
results.to_csv('results_model3_2.csv', index=False)

