import sys, time
import numpy as np
import pandas as pd
import numpy.random as rgt
from pathlib import Path
from scipy.stats import norm, truncnorm, expon, poisson

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from selectinf.base import selected_targets
from selectinf.QR_lasso import QR_lasso
from selectinf.randomization import randomization
from selectinf.approx_reference import approximate_grid_inference
from selectinf.exact_reference import exact_grid_inference
from selectinf.regreg_QR.QR_low_dim import low_dim
from selectinf.regreg_QR.QR_high_dim import high_dim

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
selectiveInference = importr('selectiveInference')
rdiag = robjects.r['diag']

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
alpha = 0.1
reps = 500
tau = 0.7
n, p = 800, 200
mu, Sig = np.zeros(p), cov_generate(np.ones(p), 0.5)
beta = np.zeros(p)
beta[0:5] = 0.5

# record the results
coverage_naive = []
length_naive = []
F1_select_naive = []
F1_infere_naive = []

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

    # ---------------------------- without randomization ---------------------------
    # selection
    select_h = max(0.05, np.sqrt(tau * (1 - tau)) * (np.log(p) / n) ** 0.25)
    selected_fit = high_dim(X,
                            Y,
                            np.zeros(p),
                            intercept=False).l1(h=select_h,
                                                tau=tau,
                                                kernel="Gaussian",
                                                Lambda=np.sqrt(np.log(p) / n))
    selected_lam = selected_fit['lambda'][0]
    selected_set = np.nonzero(selected_fit['beta'])[0]
    selected_size, nonselected_size = len(selected_set), p - len(selected_set)
    # sE = selected_fit['subgrad'][selected_set]
    sE = np.sign(selected_fit['beta'])[selected_set]
    print(selected_set)
    if selected_size == 0:
        continue

    # inference
    # infere_h = ((selected_size + np.log(n)) / n) ** 0.4
    infere_h = select_h
    infere_problem = low_dim(X[:, selected_set],
                             Y,
                             intercept=False)
    infere_model = infere_problem.fit(h=infere_h,
                                      tau=tau,
                                      kernel="Gaussian",
                                      beta0=selected_fit['beta'][selected_set])
    beta_mle = infere_model['beta']

    # matrix
    beta_bar = np.zeros(p)
    beta_bar[selected_set] = beta_mle
    V, J, grad = low_dim(X,
                         Y,
                         intercept=False).covariance(beta_bar,
                                                     h=infere_h,
                                                     tau=tau,
                                                     kernel="Gaussian").values()

    J_active, J_inactive = J[selected_set][:, selected_set], np.delete(J[:, selected_set], selected_set, axis=0)
    V_active, V_inactive = V[selected_set][:, selected_set], np.delete(V[:, selected_set], selected_set, axis=0)
    J_active_inv, V_active_inv = np.linalg.inv(J_active), np.linalg.inv(V_active)
    cov_active = J_active_inv.dot(V_active.dot(J_active_inv))

    # inference
    lci, uci = [], []
    for idx, j in enumerate(selected_set):
        sigma2_j = cov_active[idx, idx]
        Gamma_j = np.r_[np.delete(grad, selected_set) + (V_inactive.dot(V_active_inv.dot(J_active)) - J_inactive).dot(beta_mle),
                        beta_mle - beta_mle[idx] * cov_active[:, idx] / sigma2_j].reshape(p, 1)
        Gamma_j2 = (beta_mle - beta_mle[idx] * cov_active[:, idx] / sigma2_j).reshape(selected_size, 1)

        temp1 = np.c_[np.diag(np.ones(nonselected_size)), J_inactive - V_inactive.dot(V_active_inv.dot(J_active))]
        temp2 = np.sqrt(n) * selected_lam * sE.reshape(selected_size, 1)
        G = np.r_[- np.diag(sE).dot(cov_active[:, idx] / sigma2_j),
                  - J_inactive.dot(cov_active[:, idx] / sigma2_j) + V_inactive.dot(J_active_inv[:, idx] / sigma2_j),
                  J_inactive.dot(cov_active[:, idx] / sigma2_j) - V_inactive.dot(J_active_inv[:, idx] / sigma2_j)]
        K = np.r_[np.diag(sE).dot(np.sqrt(n) * Gamma_j2) - np.diag(sE).dot(J_active_inv.dot(temp2)),
                  np.sqrt(n) * selected_lam + temp1.dot(np.sqrt(n) * Gamma_j) - J_inactive.dot(J_active_inv.dot(temp2)),
                  np.sqrt(n) * selected_lam - temp1.dot(np.sqrt(n) * Gamma_j) + J_inactive.dot(J_active_inv.dot(temp2))]
        c = K.reshape(len(K), 1) / (np.sqrt(n) * G.reshape(len(G), 1))
        l = np.max(c[np.where(G < 0)[0]])
        u = np.min(c[np.where(G > 0)[0]])

        rsigma2n = sigma2_j / n
        rZ = robjects.FloatVector([beta_mle[idx], beta_mle[idx]])
        rA = rdiag(robjects.IntVector([-1, 1]))
        rb = robjects.FloatVector([-l, u])
        rSigma = robjects.r['matrix'](robjects.FloatVector([rsigma2n, rsigma2n, rsigma2n, rsigma2n]), nrow = 2)
        reta = robjects.FloatVector([.5, .5])
        robjects.r('''
                library(selectiveInference)
                TG_interval = TG.interval
                ''')
        r_TG_interval = robjects.globalenv['TG_interval']
        rinterval = r_TG_interval(rZ, rA, rb, reta, rSigma, alpha=alpha, gridpts=100)[0]
        interval_l, interval_u = np.array(rinterval)

        lci.append(interval_l)
        uci.append(interval_u)


    # confidence interval
    beta_target = np.linalg.pinv(X[:, selected_set]).dot(X.dot(beta))  # target
    lci, uci = np.array(lci), np.array(uci)

    # coverage
    coverage = (lci < beta_target) * (uci > beta_target)
    coverage_naive.append(np.mean(coverage))

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

# summary
print(np.mean(F1_select_naive))
print(np.mean(coverage_naive))
coverage_naive = np.array(coverage_naive)
length_naive = np.array(length_naive)
print(sum(np.isfinite(length_naive)))
print(np.mean(length_naive[np.isfinite(length_naive)]))
results = pd.DataFrame(np.column_stack((F1_select_naive, F1_infere_naive, coverage_naive, length_naive)),
                       columns = ['F1_select_naive', 'F1_infere_naive', 'coverage_naive', 'length_naive'])
results.to_csv('results_mid_rdm0.csv', index=False)


