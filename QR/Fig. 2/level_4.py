import sys, time
import numpy as np
import pandas as pd
import numpy.random as rgt
from pathlib import Path
from scipy.stats import norm, truncnorm, expon, poisson
from conquer.linear_model import low_dim, high_dim

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from selectinf.base import selected_targets
from selectinf.QR_lasso import QR_lasso
from selectinf.randomization import randomization
from selectinf.approx_reference import approximate_grid_inference
from selectinf.exact_reference import exact_grid_inference

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
coverage_exact = []
length_exact = []
recall_exact = []
F1_select_exact = []
F1_infere_exact = []

for i in range(reps):
    print(i)

    # generate sample
    X = rgt.multivariate_normal(mean=mu, cov=Sig, size=n)
    Y = X.dot(beta) + rgt.normal(loc=0, scale=2, size=n) - norm.ppf(tau, loc=0, scale=2)

    # true beta set
    nonzero_set = np.nonzero(beta)[0]
    zero_set = np.array([i for i in range(p) if i not in nonzero_set])

    # ------------------------- randomized -------------------------
    # selection
    randomizer = randomization.isotropic_gaussian(shape=(p,),
                                                  scale=np.sqrt(0.25 * 4) * (1 / np.sqrt(n)))
    conv = QR_lasso(X,
                    Y,
                    tau=tau,
                    randomizer=randomizer,
                    Lambda=0.5 * np.sqrt(np.log(p) / n))
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

    # confidence interval
    beta_target = np.linalg.pinv(X[:, selected_set]).dot(X.dot(beta))  # target
    exact_grid_inf = exact_grid_inference(query_spec, target_spec)
    lci, uci = exact_grid_inf._intervals(level=0.90)

    # coverage
    coverage = (lci < beta_target) * (uci > beta_target)
    coverage_exact.append(np.mean(coverage))
    print(np.mean(coverage))

    # length
    length = uci - lci
    length_exact.append(np.mean(length))

    # recall base on selection
    recall = recall_calculate(selected_set, nonzero_set, zero_set)
    recall_exact.append(recall)

    # F1 score base on selection
    F1_select = sensitivity_calculate(selected_set, nonzero_set, zero_set)
    F1_select_exact.append(F1_select)
    print(F1_select)

    # F1 score base on inference
    selected_infere = np.zeros(p)
    selected_infere[selected_set] = (lci > 0) | (uci < 0)
    F1_infere = sensitivity_calculate(np.nonzero(selected_infere)[0], nonzero_set, zero_set)
    F1_infere_exact.append(F1_infere)

# summary
print(np.mean(recall_exact))
print(np.mean(coverage_exact))
print(np.mean(length_exact))
results = pd.DataFrame(np.column_stack((F1_select_exact, F1_infere_exact, coverage_exact, length_exact, recall_exact)),
                       columns = ['F1_scores_exact', 'F1_infere_exact', 'coverage_exact', 'length_exact', 'recall_exact'])
results.to_csv('results_level_rmd_10.csv', index=False)

