import numpy as np
import pandas as pd
import numpy.random as rgt
from conquer.conquer.linear import low_dim, high_dim
#from selectinf.regreg_QR.QR_low_dim import low_dim
#from selectinf.regreg_QR.QR_high_dim import high_dim
from QR.selectinf.base import selected_targets
from QR.selectinf.QR_lasso import QR_lasso, QR_scad, QR_mcp
from QR.selectinf.randomization import randomization
from QR.selectinf.exact_reference import exact_grid_inference
from QR.selectinf.regreg_QR.QR_population import *
import matplotlib.pyplot as plt

# generate covariance matrix: \Simga_{j,k} = 0.5^{|j-k|}
def cov_generate(std, rho):
    p = len(std)
    R = np.abs(np.subtract.outer(np.arange(p), np.arange(p)))
    return np.outer(std, std) * (rho ** R)

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
reps = 100
tau = 0.7 #tau-th population conditional quantile of y given x
n, p = 800, 201
mu, Sig = np.zeros(p - 1), cov_generate(np.ones(p - 1), 0.5)
gamma = np.zeros(p - 1)
gamma[0:5] = 1

# record the results
coverage_naive = []
coverage_split = []
coverage_exact_lasso = []
coverage_exact_scad = []
coverage_exact_mcp = []

length_naive = []
length_split = []
length_exact_lasso = []
length_exact_scad = []
length_exact_mcp = []

F1_select_exact_lasso = []
F1_infere_exact_lasso = []
F1_select_exact_scad = []
F1_infere_exact_scad = []
F1_select_exact_mcp = []
F1_infere_exact_mcp = []


for i in range(reps):
    print(i)

    # generate sample
    X_tilde = rgt.multivariate_normal(mean=mu, cov=Sig, size=n)
    Y = 0.2 + X_tilde.dot(gamma) + rgt.normal(loc=0, scale=2, size=n)
    X = np.c_[np.ones(n), X_tilde] 

    # true beta set
    beta = np.append(0.2 + norm.ppf(tau, loc=0, scale=2), gamma)
    nonzero_set = np.nonzero(beta)[0]
    zero_set = np.array([i for i in range(p) if i not in nonzero_set])

    # ---------------------------- naive ---------------------------
    # selection
    select_h = max(0.05, np.sqrt(tau * (1 - tau)) * (np.log(p) / n) ** 0.25) #bandwidth for Gaussian kernel for selection stage
    selected_fit = high_dim(X,
                            Y,
                            intercept=False).l1(h=select_h,
                                                tau=tau,
                                                kernel="Gaussian",
                                                Lambda=0.6 * np.sqrt(np.log(p) / n),
                                                standardize=False)
    selected_set = np.nonzero(selected_fit['beta'])[0]
    selected_size = len(selected_set)

    if selected_size != 0:
        # inference
        infere_h = ((selected_size + np.log(n)) / n) ** 0.4 #bandwidth for Gaussian kernel for inference stage
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

    if selected_size != 0:
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

        # length
        length = uci - lci
        length_split.append(np.mean(length))

    # ------------------------- randomized LASSO-------------------------
    # selection
    randomizer = randomization.isotropic_gaussian(shape=(p,),
                                                  scale=(1 / np.sqrt(n)))
    conv = QR_lasso(X,
                    Y,
                    tau=tau,
                    randomizer=randomizer,
                    Lambda=0.8 * np.sqrt(np.log(p) / n))
    conv.fit()
    conv.setup_inference()
    query_spec = conv.specification
    target_spec, _ = selected_targets(X,
                                      Y,
                                      tau=tau,
                                      solution=conv.observed_soln) #return unpenalized problem beta, Sigma, J_{EE^{-1} and alternatives 

    # nonzero set of penalized estimator
    selected_set = np.nonzero(conv.observed_soln)[0]
    selected_size = len(selected_set)

    if selected_size != 0:
        # inference
        exact_grid_inf = exact_grid_inference(query_spec, target_spec)

        # confidence interval
        beta_target = np.linalg.pinv(X[:, selected_set]).dot(X.dot(beta))  # target
        lci, uci = exact_grid_inf._intervals(level=0.90)

        # coverage
        coverage = (lci < beta_target) * (uci > beta_target)
        coverage_exact_lasso.append(np.mean(coverage))

        # length
        length = uci - lci
        length_exact_lasso.append(np.mean(length))

        # F1 score base on selection
        F1_select = sensitivity_calculate(selected_set, nonzero_set, zero_set)
        F1_select_exact_lasso.append(F1_select)

        # F1 score base on inference
        selected_infere = np.zeros(p)
        selected_infere[selected_set] = (lci > 0) | (uci < 0)
        F1_infere = sensitivity_calculate(np.nonzero(selected_infere)[0], nonzero_set, zero_set)
        F1_infere_exact_lasso.append(F1_infere)

    # ------------------------- randomized SCAD-------------------------
    # selection

    randomizer = randomization.isotropic_gaussian(shape=(p,),
                                                  scale=(1 / np.sqrt(n)))
    conv = QR_scad(X,
                    Y,
                    tau=tau,
                    randomizer=randomizer,
                    Lambda=0.8 * np.sqrt(np.log(p) / n))
    conv.fit()
    conv.setup_inference()
    query_spec = conv.specification
    target_spec, _ = selected_targets(X,
                                      Y,
                                      tau=tau,
                                      solution=conv.observed_soln) #return unpenalized problem beta, Sigma, J_{EE^{-1} and alternatives

    # nonzero set of penalized estimator
    selected_set = np.nonzero(conv.observed_soln)[0]
    selected_size = len(selected_set)

    if selected_size != 0:
        # inference
        exact_grid_inf = exact_grid_inference(query_spec, target_spec)

        # confidence interval
        beta_target = np.linalg.pinv(X[:, selected_set]).dot(X.dot(beta))  # target
        lci, uci = exact_grid_inf._intervals(level=0.90)

        # coverage
        coverage = (lci < beta_target) * (uci > beta_target)
        coverage_exact_scad.append(np.mean(coverage))

        # length
        length = uci - lci
        length_exact_scad.append(np.mean(length))

        # F1 score base on selection
        F1_select = sensitivity_calculate(selected_set, nonzero_set, zero_set)
        F1_select_exact_scad.append(F1_select)

        # F1 score base on inference
        selected_infere = np.zeros(p)
        selected_infere[selected_set] = (lci > 0) | (uci < 0)
        F1_infere = sensitivity_calculate(np.nonzero(selected_infere)[0], nonzero_set, zero_set)
        F1_infere_exact_scad.append(F1_infere)

    # ------------------------- randomized mcp-------------------------
    # selection

    randomizer = randomization.isotropic_gaussian(shape=(p,),
                                                  scale=(1 / np.sqrt(n)))
    conv = QR_mcp(X,
                   Y,
                   tau=tau,
                   randomizer=randomizer,
                   Lambda=0.8 * np.sqrt(np.log(p) / n))
    conv.fit()
    conv.setup_inference()
    query_spec = conv.specification
    target_spec, _ = selected_targets(X,
                                      Y,
                                      tau=tau,
                                      solution=conv.observed_soln)  # return unpenalized problem beta, Sigma, J_{EE^{-1} and alternatives

    # nonzero set of penalized estimator
    selected_set = np.nonzero(conv.observed_soln)[0]
    selected_size = len(selected_set)
    
    if selected_size != 0:
        # inference
        exact_grid_inf = exact_grid_inference(query_spec, target_spec)

        # confidence interval
        beta_target = np.linalg.pinv(X[:, selected_set]).dot(X.dot(beta))  # target
        lci, uci = exact_grid_inf._intervals(level=0.90)

        # coverage
        coverage = (lci < beta_target) * (uci > beta_target)
        coverage_exact_mcp.append(np.mean(coverage))

        # length
        length = uci - lci
        length_exact_mcp.append(np.mean(length))

        # F1 score base on selection
        F1_select = sensitivity_calculate(selected_set, nonzero_set, zero_set)
        F1_select_exact_mcp.append(F1_select)

        # F1 score base on inference
        selected_infere = np.zeros(p)
        selected_infere[selected_set] = (lci > 0) | (uci < 0)
        F1_infere = sensitivity_calculate(np.nonzero(selected_infere)[0], nonzero_set, zero_set)
        F1_infere_exact_mcp.append(F1_infere)

# results
results = pd.DataFrame(np.column_stack((F1_select_exact_lasso, F1_infere_exact_lasso, coverage_naive, coverage_split, coverage_exact_lasso, coverage_exact_scad, coverage_exact_mcp,
                                        length_naive, length_split, length_exact_lasso, length_exact_scad, length_exact_mcp)),
                       columns = ['F1_select_exact_lasso', 'F1_infere_exact_lasso', 'coverage_naive', 'coverage_split', 'coverage_exact_lasso', 'coverage_exact_scad', 'coverage_exact_mcp',
                                  'length_naive', 'length_split', 'length_exact_lasso', 'length_exact_scad', 'length_exact_mcp'])
print(results)


plt.figure()
plt.boxplot(results[['coverage_naive', 'coverage_split', 'coverage_exact_lasso', 'coverage_exact_scad', 'coverage_exact_mcp']],labels=['naive', 'split', 'exact_L', 'exact_S', 'exact_M'])
plt.title('Coverage')
plt.ylabel('Coverage rate')
plt.savefig('coverage1.png')
plt.close()



plt.figure()
plt.boxplot(results[['length_naive', 'length_split', 'length_exact_lasso', 'length_exact_scad', 'length_exact_mcp']], labels=['naive', 'split', 'exact_L', 'exact_S', 'exact_M'])
plt.title('Length')
plt.ylabel('CI length')
plt.savefig('length1.png')
plt.close()