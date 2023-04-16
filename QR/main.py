import numpy as np
import numpy.random as rgt
from scipy.stats import norm
from scipy.optimize import fsolve
from selectinf.base import selected_targets
from selectinf.QR_lasso import QR_lasso
from selectinf.randomization import randomization
from selectinf.approx_reference import approximate_grid_inference
from selectinf.exact_reference import exact_grid_inference
from selectinf.regreg_QR.QR_population import *

# set random seed
np.random.seed(2023)
np.set_printoptions(threshold=np.inf)

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

num_set = []
coverage_exact1 = []
coverage_approx1 = []
coverage_exact2 = []
coverage_approx2 = []
length_exact = []
length_approx = []

n, p = 800, 200
mu, Sig = np.zeros(p), cov_generate(np.ones(p), 0.5)
beta = np.zeros(p)
beta[0:5] = 1
tau = 0.75

reps = 100
for i in range(reps):
    print(i)
    # generate sample
    X = rgt.multivariate_normal(mean=mu, cov=Sig, size=n)
    Y = X.dot(beta) + rgt.normal(loc=0, scale=1, size=n) - norm.ppf(tau, loc=0, scale=1)

    # inference
    randomizer = randomization.isotropic_gaussian(shape=(p,), scale=0.75*(1/np.sqrt(n)))
    conv = QR_lasso(X,
                    Y,
                    tau=tau,
                    randomizer=randomizer)
    conv.fit()
    conv.setup_inference()
    query_spec = conv.specification
    target_spec, bw = selected_targets(X,
                                       Y,
                                       tau=tau,
                                       solution=conv.observed_soln)

    # nonzero set of penalized estimator
    nonzero_set = np.nonzero(conv.observed_soln)[0]
    num_set.append(nonzero_set.size)
    print(conv.observed_soln[nonzero_set])
    print(target_spec.observed_target)

    # beta target
    value = fsolve(find_root, 0, args=(tau, bw, 'Gaussian'))
    beta_target1 = np.linalg.pinv(X[:, nonzero_set]).dot(X.dot(beta) - value)
    beta_target2 = np.linalg.pinv(X[:, nonzero_set]).dot(X.dot(beta))
    print(beta_target1)
    print(beta_target2)

    # exact pivot
    exact_grid_inf = exact_grid_inference(query_spec, target_spec)
    lci, uci = exact_grid_inf._intervals(level=0.90)
    length = uci - lci
    coverage1 = (lci < beta_target1) * (uci > beta_target1)
    coverage2 = (lci < beta_target2) * (uci > beta_target2)
    coverage_exact1.append(np.mean(coverage1))
    coverage_exact2.append(np.mean(coverage2))
    length_exact.append(np.mean(length))

    # approximate pivot
    approximate_grid_inf = approximate_grid_inference(query_spec, target_spec)
    lci, uci = approximate_grid_inf._intervals(level=0.90)
    length = uci - lci
    coverage1 = (lci < beta_target1) * (uci > beta_target1)
    coverage2 = (lci < beta_target2) * (uci > beta_target2)
    coverage_approx1.append(np.mean(coverage1))
    coverage_approx2.append(np.mean(coverage2))
    length_approx.append(np.mean(length))

print('result summary')
print(num_set)
print(np.mean(coverage_exact1))
print(np.mean(coverage_approx1))
print(np.mean(coverage_exact2))
print(np.mean(coverage_approx2))
print(np.mean(length_exact))
print(np.mean(length_approx))


# Linear heterogeneous model
# generate sample
# n, p = 300, 2000
# mu, Sig = np.zeros(p), cov_generate(np.ones(p), 0.5)
# beta = np.zeros(p)
# beta[0:4] = 1
# tau = 0.3

# X = rgt.multivariate_normal(mean=mu, cov=Sig, size=n)
# X[:,5] = norm.cdf(X[:,0])
# Y = X.dot(beta) + 0.5 * X[:,5] * (rgt.normal(loc=0, scale=1, size=n) - norm.ppf(tau, loc=0, scale=1))