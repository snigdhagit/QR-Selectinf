from typing import NamedTuple

import numpy as np
from .regreg_QR.QR_low_dim import low_dim

# functions construct targets of inference
# and covariance with score representation

class TargetSpec(NamedTuple):
    observed_target: np.ndarray
    cov_target: np.ndarray
    regress_target_score: np.ndarray
    alternatives: list

def selected_targets(X,
                     Y,
                     tau,
                     solution,
                     kernel="Gaussian",
                     features=None,
                     sign_info={},
                     dispersion=1,
                     solve_args={}):
    if features is None:
        features = solution != 0
    n, p = X.shape

    # solving restricted problem
    _unpenalized_problem = low_dim(X[:, features],
                                   Y,
                                   intercept=False,
                                   solve_args=solve_args)
    _unpenalized_problem_fit = _unpenalized_problem.fit(tau=tau,
                                                        kernel=kernel,
                                                        beta0=solution[features])
    observed_target = _unpenalized_problem_fit['beta']
    V_feat, J_feat = _unpenalized_problem.covariance(observed_target,
                                                     tau=tau,
                                                     kernel=kernel).values()
    bw = _unpenalized_problem_fit['bw']

    # covariance
    cov_target = np.linalg.inv(J_feat).dot(V_feat.dot(np.linalg.inv(J_feat))) / n
    regress_target_score = np.zeros((cov_target.shape[0], p))
    regress_target_score[:, features] = np.linalg.inv(J_feat)

    alternatives = ['twosided'] * features.sum()
    features_idx = np.arange(p)[features]
    for i in range(len(alternatives)):
        if features_idx[i] in sign_info.keys():
            alternatives[i] = sign_info[features_idx[i]]

    return TargetSpec(observed_target,
                      cov_target * dispersion,
                      regress_target_score,
                      alternatives), bw

def target_query_Interactspec(query_spec,
                              regress_target_score,
                              cov_target):
    QS = query_spec
    prec_target = np.linalg.inv(cov_target)

    U1 = regress_target_score.T.dot(prec_target)
    U2 = U1.T.dot(QS.M2.dot(U1))
    U3 = U1.T.dot(QS.M3.dot(U1))
    U4 = QS.M1.dot(QS.opt_linear).dot(QS.cond_cov).dot(QS.opt_linear.T.dot(QS.M1.T.dot(U1)))
    U5 = U1.T.dot(QS.M1.dot(QS.opt_linear))

    return U1, U2, U3, U4, U5