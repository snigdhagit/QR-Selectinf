from __future__ import print_function
import numpy as np
from .query import gaussian_query
from .regreg_QR.QR_high_dim import high_dim
from .regreg_QR.QR_low_dim import low_dim

class QR_lasso(gaussian_query):

    def __init__(self,
                 X,
                 Y,
                 tau,
                 randomizer,
                 kernel="Gaussian",
                 Lambda=None,
                 perturb=None):
        r"""
        Create a post-selection object for smooth quantile regression with L1 penatly

        Parameters
        ----------
        X : n by p matrix of covariates; each row is an observation vector.
        Y : an ndarray of response variables.
        tau : quantile level
        randomizer : object
            Randomizer -- contains representation of randomization density.
        perturb : np.ndarray
            Random perturbation subtracted as a linear
            term in the objective function.
        """
        self.X = X
        self.Y = Y
        self.n = X.shape[0]
        self.tau = tau
        self.randomizer = randomizer
        self.kernel = kernel
        self.Lambda = Lambda
        self._initial_omega = perturb  # random perturbation

    def fit(self,
            perturb=None,
            solve_args={}):
        """
        Fit the randomized lasso

        Parameters
        ----------
        solve_args : keyword args

        Returns
        -------
        signs : np.float
             Support and non-zero signs of randomized lasso solution.
        """

        n, p = self.X.shape
        # take a new perturbation if supplied
        if perturb is not None:
            self._initial_omega = perturb
        if self._initial_omega is None:
            self._initial_omega = self.randomizer.sample()

        # solving randomized problem
        _randomized_problem = high_dim(self.X,
                                       self.Y,
                                       self._initial_omega,
                                       intercept=False,
                                       solve_args=solve_args)
        _randomized_problem_fit = _randomized_problem.l1(tau=self.tau,
                                                         kernel=self.kernel,
                                                         Lambda=self.Lambda)
        self.observed_soln = _randomized_problem_fit['beta']
        self.observed_subgrad = _randomized_problem_fit['subgrad']

        # E for active
        # U for unpenalized
        # -E for inactive
        active_signs = np.sign(self.observed_soln)

        active = active_signs != 0
        unpenalized = _randomized_problem_fit['lambda'] == 0
        active *= ~unpenalized
        self._active = active
        self._unpenalized = unpenalized
        self._overall = (active + unpenalized) > 0
        self._inactive = ~self._overall

        _active_signs = active_signs.copy()
        _active_signs[unpenalized] = np.nan # don't release sign of unpenalized variables
        _ordered_variables = list((tuple(np.nonzero(active)[0]) +
                                  tuple(np.nonzero(unpenalized)[0])))
        self.selection_variable = {'sign': _active_signs, 'variables': _ordered_variables}

        # initial state for opt variables
        initial_scalings = np.fabs(self.observed_soln[active])
        initial_unpenalized = self.observed_soln[unpenalized]
        self.observed_opt_state = np.concatenate([initial_scalings, initial_unpenalized])
        self.num_opt_var = self.observed_opt_state.shape[0]

        # solving unpenalized problem (E \cup U)
        _unpenalized_problem = low_dim(self.X[:, self._overall],
                                       self.Y,
                                       intercept=False)
        _unpenalized_problem_fit = _unpenalized_problem.fit(tau=self.tau,
                                                            kernel=self.kernel,
                                                            beta0=self.observed_soln[self._overall])
        _unpenalized_beta = _unpenalized_problem_fit['beta']
        beta_bar = np.zeros(p)
        beta_bar[self._overall] = _unpenalized_beta

        # J, V matrix
        _V, _J, _grad = _randomized_problem.covariance(beta_bar,
                                                       tau=self.tau,
                                                       kernel=self.kernel).values()
        self.observed_score_state = - _J[:,self._overall].dot(_unpenalized_beta)
        self.observed_score_state[self._inactive] += _grad[self._inactive]

        # opt_linear matrix (contains signs)
        # E part
        opt_linear = np.zeros((p, self.num_opt_var))
        scaling_slice = slice(0, active.sum())
        if np.sum(active) == 0:
            _opt_hessian = 0
        else:
            _opt_hessian = _J[:,active] * active_signs[None, active]
        opt_linear[:, scaling_slice] = _opt_hessian
        # U part
        unpenalized_slice = slice(active.sum(), self.num_opt_var)
        if unpenalized.sum():
            opt_linear[:, unpenalized_slice] = _J[:,unpenalized]
        self.opt_linear = opt_linear

        # now make the constraints and implied gaussian
        self._setup = True
        A_scaling = -np.identity(self.num_opt_var)
        b_scaling = np.zeros(self.num_opt_var)

        # set the cov_score here without dispersion
        self._unscaled_cov_score = _V / self.n # V matrix
        self._setup_sampler_data = (A_scaling[:active.sum()],
                                    b_scaling[:active.sum()],
                                    self.opt_linear,
                                    self.observed_subgrad)
        return active_signs

    def setup_inference(self, dispersion=1):
        if self.num_opt_var > 0:
            self._setup_sampler(*self._setup_sampler_data, dispersion=dispersion)
