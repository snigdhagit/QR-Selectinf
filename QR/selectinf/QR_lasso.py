from __future__ import print_function
import numpy as np
from .query import gaussian_query
from .regreg_QR.QR_high_dim_penalties import QR_high_dim
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
        self.randomizer = randomizer #for omega
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
        _randomized_problem = QR_high_dim(self.X,
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
        # U for unpenalized #set to not shrink?
        # -E for inactive
        active_signs = np.sign(self.observed_soln)

        active = active_signs != 0 
        unpenalized = _randomized_problem_fit['lambda'] == 0
        active *= ~unpenalized #~ flip boolean value
        self._active = active #indicate the active variables
        self._unpenalized = unpenalized #indicate the unpenalized variables
        self._overall = (active + unpenalized) > 0 #active or unpenalized
        self._inactive = ~self._overall

        _active_signs = active_signs.copy()
        _active_signs[unpenalized] = np.nan # don't release sign of unpenalized variables
        _ordered_variables = list((tuple(np.nonzero(active)[0]) +
                                  tuple(np.nonzero(unpenalized)[0])))
        self.selection_variable = {'sign': _active_signs, 'variables': _ordered_variables}

        # initial state for opt variables
        initial_scalings = np.fabs(self.observed_soln[active]) #absolute values of active variables
        initial_unpenalized = self.observed_soln[unpenalized]
        self.observed_opt_state = np.concatenate([initial_scalings, initial_unpenalized])
        self.num_opt_var = self.observed_opt_state.shape[0] #q

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
        scaling_slice = slice(0, active.sum()) #select all active variables
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

        print(np.linalg.matrix_rank(_J[:, active]))

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


class QR_scad(gaussian_query):

    def __init__(self,
                 X,
                 Y,
                 tau,
                 randomizer,
                 kernel="Gaussian",
                 Lambda=None,
                 gamma = 3.7,
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
        self.randomizer = randomizer #for omega
        self.kernel = kernel
        self.Lambda = Lambda
        self.gamma = gamma
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
        _randomized_problem = QR_high_dim(self.X,
                                       self.Y,
                                       self._initial_omega,
                                       intercept=False,
                                       solve_args=solve_args)
        _randomized_problem_fit = _randomized_problem.irw(tau=self.tau,
                                                         kernel=self.kernel,
                                                         Lambda=self.Lambda,
                                                         gamma = self.gamma,
                                                         penalty="SCAD")
        self.observed_soln = _randomized_problem_fit['beta']
        self.observed_subgrad = _randomized_problem_fit['observed_subgrad']

        # E for active
        # U for unpenalized #set to not shrink?
        # -E for inactive
        active_signs = np.sign(self.observed_soln)

        active = active_signs != 0
        unpenalized = _randomized_problem_fit['lambda'] == 0
        num_unpenalized = (unpenalized.sum()).astype(int)
        active *= ~unpenalized #~ flip boolean value
        self._active = active #indicate the active variables
        self._unpenalized = unpenalized #indicate the unpenalized variables
        self._overall = (active + unpenalized) > 0  #active or unpenalized
        self._inactive = ~self._overall
        active_sol = self.observed_soln[self._active]

        I1 = np.diag(
            np.logical_and(np.abs(active_sol) > 0, np.abs(active_sol) <= self.Lambda).astype(int))
        I2 = np.diag(np.logical_and(np.abs(active_sol) > self.Lambda,
                                    np.abs(active_sol) <= self.gamma * self.Lambda).astype(int))
        I3 = np.diag(np.abs(active_sol) > self.gamma * self.Lambda).astype(int)

        _active_signs = active_signs.copy()
        _active_signs[unpenalized] = np.nan # don't release sign of unpenalized variables
        _ordered_variables = list((tuple(np.nonzero(active)[0]) +
                                  tuple(np.nonzero(unpenalized)[0])))
        self.selection_variable = {'sign': _active_signs, 'variables': _ordered_variables}

        # initial state for opt variables
        initial_scalings = np.fabs(self.observed_soln[active]) #absolute values of active variables
        initial_unpenalized = self.observed_soln[unpenalized]
        self.observed_opt_state = np.concatenate([initial_scalings, initial_unpenalized]) #need a extend version?
        self.num_opt_var = self.observed_opt_state.shape[0] #need to be more than q

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
        scaling_slice = slice(0, active.sum()) #select all active variables
        if np.sum(active) == 0:
            _opt_hessian = 0
        else:
            I_mat = np.zeros(np.shape(_J[:,active]))
            I_mat[active,:] = I2
            _opt_hessian = (_J[:,active] - I_mat/((self.gamma-1))) * active_signs[None, active]
        print(np.linalg.matrix_rank(_J[:,active]))
        #print(np.linalg.matrix_rank(_opt_hessian))
        opt_linear[:, scaling_slice] = _opt_hessian
        # U part
        unpenalized_slice = slice(active.sum(), self.num_opt_var)
        if unpenalized.sum():
            opt_linear[:, unpenalized_slice] = _J[:,unpenalized]
        self.opt_linear = opt_linear

        # now make the constraints and implied gaussian
        self._setup = True
        E1 = np.sum(I1)
        E2 = np.sum(I2)
        E3 = np.sum(I3)
        num_active = E1 + E2 + E3

        def get_scaling_mat(A,u):
            q = np.shape(A)[0]
            zero_mat = np.zeros((q,u))
            return np.vstack(((np.hstack((A, zero_mat))), (np.hstack((zero_mat.T, np.zeros((u,u)))))))

        b1 = self.Lambda * np.ones(self.num_opt_var)
        b2 = self.gamma * self.Lambda * np.ones(self.num_opt_var)

        #Constraint: sign_{|O_{E1 & E2}|} >=0 >>>>  -I_{E1 & E2} O_{E1 & E2} <= 0
        A_scaling_1 = - get_scaling_mat(I1+I2 , num_unpenalized)#-I_{E1&E2} O <= 0
        b_scaling_1 = np.zeros(self.num_opt_var)

        #Constraint: O_{E1} <= lambda & O_{E2} <=gamma*lambda & O_{E3} > gamma * lambda
        mat_1 = get_scaling_mat(I1, num_unpenalized)
        mat_2 = get_scaling_mat(I2, num_unpenalized)
        mat_3 = get_scaling_mat(I3, num_unpenalized)
        A_scaling_2 = mat_1 + mat_2 - mat_3 #O_{E1} <= lambda & O_{E2} <=gamma*lambda & O_{E3} > gamma * lambda
        b_scaling_2 = mat_1 @ b1 + mat_2 @ b2 - mat_3 @ b2

        #Constraint: O_{E2} > lambda
        A_scaling_3 = - mat_2
        b_scaling_3 = -mat_2 @ b1

        A_scaling = np.vstack((A_scaling_1, A_scaling_2, A_scaling_3))
        b_scaling = np.concatenate((b_scaling_1, b_scaling_2, b_scaling_3))
        mask = np.any(A_scaling != 0, axis = 1)
        A_scaling = A_scaling[mask]
        b_scaling = b_scaling[mask]


        # set the cov_score here without dispersion
        self._unscaled_cov_score = _V / self.n # V matrix
        self._setup_sampler_data = (A_scaling,
                                    b_scaling,
                                    self.opt_linear,
                                    self.observed_subgrad)
        return active_signs




    def setup_inference(self, dispersion=1):
        if self.num_opt_var > 0:
            self._setup_sampler(*self._setup_sampler_data, dispersion=dispersion)




class QR_mcp(gaussian_query):

    def __init__(self,
                 X,
                 Y,
                 tau,
                 randomizer,
                 kernel="Gaussian",
                 Lambda=None,
                 gamma = 3.7,
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
        self.randomizer = randomizer #for omega
        self.kernel = kernel
        self.Lambda = Lambda
        self.gamma = gamma
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
        _randomized_problem = QR_high_dim(self.X,
                                       self.Y,
                                       self._initial_omega,
                                       intercept=False,
                                       solve_args=solve_args)
        _randomized_problem_fit = _randomized_problem.irw(tau=self.tau,
                                                         kernel=self.kernel,
                                                         Lambda=self.Lambda,
                                                         gamma = self.gamma,
                                                         penalty="MCP")
        self.observed_soln = _randomized_problem_fit['beta']
        self.observed_subgrad = _randomized_problem_fit['observed_subgrad']

        # E for active
        # U for unpenalized #set to not shrink?
        # -E for inactive
        active_signs = np.sign(self.observed_soln)

        active = active_signs != 0
        unpenalized = _randomized_problem_fit['lambda'] == 0
        num_unpenalized = (unpenalized.sum()).astype(int)
        active *= ~unpenalized #~ flip boolean value
        self._active = active #indicate the active variables
        self._unpenalized = unpenalized #indicate the unpenalized variables
        self._overall = (active + unpenalized) > 0  #active or unpenalized
        self._inactive = ~self._overall
        active_sol = self.observed_soln[self._active]

        I = np.diag(
            np.logical_and(np.abs(active_sol) > 0, np.abs(active_sol) <= self.gamma * self.Lambda).astype(int))
        I2 = np.identity(active.sum()) - I


        _active_signs = active_signs.copy()
        _active_signs[unpenalized] = np.nan # don't release sign of unpenalized variables
        _ordered_variables = list((tuple(np.nonzero(active)[0]) +
                                  tuple(np.nonzero(unpenalized)[0])))
        self.selection_variable = {'sign': _active_signs, 'variables': _ordered_variables}

        # initial state for opt variables
        initial_scalings = np.fabs(self.observed_soln[active]) #absolute values of active variables
        initial_unpenalized = self.observed_soln[unpenalized]
        self.observed_opt_state = np.concatenate([initial_scalings, initial_unpenalized]) #need a extend version?
        self.num_opt_var = self.observed_opt_state.shape[0] #need to be more than q

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
        scaling_slice = slice(0, active.sum()) #select all active variables
        if np.sum(active) == 0:
            _opt_hessian = 0
        else:
            I_mat = np.zeros(np.shape(_J[:,active]))
            I_mat[active,:] = I
            _opt_hessian = (_J[:,active] - I_mat/(self.gamma)) * active_signs[None, active]
        opt_linear[:, scaling_slice] = _opt_hessian

        print(np.linalg.matrix_rank(_J[:, active]))
        # U part
        unpenalized_slice = slice(active.sum(), self.num_opt_var)
        if unpenalized.sum():
            opt_linear[:, unpenalized_slice] = _J[:,unpenalized]
        self.opt_linear = opt_linear

        # now make the constraints and implied gaussian
        self._setup = True

        def get_scaling_mat(A,u):
            q = np.shape(A)[0]
            zero_mat = np.zeros((q,u))
            return np.vstack(((np.hstack((A, zero_mat))), (np.hstack((zero_mat.T, np.zeros((u,u)))))))

        b = self.gamma * self.Lambda * np.ones(self.num_opt_var)

        #Constraint: sign_{|O_{E1}|} >=0 >>>>  -I_{E1} O <= 0
        A_scaling_1 = - get_scaling_mat(I , num_unpenalized)
        b_scaling_1 = np.zeros(self.num_opt_var)

        #Constraint: O_{E1} <= gamma * lambda, O_{-E1} > gamma * lambda
        mat_1 = get_scaling_mat(I, num_unpenalized)
        mat_2 = get_scaling_mat(I2, num_unpenalized)
        A_scaling_2 = mat_1 - mat_2 #O_{E1} <= gamma * lambda & O_{E2} > gamma*lambda
        b_scaling_2 = mat_1 @ b - mat_2 @ b


        A_scaling = np.vstack((A_scaling_1, A_scaling_2))
        b_scaling = np.concatenate((b_scaling_1, b_scaling_2))
        mask = np.any(A_scaling != 0, axis = 1)
        A_scaling = A_scaling[mask]
        b_scaling = b_scaling[mask]

        # set the cov_score here without dispersion
        self._unscaled_cov_score = _V / self.n # V matrix
        self._setup_sampler_data = (A_scaling,
                                    b_scaling,
                                    self.opt_linear,
                                    self.observed_subgrad)
        return active_signs




    def setup_inference(self, dispersion=1):
        if self.num_opt_var > 0:
            self._setup_sampler(*self._setup_sampler_data, dispersion=dispersion)
