import numpy as np
import numpy.random as rgt
from scipy.stats import norm
import warnings

class high_dim():
    """
    Regularized Convolution Smoothed Quantile Regression via ILAMM (iterative local adaptive majorize-minimization)
    """

    kernels = ["Laplacian", "Gaussian", "Logistic", "Uniform", "Epanechnikov"]
    opt = {'phi': 0.1, 'gamma': 1.25, 'max_iter': 1e3, 'tol': 1e-8, 'iter_warning': True, 'nsim': 200}

    def __init__(self, X, Y, omega, kernels=kernels, intercept=False, solve_args={}):
        """
        Parameters
        ----------
        X : n by p matrix of covariates; each row is an observation vector.
        Y : an ndarray of response variables.
        omega: randomization term
        intercept : logical flag for adding an intercept to the model.
        solve_args : a dictionary of internal statistical and optimization parameters.
            phi : initial quadratic coefficient parameter in the ILAMM algorithm;
                  default is 0.1.
            gamma : adaptive search parameter that is larger than 1; default is 1.25.
            max_iter : maximum numder of iterations in the ILAMM algorithm; default is 1e3.
            tol : the ILAMM iteration terminates when |beta^{k+1} - beta^k|_max <= tol;
                  default is 1e-8.
            iter_warning : logical flag for warning when the maximum number
                           of iterations is achieved for the l1-penalized fit.
            nsim : number of simulations for computing a data-driven lambda; default is 200.
        """
        self.n, self.p = X.shape
        self.Y = Y.reshape(self.n)
        self.itcp = intercept

        if intercept:
            self.X = np.c_[np.ones(self.n), X]
            self.omega = np.append(0, omega)
        else:
            self.X = X
            self.omega = omega

        self.kernels = kernels
        self.opt.update(solve_args)

    # def bandwidth(self, tau):
    #     h0 = 5 * (np.log(self.p) / self.n) ** 0.25
    #     return max(0.01, h0 * (tau - tau ** 2) ** 0.5)

    def bandwidth(self, tau):
        return max(0.05, np.sqrt(tau * (1 - tau)) * (np.log(self.p) / self.n) ** 0.25)

    def soft_thresh(self, x, c):
        tmp = abs(x) - c
        return np.sign(x) * np.where(tmp <= 0, 0, tmp)

    def self_tuning(self, tau=0.5):
        """
        A Simulation-based Approach for Choosing the Penalty Level (Lambda)

        Reference
        ----------
        l1-Penalized quantile regression in high-dimensinoal sparse models (2011)
        by Alexandre Belloni and Victor Chernozhukov
        The Annals of Statistics 39(1): 82--130.

        Parameters
        ----------
        tau : quantile level; default is 0.5.

        Returns
        ----------
        lambda_sim : an ndarray of simulated lambda values.
        """
        lambda_sim = np.array([max(abs(self.X.T.dot(tau - (rgt.uniform(0, 1, self.n) <= tau))))
                               for b in range(self.opt['nsim'])])
        return lambda_sim / self.n

    def kernel_pdf(self, x, kernel='Laplacian'):
        # pdf of kernel
        if kernel=='Laplacian':
            K = lambda x : 0.5 * np.exp(-abs(x))
        elif kernel=='Gaussian':
            K = lambda x : norm.pdf(x)
        elif kernel=='Logistic':
            K = lambda x : np.exp(-x) / (1 + np.exp(-x)) ** 2
        elif kernel=='Uniform':
            K = lambda x : np.where(abs(x) <= 1, 0.5, 0)
        elif kernel=='Epanechnikov':
            K = lambda x : np.where(abs(x) <= 1, 0.75 * (1 - x ** 2), 0)
        return K(x)

    def conquer_weight(self, x, tau, kernel="Laplacian"):
        # cdf of kernel
        if kernel == 'Laplacian':
            Ker = lambda x: 0.5 + 0.5 * np.sign(x) * (1 - np.exp(-abs(x)))
        elif kernel == 'Gaussian':
            Ker = lambda x: norm.cdf(x)
        elif kernel == 'Logistic':
            Ker = lambda x: 1 / (1 + np.exp(-x))
        elif kernel == 'Uniform':
            Ker = lambda x: np.where(x > 1, 1, 0) + np.where(abs(x) <= 1, 0.5 * (1 + x), 0)
        elif kernel == 'Epanechnikov':
            Ker = lambda x: 0.25 * (2 + 3 * x / 5 ** 0.5 - (x / 5 ** 0.5) ** 3) * (abs(x) <= 5 ** 0.5) \
                            + (x > 5 ** 0.5)
        return (Ker(x) - tau)

    def smooth_check(self, x, tau=0.5, h=None, kernel='Laplacian'):
        # loss function
        if h == None: h = self.bandwidth(tau)
        if kernel == 'Laplacian':
            loss = lambda x: np.where(x >= 0, tau * x, (tau - 1) * x) + 0.5 * h * np.exp(-abs(x) / h)
        elif kernel == 'Gaussian':
            loss = lambda x: (tau - norm.cdf(-x / h)) * x \
                             + 0.5 * h * np.sqrt(2 / np.pi) * np.exp(-(x / h) ** 2 / 2)
        elif kernel == 'Logistic':
            loss = lambda x: tau * x + h * np.log(1 + np.exp(-x / h))
        elif kernel == 'Uniform':
            loss = lambda x: (tau - 0.5) * x + h * (0.25 * (x / h) ** 2 + 0.25) * (abs(x) < h) \
                             + 0.5 * abs(x) * (abs(x) >= h)
        elif kernel == 'Epanechnikov':
            loss = lambda x: (tau - 0.5) * x + 0.5 * h * (0.75 * (x / h) ** 2
                                                          - (x / h) ** 4 / 8 + 3 / 8) * (abs(x) < h) \
                             + 0.5 * abs(x) * (abs(x) >= h)
        return np.mean(loss(x))

    def l1(self, tau=0.5, Lambda=None, h=None, kernel="Laplacian", beta0=np.array([])):
        """
        L1-Penalized Convolution Smoothed Quantile Regression (l1-conquer)

        Parameters
        ----------
        tau : quantile level; default is 0.5.
        Lambda : regularization parameter. This should be either a scalar, or
                 a vector of length equal to the column dimension of X. If unspecified,
                 it will be computed by self.self_tuning().
        h : bandwidth/smoothing parameter; the default value is computed by self.bandwidth().
        kernel : a character string representing one of the built-in smoothing kernels;
                 default is "Laplacian".
        beta0 : initial estimate. If unspecified, it will be set as a vector of zeros.

        Returns
        ----------
        'beta' : an ndarray of estimated coefficients.
        'res' : an ndarray of fitted residuals.
        'niter' : number of iterations.
        'lambda' : lambda value.
        """

        if not Lambda:
            Lambda = np.sqrt(np.log(self.p) / self.n) * np.ones(self.p)
        elif np.size(Lambda) == 1:
            Lambda = Lambda * np.ones(self.p)

        if h == None: h = self.bandwidth(tau)
        if kernel not in self.kernels:
            raise ValueError("kernel must be either Laplacian, Gaussian, Logistic, Uniform or Epanechnikov")

        if len(beta0) == 0:
            beta0 = np.zeros(self.X.shape[1])
            if self.itcp: beta0[0] = np.quantile(self.Y, tau)
            res = self.Y - beta0[0]
        elif len(beta0) == self.X.shape[1]:
            res = self.Y - self.X.dot(beta0)
        else:
            raise ValueError("dimension of beta0 must match parameter dimension")

        phi, r0, t = self.opt['phi'], 1, 0
        while r0 > self.opt['tol'] and t < self.opt['max_iter']:

            grad0 = self.X.T.dot(self.conquer_weight(-res / h, tau, kernel) / self.n) - self.omega
            loss_eval0 = self.smooth_check(res, tau, h, kernel) - self.omega.dot(beta0)
            beta1 = beta0 - grad0 / phi
            beta1[self.itcp:] = self.soft_thresh(beta1[self.itcp:], Lambda / phi)
            diff_beta = beta1 - beta0
            r0 = diff_beta.dot(diff_beta)
            res = self.Y - self.X.dot(beta1)
            loss_proxy = loss_eval0 + diff_beta.dot(grad0) + 0.5 * phi * r0
            loss_eval1 = self.smooth_check(res, tau, h, kernel) - self.omega.dot(beta1)

            while loss_proxy < loss_eval1:
                phi *= self.opt['gamma']
                beta1 = beta0 - grad0 / phi
                beta1[self.itcp:] = self.soft_thresh(beta1[self.itcp:], Lambda / phi)
                diff_beta = beta1 - beta0
                r0 = diff_beta.dot(diff_beta)
                res = self.Y - self.X.dot(beta1)
                loss_proxy = loss_eval0 + diff_beta.dot(grad0) + 0.5 * phi * r0
                loss_eval1 = self.smooth_check(res, tau, h, kernel) - self.omega.dot(beta1)

            beta0, phi = beta1, self.opt['phi']
            t += 1

        if t == self.opt['max_iter'] and self.opt['iter_warning']:
            warnings.warn("Maximum number of iterations achieved with Lambda={} and tau={}".format(Lambda[0], tau))

        subgrad = self.omega - self.X.T.dot(self.conquer_weight(-res / h, tau, kernel)) / self.n

        return {'beta': beta1,
                'subgrad': subgrad,
                'res': res,
                'niter': t,
                'lambda': Lambda,
                'bw': h}

    def covariance(self, beta, tau=0.5, h=None, kernel="Laplacian"):
        if h == None: h = self.bandwidth(tau)
        res = self.Y - self.X.dot(beta)
        grad = self.X.T * (self.conquer_weight(-res/ h, tau, kernel))
        hat_V = grad.dot(grad.T) / self.n
        hat_J = (self.X.T * self.kernel_pdf(res / h)).dot(self.X) / (self.n * h)
        return {'hat_V': hat_V,
                'hat_J': hat_J,
                'grad': self.X.T.dot(self.conquer_weight(-res/ h, tau, kernel)) / self.n}



