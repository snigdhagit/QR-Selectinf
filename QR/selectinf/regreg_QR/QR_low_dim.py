import numpy as np
import numpy.random as rgt
from scipy.stats import norm
from scipy.special import logsumexp


class low_dim():
    """
    Convolution Smoothed Quantile Regression
    """
    kernels = ["Laplacian", "Gaussian", "Logistic", "Uniform", "Epanechnikov"]
    opt = {'max_iter': 1e3, 'max_lr': 10, 'tol': 1e-5, 'nboot': 200}

    def __init__(self, X, Y, kernels=kernels, intercept=False, solve_args={}):
        """
        Parameters
        ----------
        X : n by p matrix of covariates; each row is an observation vector.
        Y : an ndarray of response variables.
        intercept : logical flag for adding an intercept to the model; default is TRUE.
        solve_args : a dictionary of internal statistical and optimization parameters.
            max_iter : maximum numder of iterations in the GD-BB algorithm; default is 500.
            max_lr : maximum step size/learning rate. If max_lr == False, there will be no
                     contraint on the maximum step size.
            tol : the iteration will stop when max{|g_j|: j = 1, ..., p} <= tol
                  where g_j is the j-th component of the (smoothed) gradient; default is 1e-4.
            nboot : number of bootstrap samples for inference.
        """
        self.n, self.p = X.shape
        if X.shape[1] >= self.n: raise ValueError("covariate dimension exceeds sample size")

        self.Y = Y.reshape(self.n)
        self.itcp = intercept

        if intercept:
            self.X = np.c_[np.ones(self.n), X]
        else:
            self.X = X

        self.kernels = kernels
        self.opt.update(solve_args)

    # def bandwidth(self, tau):
    #     h0 = 5 * min((self.X.shape[1] + np.log(self.n)) / self.n, 0.5) ** 0.4
    #     return max(0.01, h0 * (tau - tau ** 2) ** 0.5)

    def bandwidth(self, tau):
        return ((self.p + np.log(self.n)) / self.n) ** 0.4

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
        if kernel=='Laplacian':
            Ker = lambda x : 0.5 + 0.5 * np.sign(x) * (1 - np.exp(-abs(x)))
        elif kernel=='Gaussian':
            Ker = lambda x : norm.cdf(x)
        elif kernel=='Logistic':
            Ker = lambda x : 1 / (1 + np.exp(-x))
        elif kernel=='Uniform':
            Ker = lambda x : np.where(x > 1, 1, 0) + np.where(abs(x) <= 1, 0.5 * (1 + x), 0)
        elif kernel=='Epanechnikov':
            Ker = lambda x : 0.25 * (2 + 3 * x / 5 ** 0.5
                             - (x / 5 ** 0.5)**3 ) * (abs(x) <= 5 ** 0.5) \
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

    def fit(self, tau=0.5, h=None, kernel="Laplacian", beta0=np.array([])):
        """
        Convolution Smoothed Quantile Regression

        Parameters
        ----------
        tau : quantile level between 0 and 1; default is 0.5.
        h : bandwidth/smoothing parameter; the default value is computed by self.bandwidth(tau).
        kernel : a character string representing one of the built-in smoothing kernels;
                 default is "Laplacian".
        beta0 : initial estimate; default is np.array([]).
        res : an ndarray of fitted residuals; default is np.array([]).

        Returns
        ----------
        'beta' : conquer estimate.
        'res' : an ndarray of fitted residuals.
        'niter' : number of iterations.
        'bw' : bandwidth.
        'lr_seq' : a sequence of learning rates determined by the BB method.
        'lval_seq' : a sequence of (smoothed check) loss values at the iterations.
        """
        if h == None: h = self.bandwidth(tau)
        if kernel not in self.kernels:
            raise ValueError("kernel must be either Laplacian, Gaussian, Logistic, Uniform or Epanechnikov")
        if len(beta0) == 0:
            beta0 = rgt.randn(self.X.shape[1]) / self.X.shape[1] ** 0.5
            res = self.Y - self.X.dot(beta0)
        elif len(beta0) == self.X.shape[1]:
            res = self.Y - self.X.dot(beta0)
        else:
            raise ValueError("dimension of beta0 must match parameter dimension")
        #SQR Algorithm 1 
        lr_seq, lval_seq = [], []
        grad0 = self.X.T.dot(self.conquer_weight(-res / h, tau, kernel)) / self.n
        diff_beta = -grad0
        beta = beta0 + diff_beta
        res, t = self.Y - self.X.dot(beta), 0
        lval_seq.append(self.smooth_check(res, tau, h, kernel))

        while t < self.opt['max_iter'] and max(abs(diff_beta)) > self.opt['tol']:
            grad1 = self.X.T.dot(self.conquer_weight(-res / h, tau, kernel)) / self.n
            diff_grad = grad1 - grad0
            r0, r1 = diff_beta.dot(diff_beta), diff_grad.dot(diff_grad) 
            if r1 == 0:
                lr = 1
            else:
                r01 = diff_grad.dot(diff_beta)
                lr = min(logsumexp(abs(r01 / r1)), logsumexp(abs(r0 / r01)))

            if self.opt['max_lr']: lr = min(lr, self.opt['max_lr'])
            lr_seq.append(lr)
            grad0, diff_beta = grad1, -lr * grad1
            beta += diff_beta
            res = self.Y - self.X.dot(beta)
            lval_seq.append(self.smooth_check(res, tau, h, kernel))
            t += 1

        return {'beta': beta,
                'bw': h,
                'niter': t,
                'lval_seq': np.array(lval_seq),
                'lr_seq': np.array(lr_seq),
                'res': res}

    def covariance(self, beta, tau=0.5, h=None, kernel="Laplacian"):
        #SQR(2.11)
        if h == None: h = self.bandwidth(tau)
        res = self.Y - self.X.dot(beta)
        grad = self.X.T * (self.conquer_weight(-res / h, tau, kernel))
        hat_V = grad.dot(grad.T) / self.n
        hat_J = (self.X.T * self.kernel_pdf(res / h)).dot(self.X) / (self.n * h)
        return {'hat_V': hat_V,
                'hat_J': hat_J,
                'grad': self.X.T.dot(self.conquer_weight(-res/ h, tau, kernel)) / self.n}




