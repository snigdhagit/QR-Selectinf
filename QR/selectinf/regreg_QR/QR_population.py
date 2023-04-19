import numpy as np
from scipy.stats import norm
from scipy import integrate

def kernel_pdf(x, kernel='Laplacian'):
    # pdf of kernel
    if kernel == 'Laplacian':
        return 0.5 * np.exp(-abs(x))
    elif kernel == 'Gaussian':
        return norm.pdf(x)
    elif kernel == 'Logistic':
        return np.exp(-x) / (1 + np.exp(-x)) ** 2
    elif kernel == 'Uniform':
        return np.where(abs(x) <= 1, 0.5, 0)
    elif kernel == 'Epanechnikov':
        return np.where(abs(x) <= 1, 0.75 * (1 - x ** 2), 0)

def find_root(x, tau, bw, kernel):
    def f(u):
        return kernel_pdf(-x / bw + norm.ppf(tau) / bw + u, kernel) * norm.cdf(- bw * u)
    integral = integrate.quad(f, -np.inf, np.inf)
    return integral[0] - tau

def find_root_scale(x, scale, tau, bw, kernel):
    def f(u):
        return kernel_pdf(-x / bw + norm.ppf(tau) * scale / bw +  u * scale, kernel) * \
            norm.cdf(- bw * u)
    integral = integrate.quad(f, -np.inf, np.inf)
    return integral[0] - tau
