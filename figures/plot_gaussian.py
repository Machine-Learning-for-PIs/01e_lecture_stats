import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import tikzplotlib as tikz


def gaussian_pdf(x, mu=0., sigma=1.):
    return 1./(sigma * np.sqrt(2*np.pi))*np.exp(-1/2 * ((x - mu) / sigma)**2)

def forward_euler(x, fun, X_0 = 0.):
    """https://en.wikipedia.org/wiki/Euler_method  """
    X = [X_0]
    dx = (max(x)-min(x))/len(x)
    for pos_x in x:
        X.append(X[-1] + dx*fun(pos_x))
    return np.stack(X[1:])


params = ((0., np.sqrt(0.2)),
          (0., np.sqrt(1.0)),
          (0., np.sqrt(5.)),
          (-2.,np.sqrt(0.5)))
for param in params:
    x = np.linspace(-5., 5., 500)
    plt.plot(x, gaussian_pdf(x, param[0], param[1]),
                label=f"$\mu = {param[0]}, \sigma^2 = {param[1]**2:2.1f}$")
    plt.title("Probability Density Function")
    plt.xlabel("x")
    plt.ylabel("pdf(x)")
plt.legend()
plt.show()


for param in params:
    partial_gaussian_pdf = partial(gaussian_pdf, mu=param[0], sigma=param[1])
    X = forward_euler(x, partial_gaussian_pdf)
    plt.plot(x, X,
        label=f"$\mu = {param[0]}, \sigma^2 = {param[1]**2:2.1f}$")
    plt.title("Cumulative distribution function")
    plt.xlabel("x")
    plt.ylabel("cdf(x)")
plt.legend()
plt.show()



for param in params:
    x = np.linspace(-5., 5., 500)
    plt.plot(x, np.log(gaussian_pdf(x, param[0], param[1])),
                label=f"$\mu = {param[0]}, \sigma^2 = {param[1]**2:2.1f}$")
    plt.title("Log Probability Density Function")
    plt.xlabel("x")
    plt.ylabel("pdf(x)")
plt.legend()
plt.show()
