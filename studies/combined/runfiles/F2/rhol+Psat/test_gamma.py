import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import distributions

trace = np.load('/media/owenmadin/storage/rjmc_output/local/output/C2H2/rhol+Psat/2020-09-29/intermediate/mcmc_prior/AUA+Q/trace.npy')

quadrupole_trace = trace[:, 4]

loc = np.mean(quadrupole_trace)
shape = 0.2/np.std(quadrupole_trace)


counts, bins = np.histogram(quadrupole_trace)
for i in range(len(bins)):
    if bins[i] < loc < bins[i+1]:
        argloc = i

xvec = np.linspace(0, 2*max(quadrupole_trace),num=500)
if counts[0] > counts[argloc]:
    print('exponential')

else:
    print('gamma')

expon = distributions.expon(scale=loc)
yexpon = expon.pdf(xvec)
plt.plot(xvec, yexpon, label='exponential',color='k')
gamma = distributions.gamma(shape, scale=loc / shape)
ygamma = gamma.pdf(xvec)
plt.plot(xvec, ygamma, label='gamma',color='k',ls='--')
plt.hist(quadrupole_trace,label='trace',bins=50,density=True,color='b',alpha=0.3)
plt.legend()
plt.show()
