"""Contains a class that can used to generate random numbers from an
arbitrary (discrete) distribution, mostly taken from Neil Crighton's
old astro package.
"""

import matplotlib.pyplot as pl
import numpy as np
from numpy import log10

npversion = map(int, np.__version__.split('.'))

class RanDist(object):
    """ Take random samples from an arbitrary discrete distribution."""

    def __init__(self, x, dist):
        """
        Inputs:
        x = values sampling the probability density
        dist = probability density distribution at each x value
        This finds the normalised probability density function
        (actually the mass density function because it's discrete) and
        the cumulative distribution function.
        Make sure the probability distribution is sampled densely
        enough (i.e. there are enough x values to properly define the
        shape of both the cdf and its inverse), because linear
        interpolation is used between the provided x values and cdf to
        infer new x values when generating the random numbers. A log
        sampling of x values is appropriate for distributions like
        inverse power laws, for example.
        """


        # normalise such that area under pdf is 1.
        self.pdf = dist / np.trapz(dist, x=x)
        # cumulative probability distribution
        self.cdf = dist.cumsum()
        self.cdf = self.cdf / float(self.cdf[-1])
        self.x = x

    def random(self, N=1, seed=None):
        """Return N random numbers with the requested distribution."""
        if seed is not None:  np.random.seed(seed)
        i = np.random.rand(N)
        y = np.interp(i, self.cdf, self.x)
        return y

    def plot_pdf(self):
        pl.plot(self.x, self.pdf)

    def self_test(self, N=int(1e4), log=False, seed=None, nbins=50):
        """ Make plots of the CDF, the PDF, and a histogram of N
        random samples from the distribution.
        """
        pl.figure()
        pl.subplots_adjust(hspace=0.001)
        # The cdf
        ax = pl.subplot(211)
        if log:
            ax.semilogx(self.x, self.cdf, 'b-')
        else:
            ax.plot(self.x, self.cdf, 'b-')
        ax.set_ylabel('cdf')
        ax.set_ylim(0,1)
        #ax.set_xticklabels([])

        # The actual generated numbers
        ax = pl.subplot(212, sharex=ax)
        y = self.random(N, seed=seed)
        if log:
            bins = np.logspace(log10(min(y)), log10(max(y)), nbins)
        else:
            bins = nbins

        if list(npversion) < [1, 3, 0]:
            vals, edges = np.histogram(y, bins=bins, normed=True)
        else:
            vals, edges = np.histogram(y, normed=True, bins=bins)
        if log:
            ax.loglog(self.x, np.where(self.pdf > 0, self.pdf, 1e-20), 'r-',
                      label='pdf')
            ax.loglog(edges[:-1], np.where(vals > 0, vals, 1e-20),
                      ls='steps-post', label='random values')
        else:
            ax.plot(self.x, self.pdf, 'r-', label='pdf')
            ax.plot(edges[:-1], vals, ls='steps-post', label='random values')
        ax.legend(frameon=False)
        ax.set_ylabel('pdf')

# test distributions
def test_dist():
    seed = 101
    # check a column density power law distribution
    beta = 1.5
    def ndist(n):
        return n**-beta

    nvals = np.logspace(12.6, 16, 1000)
    ran = RanDist(nvals, ndist(nvals))
    ran.self_test(log=1, seed=seed)

    bsig = 24.0
    # b param distribution
    def bdist(b):
        b1 = bsig / b
        return  b1**5 * np.exp(-b1**4)

    bvals = np.linspace(10, 150, 1000)
    ran = RanDist(bvals, bdist(bvals))
    ran.self_test(seed=seed)

    gamma = 2.04
    # z distribution
    def zdist(z):
        return  (1 + z)**gamma

    zp1vals = np.logspace(log10(1+2.5), log10(1+4), 1000)
    ran = RanDist(zp1vals, zdist(zp1vals))
    ran.self_test(log=1, seed=seed, N=int(1e5), nbins=20)
    pl.show()

if __name__ == '__main__':
    test_dist()
