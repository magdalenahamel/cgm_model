from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

from numpy import random
from scipy.spatial.distance import pdist, cdist
from scipy.stats import kstwobign, pearsonr
from scipy.stats import genextreme

import concurrent.futures
import itertools
import functools

__all__ = ['ks2d2s', 'estat', 'estat2d']

churchill_iso = pd.read_csv('Churchill_iso_full.txt', error_bad_lines=False, delim_whitespace=True)
W_r_churchill_iso = churchill_iso['Wr'].to_numpy()
D_R_vir_churchill_iso = churchill_iso['etav'].to_numpy()
e_Wr = churchill_iso['eWr'].to_numpy()

con_upper = (e_Wr == -1.)
W_r_churchill_upper = W_r_churchill_iso[con_upper]
D_R_vir_churchill_upper = D_R_vir_churchill_iso[con_upper]
e_Wr_upper = e_Wr[con_upper]

W_r_churchill_no_upper = W_r_churchill_iso[~con_upper]
D_R_vir_churchill_no_upper = D_R_vir_churchill_iso[~con_upper]
e_Wr_no_upper = e_Wr[~con_upper]

all_d = np.concatenate((D_R_vir_churchill_no_upper, D_R_vir_churchill_upper))

### 2D KS test ###

def ks2d2s(x1, y1, x2, y2, nboot=None, extra=False):
    '''Two-dimensional Kolmogorov-Smirnov test on two samples. 
    Parameters
    ----------
    x1, y1 : ndarray, shape (n1, )
        Data of sample 1.
    x2, y2 : ndarray, shape (n2, )
        Data of sample 2. Size of two samples can be different.
    extra: bool, optional
        If True, KS statistic is also returned. Default is False.
    Returns
    -------
    p : float
        Two-tailed p-value.
    D : float, optional
        KS statistic. Returned if keyword `extra` is True.
    Notes
    -----
    This is the two-sided K-S test. Small p-values means that the two samples are significantly different. Note that the p-value is only an approximation as the analytic distribution is unkonwn. The approximation is accurate enough when N > ~20 and p-value < ~0.20 or so. When p-value > 0.20, the value may not be accurate, but it certainly implies that the two samples are not significantly different. (cf. Press 2007)
    References
    ----------
    Peacock, J.A. 1983, Two-Dimensional Goodness-of-Fit Testing in Astronomy, Monthly Notices of the Royal Astronomical Society, vol. 202, pp. 615-627
    Fasano, G. and Franceschini, A. 1987, A Multidimensional Version of the Kolmogorov-Smirnov Test, Monthly Notices of the Royal Astronomical Society, vol. 225, pp. 155-170
    Press, W.H. et al. 2007, Numerical Recipes, section 14.8
    '''
    assert (len(x1) == len(y1)) and (len(x2) == len(y2))
    n1, n2 = len(x1), len(x2)
    D = avgmaxdist(x1, y1, x2, y2)

    if nboot is None:
        sqen = np.sqrt(n1 * n2 / (n1 + n2))
        r1 = pearsonr(x1, y1)[0]
        r2 = pearsonr(x2, y2)[0]
        r = np.sqrt(1 - 0.5 * (r1**2 + r2**2))
        d = D * sqen / (1 + r * (0.25 - 0.75 / sqen))
        p = kstwobign.sf(d)
    else:
        n = n1 + n2
        x = np.concatenate([x1, x2])
        y = np.concatenate([y1, y2])
        d = np.empty(nboot, 'f')
        for i in range(nboot):
            idx = random.choice(n, n, replace=True)
            ix1, ix2 = idx[:n1], idx[n1:]
            #ix1 = random.choice(n, n1, replace=True)
            #ix2 = random.choice(n, n2, replace=True)
            d[i] = avgmaxdist(x[ix1], y[ix1], x[ix2], y[ix2])
        p = np.sum(d > D).astype('f') / nboot
    if extra:
        return p, D
    else:
        return p


def avgmaxdist(x1, y1, x2, y2):
    D1 = maxdist(x1, y1, x2, y2)
    D2 = maxdist(x2, y2, x1, y1)
    return (D1 + D2) / 2


def maxdist(x1, y1, x2, y2):
    n1 = len(x1)
    D1 = np.empty((n1, 4))
    for i in range(n1):
        a1, b1, c1, d1 = quadct(x1[i], y1[i], x1, y1)
        a2, b2, c2, d2 = quadct(x1[i], y1[i], x2, y2)
        D1[i] = [a1 - a2, b1 - b2, c1 - c2, d1 - d2]

    # re-assign the point to maximize difference,
    # the discrepancy is significant for N < ~50
    D1[:, 0] -= 1 / n1

    dmin, dmax = -D1.min(), D1.max() + 1 / n1
    return max(dmin, dmax)


def quadct(x, y, xx, yy):
    n = len(xx)
    ix1, ix2 = xx <= x, yy <= y
    a = np.sum(ix1 & ix2) / n
    b = np.sum(ix1 & ~ix2) / n
    c = np.sum(~ix1 & ix2) / n
    d = 1 - a - b - c
    return a, b, c, d


def estat2d(x1, y1, x2, y2, **kwds):
    return estat(np.c_[x1, y1], np.c_[x2, y2], **kwds)


def estat(x, y, nboot=1000, replace=False, method='log', fitting=False):
    '''
    Energy distance statistics test.
    Reference
    ---------
    Aslan, B, Zech, G (2005) Statistical energy as a tool for binning-free
      multivariate goodness-of-fit tests, two-sample comparison and unfolding.
      Nuc Instr and Meth in Phys Res A 537: 626-636
    Szekely, G, Rizzo, M (2014) Energy statistics: A class of statistics
      based on distances. J Stat Planning & Infer 143: 1249-1272
    Brian Lau, multdist, https://github.com/brian-lau/multdist
    '''
    n, N = len(x), len(x) + len(y)
    stack = np.vstack([x, y])
    stack = (stack - stack.mean(0)) / stack.std(0)
    if replace:
        rand = lambda x: random.randint(x, size=x)
    else:
        rand = random.permutation

    en = energy(stack[:n], stack[n:], method)
    en_boot = np.zeros(nboot, 'f')
    for i in range(nboot):
        idx = rand(N)
        en_boot[i] = energy(stack[idx[:n]], stack[idx[n:]], method)

    if fitting:
        param = genextreme.fit(en_boot)
        p = genextreme.sf(en, *param)
        return p, en, param
    else:
        p = (en_boot >= en).sum() / nboot
        return p, en, en_boot


def energy(x, y, method='log'):
    dx, dy, dxy = pdist(x), pdist(y), cdist(x, y)
    n, m = len(x), len(y)
    if method == 'log':
        dx, dy, dxy = np.log(dx), np.log(dy), np.log(dxy)
    elif method == 'gaussian':
        raise NotImplementedError
    elif method == 'linear':
        pass
    else:
        raise ValueError
    z = dxy.sum() / (n * m) - dx.sum() / n**2 - dy.sum() / m**2
    # z = ((n*m)/(n+m)) * z # ref. SR
    return z

### bootstrap ###

'''bs = np.linspace(0.1,4,10) # characteristic radius of the exponential function (it is accually a porcentage of Rvir) in log scale to make the range more homogeneous in lin scale
csize = np.linspace(0.01,2,10) #poner en escala mas separada
hs = 5 #bajar un poco para que no sea un  1,10,20
hv = 10 #bajar maximo a 100

params = [bs,csize]
results_r_2 = np.load('mp_mcmc_10.npy')'''


bs_12 = np.linspace(0.1,4,7) # characteristic radius of the exponential function (it is accually a porcentage of Rvir) in log scale to make the range more homogeneous in lin scale
csize_12 = np.linspace(0.01,1,7) #poner en escala mas separada
hs_12 = np.linspace(1,20,7) #bajar un poco para que no sea un  1,10,20
hv_12 = np.linspace(0, 20,7) #bajar maximo a 100

params_12 = [bs_12,csize_12,hs_12,hv_12]

params_name_12 = ['f_v', 'cloud size', 'disc height', 'velocity scale height']

results_r_12 = np.load('mp_mcmc_12.npy')

'''bs_2 = np.linspace(0.1,5,7) 
csize_2 = np.linspace(0.01,2,7) 
hs_2 = np.linspace(5,40,7) 
hv_2 = np.linspace(0, 50,7) 

params_2 = [bs_2,csize_2,hs_2,hv_2]

params_name_2 = ['f_v', 'cloud size', 'disc height', 'velocity scale height']

results_r_2 = np.load('mcmc_10.npy')'''



'''bs_3 = np.linspace(0.1,5,7) # characteristic radius of the exponential function (it is accually a porcentage of Rvir) in log scale to make the range more homogeneous in lin scale
csize_3 = np.linspace(0.01,2,7) #poner en escala mas separada
hs_3 = np.linspace(5,40,7) #bajar un poco para que no sea un  1,10,20
hv_3 = np.linspace(0, 50,7) #bajar maximo a 100

params_3 = [bs_3,csize_3,hs_3,hv_3]

params_name_3 = ['f_v', 'cloud size', 'disc height', 'velocity scale height']

results_r_3 = np.load('mcmc_3.npy')


results_r_3.shape'''

def boot_sample(params):
    model_D_R_vir = params[0]
    model_Wr = params[1]
    no_upper_sample = random.normal(loc=W_r_churchill_no_upper, scale=e_Wr_no_upper, size=None)
    upper_sample = random.uniform(low=0.0, high=W_r_churchill_upper, size=None)
    all_sample = np.concatenate((no_upper_sample, upper_sample))
    p = ks2d2s(all_d,all_sample,model_D_R_vir, model_Wr)
    return(p)

def getpgrid_boot(modelgrid, boot = 1000):
        #Determine the grid in terms of deviation from sigma
        pgrid=np.zeros((7,7,7,7)) + 1.0
         #Loop through each constraint
        for i in range(7):
            for j in range(7):
                for k in range(7):
                    for l in range(7):
                        print(i,j,k,l)
                        model_Wr = modelgrid[0][i][j][k][l]
                        model_D = modelgrid[1][i][j][k][l]
                        model_R_vir = modelgrid[2][i][j][k][l]
                        model_D_R_vir= model_D/model_R_vir
                        ks = []
                        
                        params_par = [model_D_R_vir,model_Wr]
                        
                        with concurrent.futures.ProcessPoolExecutor() as executor:
                            results = [executor.submit(boot_sample, params_par) for _ in range(boot)]
                    
                            for r in concurrent.futures.as_completed(results):
                                ks.append(r.result())
                            
                            p_med = np.mean(ks)
                            pgrid[i][j][k][l] = p_med
                        
                        
                        
                        ''' for m in range(boot):
                            print('b', m)
                            no_upper_sample = random.normal(loc=W_r_churchill_no_upper, scale=e_Wr_no_upper, size=None)
                            upper_sample = random.uniform(low=0.0, high=W_r_churchill_upper, size=None)
                            all_sample = np.concatenate((no_upper_sample, upper_sample))
                            p = ks2d2s(all_d,all_sample,model_D_R_vir, model_Wr)
                            ks.append(p)'''
                            
                        
                        
    
        return pgrid
    
def getpgrid_boot_2(modelgrid, boot = 1000):
        #Determine the grid in terms of deviation from sigma
        pgrid=np.zeros((10,10)) + 1.0
        #Loop through each constraint
        for i in range(10):
            for j in range(10):
                print(i,j)
                model_Wr = modelgrid[0][i][j]
                model_D = modelgrid[1][i][j]
                model_R_vir = modelgrid[2][i][j]
                model_D_R_vir= model_D/model_R_vir
                ks = []
                params_par = [model_D_R_vir,model_Wr]
                        
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    results = [executor.submit(boot_sample, params_par) for _ in range(boot)]
                    
                    for r in concurrent.futures.as_completed(results):
                        ks.append(r.result())
                        
                    ''' for m in range(boot):
                            print('b', m)
                            no_upper_sample = random.normal(loc=W_r_churchill_no_upper, scale=e_Wr_no_upper, size=None)
                            upper_sample = random.uniform(low=0.0, high=W_r_churchill_upper, size=None)
                            all_sample = np.concatenate((no_upper_sample, upper_sample))
                            p = ks2d2s(all_d,all_sample,model_D_R_vir, model_Wr)
                            ks.append(p)'''
                            
                    p_med = np.mean(ks)
                    pgrid[i][j] = p_med
        return(pgrid)
                       
prob_2_boot = getpgrid_boot(results_r_12)
    
np.save('pgrid_boot_12', prob_2_boot)
