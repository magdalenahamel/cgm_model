from __future__ import division


from numpy import random
from scipy.spatial.distance import pdist, cdist
from scipy.stats import kstwobign, pearsonr
from scipy.stats import genextreme

import concurrent.futures
import functools

from tqdm.notebook import tqdm, trange
import time
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

import Sample
#import cgmspec.Sample as sample

import concurrent.futures
import itertools

from astropy import constants as const
from astropy.convolution import convolve, Gaussian1DKernel

import time

start = time.process_time()
# your code here    


### TPCF ###


def filtrogauss(R, spec_res, lam_0, flux):
    del_lam = lam_0/R
    del_lam_pix = del_lam/spec_res
    gauss_kernel = (Gaussian1DKernel(del_lam_pix))
    gausflux = convolve(flux, gauss_kernel)
    return(gausflux)

minor_tpcf = pd.read_csv('2minor.txt', delimiter='     ', engine='python')
major_tpcf = pd.read_csv('2major.txt', delimiter='     ', engine='python')

minor_vel = minor_tpcf['vel'].to_numpy()
minor_bins = minor_vel - 5
minor_bins = np.append(minor_bins, minor_bins[-1] +10)
minor_tpcf_val = minor_tpcf['TPCF'].to_numpy()
minor_error = np.abs(minor_tpcf['minus_error'].to_numpy() - minor_tpcf['plus_error'].to_numpy())
      
major_vel = major_tpcf['vel'].to_numpy()
major_bins = major_vel - 5
major_bins = np.append(major_bins, major_bins[-1] +10)
major_tpcf_val = major_tpcf['TPCF'].to_numpy()
major_error = np.abs(major_tpcf['minus_error'].to_numpy() - major_tpcf['plus_error'].to_numpy())

from itertools import combinations

def TPCF(params):
    speci_empty_t = params[0]
    pos_alpha = params[1]
    print('TPCF', pos_alpha)
    #cond = np.asarray(nr_clouds) == 0
    #if len(speci_empty_t) == 0:
        #return(np.zero(len(major_vel)))
        
    if len(speci_empty_t)==0 and pos_alpha =='minor':
            return(np.zeros(len(minor_vel)))
    elif len(speci_empty_t)==0 and pos_alpha =='major':
            return(np.zeros(len(major_vel)))
    gauss_specs = []
    abs_specs = []
    vels_abs = []
    #speci_empty_t = np.asarray(speci_empty)[~cond]
    print('how many specs', len(speci_empty_t))
    
    for m in range(len(speci_empty_t)):
        
        gauss_specj = filtrogauss(45000,0.03,2796.35,speci_empty_t[m])
        gauss_specs.append(gauss_specj)
        zabs=0.77086

        cond_abs1 = gauss_specj < 0.98
        cond_abs2 = np.abs(vels_wave) < 800
        abs_gauss_spec_major = vels_wave[cond_abs1 & cond_abs2]
        abs_specs.append(abs_gauss_spec_major)

    # Convert input list to a numpy array
    abs_specs_f = np.concatenate(np.asarray(abs_specs))
   # print('start tpcf')
    comb = combinations(abs_specs_f, 2)
    '''with concurrent.futures.ProcessPoolExecutor() as executor:
        result = [executor.submit(absdif, co) for co in comb]
        print('finish tpcf')
        # bla = [abs(a -b) for a, b in combinations(abs_specs_f, 2)]
        if pos_alpha == 'minor':
           bla2 = np.histogram(result,bins=minor_vel)
        elif pos_alpha == 'major':
           bla2 = np.histogram(result,bins=major_vel)
        bla_t = bla2[0]/len(result)
        return(bla_t)'''
    results = [absdif(co, pos_alpha) for co in comb]
    if pos_alpha == 'minor':
       bla2 = np.histogram(results,bins=minor_bins)
    elif pos_alpha == 'major':
       bla2 = np.histogram(results,bins=major_bins)
    bla_t = bla2[0]/len(results)
    print(' end TPCF', pos_alpha)
    return(bla_t)

def absdif(bla, bla2):
    #print('absdif',bla, bla2)
    a = bla[0]
    b = bla[1]
    return(abs(a -b))



###possible filling factor functions

def prob_hit_log_lin(r, r_vir, a, b, por_r_vir = 0.5):
    r_t = r/r_vir
    return(np.exp(a)*(np.exp(-b*r_t)))

#### define grids for the poor mans mcmc

bs = np.linspace(0.1,10,7) # characteristic radius of the exponential function (it is accually a porcentage of Rvir) in log scale to make the range more homogeneous in lin scale
csize = np.linspace(0.01,1,7) #poner en escala mas separada
hs = np.linspace(1,20,7) #bajar un poco para que no sea un  1,10,20
hv = np.linspace(0, 20,7) #bajar maximo a 100

#params = [bs,csize]

zabs = 0.77086
lam0 = 2796.35

vel_min = -1500
vel_max = 1500
lam_min = ((vel_min/const.c.to('km/s').value)+1)*(lam0*(1+zabs)) 
lam_max = ((vel_max/const.c.to('km/s').value)+1)*(lam0*(1+zabs)) 

w_spectral = 0.03

wave = np.arange(lam_min,lam_max+w_spectral, w_spectral)
vels_wave = (const.c.to('km/s').value * ((wave/ (lam0 * (1 + zabs))) - 1))

### run the model in the parameter grid

results_Wr = []
results_D = []
results_R_vir = []
results_specs = []
results_tpcf_minor = []
results_tpcf_major = []



for l in range(len(bs)):
    for i in range(len(csize)):
        for j in range(len(hs)):
            for k in range(len(hv)):
                print(l,i)
                exp_fill_fac = Sample.Sample(prob_hit_log_lin,200,sample_size=100, csize=csize[i], h=hs[j], hv=hv[k])
                e3_a_1 = exp_fill_fac.Nielsen_sample(np.log(100),bs[l],0.2)
                print('specs, alphas', len(e3_a_1[1]))
                cond_spec = e3_a_1[0] == 0
                spec_abs = e3_a_1[1][~cond_spec]
                alphas_abs = e3_a_1[2][~cond_spec]
                cond_minor = alphas_abs < 45
                cond_major = alphas_abs > 45
        
        
                spec_minor = spec_abs[cond_minor]
                spec_major = spec_abs[cond_major]
                specs_tot = [(spec_minor,'minor'), (spec_major, 'major')]
                print('empieza TPCF', l,i)
        
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    results = executor.map(TPCF, specs_tot)
                    list_res = list(results)
            
                    results_tpcf_minor.append(list_res[0])
                    results_tpcf_major.append(list_res[1])
            
            
        
       
                results_Wr.append(e3_a_1[8])
                results_D.append(e3_a_1[3])
                results_R_vir.append(e3_a_1[7])
        
                
                
results_Wr_r = np.reshape(results_Wr, (7,7,7,7,100))
results_D_r = np.reshape(results_D, (7,7,7,7,100))
results_R_vir_r = np.reshape(results_R_vir, (7,7,7,7,100))
results_r = [results_Wr_r, results_D_r, results_R_vir_r]
results_tpcf_minor_r = np.reshape(results_tpcf_minor,(7,7,7,7,len(minor_vel)))
results_tpcf_major_r = np.reshape(results_tpcf_major,(7,7,7,7,len(major_vel)))
#specs_r = np.reshape(results_specs, (10,10,300,len(wave)))

np.save('mp_mcmc_15_a', results_r)
np.save('mp_mcmc_15_tpcf_minor_a',results_tpcf_minor_r)
np.save('mp_mcmc_15_tpcf_major_a',results_tpcf_major_r)

#### hacer la parte de bootstrap####

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
                        
         
                            
                        
                        
    
        return(pgrid)
    
prob_2_boot = getpgrid_boot(results_r)
np.save('pgrid_boot_15_a', prob_2_boot)

print(time.process_time() - start)
