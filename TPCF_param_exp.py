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
import os

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

#### define grids for the parameter exploration

bs =  5 # characteristic radius of the exponential function (it is accually a porcentage of Rvir) in log scale to make the range more homogeneous in lin scale
csize = 1 #poner en escala mas separada
hs = [1, 10, 30] #bajar un poco para que no sea un  1,10,20
hv = 10 #bajar maximo a 100

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

results_nr_clouds = []
results_specs = []
results_alphas = []
results_D = []
results_vels = []
results_b = []
results_inclis = []
results_R_vir = []
results_Wr = []



results_tpcf_minor = []
results_tpcf_major = []



for l in range(len(csize)):
    print(l)
    exp_fill_fac = Sample.Sample(prob_hit_log_lin,200,sample_size=200, csize=csize, h=hs[l], hv=hv)
    e3_a_1 = exp_fill_fac.Nielsen_sample(np.log(100),bs,0.2)
    print('specs, alphas', len(e3_a_1[1]))
    cond_spec = e3_a_1[0] == 0
    spec_abs = e3_a_1[1][~cond_spec]
    alphas_abs = e3_a_1[2][~cond_spec]
    cond_minor = alphas_abs < 45
    cond_major = alphas_abs > 45


    spec_minor = spec_abs[cond_minor]
    spec_major = spec_abs[cond_major]
    specs_tot = [(spec_minor,'minor'), (spec_major, 'major')]
    print('empieza TPCF', l)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(TPCF, specs_tot)
        list_res = list(results)

        results_tpcf_minor.append(list_res[0])
        results_tpcf_major.append(list_res[1])



    results_nr_clouds.append(e3_a_1[0])
    results_specs.append(e3_a_1[1])
    results_alphas.append(e3_a_1[2])
    results_D.append(e3_a_1[3])
    results_vels.append(e3_a_1[4])
    results_b.append(e3_a_1[5])
    results_inclis.append(e3_a_1[6])
    results_R_vir.append(e3_a_1[7])
    results_Wr.append(e3_a_1[8])



dirName = 'TPCF_param_exp/V4'

os.makedirs(dirName)
#specs_r = np.reshape(results_specs, (10,10,300,len(wave)))

np.save('TPCF_param_exp/V4/nr_clouds', results_nr_clouds)
np.save('TPCF_param_exp/V4/specs', results_specs)
np.save('TPCF_param_exp/V4/alphas', results_alphas)
np.save('TPCF_param_exp/V4/D', results_D)
np.save('TPCF_param_exp/V4/vels', results_vels)
np.save('TPCF_param_exp/V4/b', results_b)
np.save('TPCF_param_exp/V4/inclis', results_inclis)
np.save('TPCF_param_exp/V4/R_vir', results_R_vir)
np.save('TPCF_param_exp/V4/Wr', results_Wr)

np.save('TPCF_param_exp/V4/tpcf_minor',results_tpcf_minor)
np.save('TPCF_param_exp/V4/tpcf_major',results_tpcf_major)


print(time.process_time() - start)
