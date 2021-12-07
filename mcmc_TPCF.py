from tqdm.notebook import tqdm, trange
import time
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from astropy import constants as const

import cgmspec.Sample as sample

from itertools import combinations

def TPCF(speci_empty_t):
    gauss_specs = []
    abs_specs = []
    vels_abs = []
    for m in range(len(speci_empty_t)):
        gauss_specj = filtrogauss(45000,0.03,2796.35,speci_empty_t[m])
        gauss_specs.append(gauss_specj)
        zabs=0.77086
        wave = np.arange(lam1,lam2,0.05)
        vels_wave = (const.c.to('km/s').value * ((wave/ (2796.35 * (1 + zabs))) - 1))
        cond_abs1 = gauss_specj < 0.98
        cond_abs2 = np.abs(vels_wave) < 1000
        abs_gauss_spec_major = vels_wave[cond_abs1 & cond_abs2]
        abs_specs.append(abs_gauss_spec_major)
    #vels_abs_major_i = [abs(i-j) for i in abs_gauss_spec_major for j in abs_gauss_spec_major if i != j]
    #vels_abs.append(vels_abs_major_i)

# Convert input list to a numpy array
    abs_specs_f = np.concatenate(np.asarray(abs_specs))
    bla = [abs(a -b) for a, b in combinations(abs_specs_f, 2)]
    bla2 = np.histogram(bla,bins=minor_vel)
    return(bla2)


def log_log(r,a,b):
    return(10**(a) * r**(b))

def prob_hit_log_lin(r, r_vir, a, b, por_r_vir = 0.5):
    r_t = r/r_vir
    return(10**(a)*(10**(-b*r_t)))

minor_tpcf = pd.read_csv('2minor.txt', delimiter='     ', engine='python')
major_tpcf = pd.read_csv('2major.txt', delimiter='     ', engine='python')

minor_vel = minor_tpcf['vel'].to_numpy()
minor_tpcf_val = minor_tpcf['TPCF'].to_numpy()
minor_error = np.abs(minor_tpcf['minus_error'].to_numpy() - minor_tpcf['plus_error'].to_numpy())

major_vel = major_tpcf['vel'].to_numpy()
major_tpcf_val = major_tpcf['TPCF'].to_numpy()
major_error = np.abs(major_tpcf['minus_error'].to_numpy() - major_tpcf['plus_error'].to_numpy())

def chisq(obs, exp, error):
    return np.sum((obs - exp) ** 2 / (error ** 2))




from astropy.convolution import convolve, Gaussian1DKernel
def filtrogauss(R, spec_res, lam_0, flux):
    del_lam = lam_0/R
    del_lam_pix = del_lam/spec_res
    gauss_kernel = (Gaussian1DKernel(del_lam_pix))
    gausflux = convolve(flux, gauss_kernel)
    return(gausflux)

bs = np.linspace(0.1,5,7) 
csize = np.linspace(0.01,2,7) 
hs = np.linspace(5,40,7) 
hv = np.linspace(0, 50,7) 

params = [bs,csize,hs,hv]


results_TPCF_minor = []
results_TPCF_major = []
results_R_vir = []

for l in range(len(bs)):
    for i in range(len(csize)):
        for j in range(len(hs)):
            for k in range(len(hv)):
                print(l,i,j,k)
                exp_fill_fac = sample.Sample(prob_hit_log_lin,200,sample_size=300, csize=csize[i], h=hs[j], hv=hv[k])
                e3_a_1 = exp_fill_fac.Nielsen_sample(2,bs[l],0.2)


                cond_minor = (np.fabs(e3_a_1[2])>45) & (np.fabs(e3_a_1[2])<135)
                cond_major = (np.fabs(e3_a_1[2])<45) | (np.fabs(e3_a_1[2])>135)

                specs_minor = e3_a_1[1][cond_minor]
                specs_major = e3_a_1[1][cond_major]
                
                TPCF_model_minor = TPCF(specs_minor)
                TPCF_model_major = TPCF(specs_major)

                results_TPCF_minor.append(TPCF_model_minor)
                results_TPCF_major.append(TPCF_model_major)

np.save('TPCF_minor_2', results_TPCF_minor)
np.save('TPCF_major_2', rresults_TPCF_major)
