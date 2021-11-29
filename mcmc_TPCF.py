from tqdm.notebook import tqdm, trange
import time
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from astropy import constants as const

import cgmspec.Sample as sample

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


def TPCF(spec, vel):
    cond1 = spec < 0.98
    cond2 = np.abs(vels_wave) < 1000
    abs_spec = vel[cond1 & cond2]
    sub_arr = np.abs(abs_spec[:,None] - abs_spec)
    N = abs_spec.size
    rem_idx = np.arange(N)*(N+1)
    out = np.delete(sub_arr,rem_idx)
    return(out)


from astropy.convolution import convolve, Gaussian1DKernel
def filtrogauss(R, spec_res, lam_0, flux):
    del_lam = lam_0/R
    del_lam_pix = del_lam/spec_res
    gauss_kernel = (Gaussian1DKernel(del_lam_pix))
    gausflux = convolve(flux, gauss_kernel)
    return(gausflux)

bs = np.linspace(0.1,20,3) # characteristic radius of the exponential function (it is accually a porcentage of Rvir) in log scale to make the range more homogeneous in lin scale
csize = np.asarray([0.1, 1, 10]) #poner en escala mas separada
hs = np.linspace(1,50,3) #bajar un poco para que no sea un  1,10,20
hv = np.linspace(0, 50,3) #bajar maximo a 100

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

                gauss_specs_minor = []
                abs_specs_minor = []
                vels_abs_minor = []

                gauss_specs_major = []
                abs_specs_major = []
                vels_abs_major = []

                for m in range(len(specs_minor)):
                    gauss_specj = filtrogauss(45000,0.03,2796.35,specs_minor[m])
                    gauss_specs_minor.append(gauss_specj)
                    zabs=0.77086
                    wave = np.arange(4849.58349609375,5098.33349609375+0.125, 0.03)
                    vels_wave = (const.c.to('km/s').value * ((wave/ (2796.35 * (1 + zabs))) - 1))
                    cond_abs1 = gauss_specj < 0.98
                    cond_abs2 = np.abs(vels_wave) < 1000
                    abs_gauss_spec_minor = vels_wave[cond_abs1 & cond_abs2]
                    abs_specs_minor.append(abs_gauss_spec_minor)
                    vels_abs_minor_i = [abs(i-j) for i in abs_gauss_spec_minor for j in abs_gauss_spec_minor if i != j]
                    vels_abs_minor.append(vels_abs_minor_i)

                for m in range(len(specs_major)):
                    gauss_specj = filtrogauss(45000,0.03,2796.35,specs_major[m])
                    gauss_specs_major.append(gauss_specj)
                    zabs=0.77086
                    wave = np.arange(4849.58349609375,5098.33349609375+0.125, 0.03)
                    vels_wave = (const.c.to('km/s').value * ((wave/ (2796.35 * (1 + zabs))) - 1))
                    cond_abs1 = gauss_specj < 0.98
                    cond_abs2 = np.abs(vels_wave) < 1000
                    abs_gauss_spec_major = vels_wave[cond_abs1 & cond_abs2]
                    abs_specs_major.append(abs_gauss_spec_major)
                    vels_abs_major_i = [abs(i-j) for i in abs_gauss_spec_major for j in abs_gauss_spec_major if i != j]
                    vels_abs_major.append(vels_abs_major_i)

# Convert input list to a numpy array

                out_minor = [item for sublist in vels_abs_minor for item in sublist]
                out_major = [item for sublist in vels_abs_major for item in sublist]


                TPCF_model_minor = np.histogram(out_minor, bins=minor_vel, density=True)
                TPCF_model_major = np.histogram(out_major, bins=major_vel, density=True)

                results_TPCF_minor.append(TPCF_model_minor)
                results_TPCF_major.append(TPCF_model_major)

np.save('TPCF_minor_1', results_TPCF_minor)
np.save('TPCF_major_1', rresults_TPCF_major)
