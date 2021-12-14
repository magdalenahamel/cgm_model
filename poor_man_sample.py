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

###possible filling factor functions

def prob_hit_log_lin(r, r_vir, a, b, por_r_vir = 0.5):
    r_t = r/r_vir
    return(np.exp(a)*(np.exp(-b*r_t)))

#### define grids for the poor mans mcmc

'''bs = np.linspace(0.1,4,7) # characteristic radius of the exponential function (it is accually a porcentage of Rvir) in log scale to make the range more homogeneous in lin scale
csize = np.linspace(0.01,1,7) #poner en escala mas separada
hs = np.linspace(1,20,7) #bajar un poco para que no sea un  1,10,20
hv = np.linspace(0, 20,7) #bajar maximo a 100'''

'''bs = np.linspace(0.1,4,2) # characteristic radius of the exponential function (it is accually a porcentage of Rvir) in log scale to make the range more homogeneous in lin scale
csize = np.linspace(1,10,2) #poner en escala mas separada
hs = np.linspace(10,20,2) #bajar un poco para que no sea un  1,10,20
hv = np.linspace(0, 20,2) #bajar maximo a 100
params = [bs,csize,hs,hv]'''

bs = np.linspace(0.1,4,10) # characteristic radius of the exponential function (it is accually a porcentage of Rvir) in log scale to make the range more homogeneous in lin scale
csize = np.linspace(0.01,2,10) #poner en escala mas separada
hs = 5 #bajar un poco para que no sea un  1,10,20
hv = 10 #bajar maximo a 100

params = [bs,csize]

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

for l in range(len(bs)):
    for i in range(len(csize)):
        print(l,i)
        exp_fill_fac = Sample.Sample(prob_hit_log_lin,200,sample_size=300, csize=csize[i], h=hs, hv=hv)
        e3_a_1 = exp_fill_fac.Nielsen_sample(np.log(100),bs[l],0.2)
        results_Wr.append(e3_a_1[8])
        results_D.append(e3_a_1[3])
        results_R_vir.append(e3_a_1[7])
        results_specs.append(e3_a_1[1])

'''for l in range(len(bs)):
    for i in range(len(csize)):
        print(l,i)
        exp_fill_fac = sample.Sample(prob_hit_log_lin,200,sample_size=300, csize=csize[i], h=hs, hv=hv)
        e3_a_1 = exp_fill_fac.Nielsen_sample(np.log(100),bs[l],0.2)
        results_Wr.append(e3_a_1[8])
        results_D.append(e3_a_1[3])
        results_R_vir.append(e3_a_1[7])
        results_specs.append(e3_a_1[1])'''
                
                
results_Wr_r = np.reshape(results_Wr, (10,10,300))
results_D_r = np.reshape(results_D, (10,10,300))
results_R_vir_r = np.reshape(results_R_vir, (10,10,300))
results_r = [results_Wr_r, results_D_r, results_R_vir_r]
specs_r = np.reshape(results_specs, (10,10,300,len(wave)))

### Multiprocess ###

#groups = list(list(itertools.product(bs,csize,hs,hv)))

#print('g', groups)

'''def get_sample(bla):
    print('empieza:', bla)
    bs, csize, hs, h_v = bla[0], bla[1],bla[2], bla[3]
    exp_fill_fac = Sample.Sample(prob_hit_log_lin,200,sample_size=300, csize=csize, h=hs, hv=h_v)
    print('crea:', bs, csize, hs, h_v)
    e3_a_1 = exp_fill_fac.Nielsen_sample(np.log(100),bs,0.2)
    #print('calcula:', bla)
    Wr = e3_a_1[8]
    D = e3_a_1[3]
    R_vir = e3_a_1[7]
    specs = e3_a_1[1]
    print('termina',bs, csize, hs, h_v)
    return(Wr, D, R_vir, specs)
    
    
results = map(get_sample, groups)
list=list(results)'''
'''with concurrent.futures.ProcessPoolExecutor() as executor:
    results = executor.map(get_sample, groups)
    list = list(results)'''
    
#for r in results:
    #print(r)

    
'''EW_results = results[:][0]
D_results = results[:][1]
R_vir_results = results[:][2]
specs_results = results[:][3]

EW_r = np.reshape(EW_results, (7,7,7,7,300))
D_r = np.reshape(D_results, (7,7,7,7,300))
R_vir_r = np.reshape(R_vir_results, (7,7,7,7,300))
specs_r = np.reshape(specs_results, (7,7,7,7,300,len(wave)))
results_r = [results_Wr_r, results_D_r, results_R_vir_r]'''


np.save('mp_mcmc_9', results_r)
np.save('mp_mcmc_9_specs',specs_r)
