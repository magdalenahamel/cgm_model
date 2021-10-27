from tqdm.notebook import tqdm, trange
import time
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

import cgmspec.Sample as sample

###possible filling factor functions

def log_log(r,a,b):
    return(10**(a) * r**(b))

def prob_hit_log_lin(r, r_vir, a, b, por_r_vir = 0.5):
    r_t = r/r_vir
    return(10**(a)*(10**(-b*r_t)))


def prob_hit_don(r, r_vir, prob_rc, rmax, por_r_vir = 0.5):
        """
        Probability of hitting a cloud at distance r in the plane xy of a disc of radius rmax. For the moment is a power law.

        :param r: np.array, array with distances to the center of disc in plane xy in kpc
        :param r_0: float, the characteristicradius of the power law
        :param prob_rmin: float, probability at Rcore or below, default 100% probability of crossing a cloud
        :return: float, probability of hitting a cloud
        """

        prob = np.ones(len(r)) * prob_rc
        prob = np.where(r<=por_r_vir*r_vir,0, prob)
        prob = np.where(r>rmax*r_vir, 0, prob)
        return prob

def prob_hit_don_don(r, r_vir, prob_rc, rmax, por_r_vir = 0.5):
        """
        Probability of hitting a cloud at distance r in the plane xy of a disc of radius rmax. For the moment is a power law.

        :param r: np.array, array with distances to the center of disc in plane xy in kpc
        :param r_0: float, the characteristicradius of the power law
        :param prob_rmin: float, probability at Rcore or below, default 100% probability of crossing a cloud
        :return: float, probability of hitting a cloud
        """

        prob = np.ones(len(r)) * prob_rc
        prob = np.where(r>por_r_vir*r_vir,0, prob)
        prob = np.where(r>rmax*r_vir, 0, prob)
        return prob



def prob_hit_exp(r, r_vir, prob_rc, rmax, por_r_vir = 0.5):
        """
        Probability of hitting a cloud at distance r in the plane xy of a disc of radius rmax. For the moment is a power law.

        :param r: np.array, array with distances to the center of disc in plane xy in kpc
        :param r_0: float, the characteristicradius of the power law
        :param prob_rmin: float, probability at Rcore or below, default 100% probability of crossing a cloud
        :return: float, probability of hitting a cloud
        """
        print(r_vir, prob_rc,rmax,por_r_vir)
        rmin = 0.1
        A = 100
        B = -(por_r_vir) * (1/np.log(prob_rc/100))
        prob = A*np.exp(-r/(B*r_vir))
        prob = np.where(r<=rmin, 100, prob)
        prob = np.where(r>rmax*r_vir, 0, prob)
        return prob

def prob_hit_pow_law(r, r_vir, prob_rc, rmax, por_r_vir = 0.5):
        """
        Probability of hitting a cloud at distance r in the plane xy of a disc of radius rmax. For the moment is a power law.

        :param r: np.array, array with distances to the center of disc in plane xy in kpc
        :param r_0: float, the characteristicradius of the power law
        :param prob_rmin: float, probability at Rcore or below, default 100% probability of crossing a cloud
        :return: float, probability of hitting a cloud
        """
        rmin = 0.1
        ind = np.log(prob_rc/100)/(np.log(por_r_vir*r_vir)-np.log(rmin))
        b = r/r_vir
        A = prob_rc/((por_r_vir)**ind)
        prob = A * (b**(ind))
        prob = np.where(r<=rmin,100, prob)
        prob = np.where(prob>100,100, prob)
        prob = np.where(r>rmax*r_vir, 0, prob)
        return prob


#### define grids for the poor mans mcmc

bs = np.linspace(0.1,5,7) # characteristic radius of the exponential function (it is accually a porcentage of Rvir) in log scale to make the range more homogeneous in lin scale
csize = np.linspace(0.01,2,7) #poner en escala mas separada
hs = np.linspace(5,40,7) #bajar un poco para que no sea un  1,10,20
hv = np.linspace(0, 50,7) #bajar maximo a 100

params = [bs,csize,hs,hv]



### run the model in the parameter grid
results_Wr = []
results_D = []
results_R_vir = []

for l in range(len(bs)):
    for i in range(len(csize)):
        for j in range(len(hs)):
            for k in range(len(hv)):
                print(l,i,j,k)
                exp_fill_fac = sample.Sample(prob_hit_log_lin,200,sample_size=300, csize=csize[i], h=hs[j], hv=hv[k])
                e3_a_1 = exp_fill_fac.Nielsen_sample(2,bs[l],0.2)
                results_Wr.append(e3_a_1[8])
                results_D.append(e3_a_1[3])
                results_R_vir.append(e3_a_1[7])
                
                
results_Wr_r = np.reshape(results_Wr, (7,7,7,7,300))
results_D_r = np.reshape(results_D, (7,7,7,7,300))
results_R_vir_r = np.reshape(results_R_vir, (7,7,7,7,300))
results_r = [results_Wr_r, results_D_r, results_R_vir_r]

np.save('mcmc_3', results_r)
