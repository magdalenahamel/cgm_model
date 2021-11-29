import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

bs_3 = np.linspace(0.1,5,7) # characteristic radius of the exponential function (it is accually a porcentage of Rvir) in log scale to make the range more homogeneous in lin scale
csize_3 = np.linspace(0.01,2,7) #poner en escala mas separada
hs_3 = np.linspace(5,40,7) #bajar un poco para que no sea un  1,10,20
hv_3 = np.linspace(0, 50,7) #bajar maximo a 100

params_3 = [bs_3,csize_3,hs_3,hv_3]

params_name_3 = ['f_v', 'cloud size', 'disc height', 'velocity scale height']

results_r_3 = np.load('mcmc_3.npy')


results_r_3.shape

def getpgrid_boot(modelgrid, comparison_data, boot = 1000):
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
                        for m in range(boot):
                            print('b', m)
                            no_upper_sample = random.normal(loc=W_r_churchill_no_upper, scale=e_Wr_no_upper, size=None)
                            upper_sample = random.uniform(low=0.0, high=W_r_churchill_upper, size=None)
                            all_sample = np.concatenate((no_upper_sample, upper_sample))
                            p = ks2d2s(all_d,all_sample,model_D_R_vir, model_Wr)
                            ks.append(p)
                            
                        p_med = np.mean(ks)
                        pgrid[i][j][k][l] = p_med
                        
    
        return pgrid
    
    prob_3_boot = getpgrid_boot(results_r_3, magii_comp)
    
    np.save('pgrid_boot_3', pgrid)
