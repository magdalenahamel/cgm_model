from cgmspec import utils as csu
import pandas as pd

import cgmspec.disco as cgm
from cgmspec.sampledist import RanDist

from cgmspec import utils as csu
import numpy as np
import matplotlib.pyplot as plt
from astropy import constants as const

from mpdaf.obj import Image
from mpdaf.obj import Cube
from mpdaf.obj import WCS, WaveCoord

import astropy.units as u
from astropy.coordinates import SkyCoord

from cgmspec.sampledist import RanDist

from scipy.special import wofz
from scipy.special import gamma

from scipy import stats
import random
from astropy.cosmology import FlatLambdaCDM

import importlib
from numpy import float32

#### Define the spectral resolution ####

zabs = 0.77086
lam0 = 2796.35

vel_min = -1500
vel_max = 1500
lam_min = ((vel_min/const.c.to('km/s').value)+1)*(lam0*(1+zabs)) 
lam_max = ((vel_max/const.c.to('km/s').value)+1)*(lam0*(1+zabs)) 

w_spectral = 0.03

wave = np.arange(lam_min,lam_max+w_spectral, w_spectral)
vels_wave = (const.c.to('km/s').value * ((wave/ (lam0 * (1 + zabs))) - 1))

##### Data ####

magiicat_iso = data_r_vir = pd.read_csv('magiicat_isolated.txt', error_bad_lines=False, delim_whitespace=True)
D_magiicat = magiicat_iso['D'].to_numpy()
R_vir_magiicat = D_magiicat/magiicat_iso['D/R_vir'].to_numpy()
v_magiicat = magiicat_iso['V_circ'].to_numpy()
z_gal_magiicat = magiicat_iso['z_gal'].to_numpy()

D_R_v_magiicat = np.array((D_magiicat,R_vir_magiicat, v_magiicat, z_gal_magiicat)).T

churchill_iso = data_r_vir = pd.read_csv('Churchill_iso_full.txt', error_bad_lines=False, delim_whitespace=True)
D_R_vir_churchill_iso = churchill_iso['etav'].to_numpy()
W_r_churchill_iso = churchill_iso['Wr'].to_numpy()
D_churchill_iso = churchill_iso['D'].to_numpy()

chen_iso = pd.read_csv('chen_data.txt', error_bad_lines=False, delim_whitespace=True)
D_chen = chen_iso['rho'].to_numpy()

#### Parameters distributions ####

'''Inclination distribution'''
def sin_i_dist(y, ymin):
    Ay = 1/np.sqrt(1-ymin**2)
    return(Ay * y / np.sqrt(1-(y**2)))

sinivals = np.linspace(np.sin(np.radians(5.7)),0.99,100)
fN = RanDist(sinivals, sin_i_dist(sinivals,np.radians(5.7)))

'''Doppler parameter distribution'''

df = 6.71701 # parametro de forma.
chi2 = stats.chi2(df)
x = np.linspace(0,20)
fp = chi2.pdf(x) # Función de Probabilidad

bvals = np.linspace(0,20,100)
fNb = RanDist(bvals, chi2.pdf(bvals))

'''v_max distribution'''

def rot_vel_dist(v, phi_c, v_c, alpha, beta):
    a = phi_c*((v/v_c)**alpha)
    b = np.exp(-(v/v_c)**beta)
    c = beta/gamma(alpha/beta)
    return(a*b)

'''N distribution'''

def ndist(n, beta = 1.5 ):
    return n**-beta

nvals = np.logspace(12.6, 16, 1000)
fN = RanDist(nvals, ndist(nvals))


#### Model fuctions ####

def get_clouds(ypos,zpos,probs,velos):
    randomnum = np.random.uniform(0, 100, len(probs))
    selected = probs >= randomnum
    return(velos[selected])

def averagelos(model, D, alpha, lam, iter,X, z, grid_size, b, r_0, v_max, h_v, v_inf, results):
    h = model.h
    incli = model.incl

        #list of len(iter) velocities satisfying prob_hit
    results = np.asarray(results)



    fluxes = [0]*iter
    fluxtoaver = [losspec(model, lam,results[x],X,b,z) for x in fluxes]
    fluxtoaver = np.asarray(fluxtoaver)
    totflux = np.median(fluxtoaver, axis=0)

    return(totflux)

def ndist(n, beta = 1.5 ):
    return n**-beta

def losspec(model,lam,velos,X, b,z):
    nvals = np.logspace(12.6, 16, 1000)
    fN = RanDist(nvals, ndist(nvals))

    Ns = fN.random(len(velos))
        #print(Ns)
    N = np.empty([len(velos), 1])
    for i in range(len(Ns)):
        N[i,0]=Ns[i]
    taus = Tau(lam,velos,X,N,b,z)
    tottau = np.sum(taus,axis=0)
    return(np.exp(-tottau))

def Tau(lam,vel,X,N, b,z):
    if X ==1:
        lam0 = [2796.35]
        f = [0.6155]
    if X ==2:
        lam0 = [2803.53]
        f = [0.3054]
    if X ==12:

        lam0 = [2796.35, 2803.53]
        f = [0.6155, 0.3054]

    gamma, mass = [2.68e8, 24.305]
    c  = const.c.to('cm/s').value
    sigma0 = 0.0263
    taus = []
    for i in range(len(lam0)):

        lamc = ((vel[:,None]/const.c.to('km/s').value)+1)*((lam0[i]))
            #print('lamc', lamc)
        nu = c/(lam*1e-8)
        nu0 = c/(lamc*1e-8)

        dnu = nu - (nu0/(1+z))
        dnud = (b[:, np.newaxis]*100000)*nu/c
       # print(len(dnud))

        x = dnu/dnud
        y = gamma/(4*np.pi*dnud)
        zi = x + 1j*y
        v = np.asarray(np.real(wofz(zi)/(np.sqrt(np.pi)*dnud)))

        #print('N', N)
        #print('v', v)

        taut =N * sigma0*f[i] * v

        taus.append(taut)

    taus = np.asarray(taus)
    taust = taus.sum(axis=0)
    #print(taust)
    return(taust)

def get_cells(model,D,alpha,size,r_0,p_r_0, vR,hv,prob_func,  rmax, por_r_vir):
    
    h = model.h
    incli = model.incl

    m = -np.tan(np.radians(90-incli))


    x0 = D * np.cos(np.radians(alpha))
    y0 = D*np.sin(np.radians(alpha))/np.cos(np.radians(incli))
    n = -m*y0

    y1 = ((h/2)-n)/m
    y2 = (-(h/2)-n)/m
        #print('y1y2', y1,y2)
    mindis = np.sqrt(2*(size**2))/2
    z1 = h/2
    z2 = -h/2
    b = -1
    zgrid = np.arange((-h/2) + (size/2), (h/2) + (size/2), size)
    ymin = int(y1/size) * size + (size/2)
    ymax = int(y2/size)*size +(size/2)



        #print('yminmax', ymin,ymax)
    ygrid = np.arange(ymin,ymax,size)
    points = abs((m * ygrid + b * zgrid[:,None] + n)) / (np.sqrt(m * m + b * b))
    selected = points <= mindis
    yv, zv = np.meshgrid(ygrid, zgrid)
    ypos = yv[selected]
    zpos = zv[selected]
        #print('yposis', ypos)
    radios = np.sqrt((x0**2)+ypos**2)
    probs = prob_func(radios, r_0, p_r_0, rmax, por_r_vir)
    velos = los_vel(model, ypos, D, alpha, vR, hv)
    return(ypos,zpos, probs, velos)

def los_vel(model, y, D, alpha, vR, hv, v_inf=0):

    v_los_inf = (v_inf * np.sin(model.incl_rad)) * (y/(np.sqrt((y**2) + D**2)))
    al_rad = np.radians(alpha)

    R = D * np.sqrt(1+(np.sin(al_rad)**2)*np.tan(model.incl_rad)**2)
    vrot = (2/np.pi)*np.arctan2(R,1)

    x0 = D * np.cos(al_rad)  # this is p in Ho et al. 2019, fig 10.
        #print('al,x0',al_rad,x0)
    y0 = D * np.sin(al_rad) / np.cos(model.incl_rad)  # this is y0 in the same fig.
    if x0>=0:
        a = np.sin(model.incl_rad) / np.sqrt(1 + (y/x0)**2)
    else:
        a = -np.sin(model.incl_rad) / np.sqrt(1 + (y/x0)**2)

    b = np.exp(-np.fabs(y - y0) / hv * np.tan(model.incl_rad))
        #print(b)
    vr = (vR*vrot*a*b) + v_los_inf

        #print('vel', vr)
    return(vr)

xs = np.linspace(-200,200,2*200)
ys = np.linspace(-200,200,2*200)
x, y = np.meshgrid(xs, ys)
print('before csu')
d_alpha_t = csu.xy2alpha(x, y)
print('after csu')
ds = []
alphas = []
for i in range(len(d_alpha_t[0])):
    for j in range(len(d_alpha_t[0][0])):
        if d_alpha_t[0][i][j]>200:
           pass
        else:
           ds.append(d_alpha_t[0][i][j])
           alphas.append(d_alpha_t[1][i][j])


"""This Class Sample represents a sample of MgII absorbers from Galaxies with the model Disco"""


class Sample:
    """Represents a sample of MgII absorbers"""

    def __init__(self, filling_factor, dmax, h=10, w_pix = 0.03, zabs=0.77086, csize=1, hv=10, sample_size=2000):
        """c"""
        self.filling_factor  = filling_factor
        self.dmax = dmax
        self.h = h
        self.w_pix = w_pix
        self.zabs=zabs
        self.csize=csize
        self.hv=hv
        self.sample_size=sample_size


    def Nielsen_sample(self, prob_r_cs, rmax, por_r_vir):
        print('runing sample bla')
        dmax = self.dmax
        filling_factor = self.filling_factor
        dmax = self.dmax
        h = self.h
        w_pix = self.w_pix
        zabs = self.zabs
        csize = self.csize
        hv = self.hv
        sample_size = self.sample_size

        xs = np.linspace(-dmax,dmax,2*dmax)
        ys = np.linspace(-dmax,dmax,2*dmax)
        x, y = np.meshgrid(xs, ys)
        print('before csu')
        d_alpha_t = csu.xy2alpha(x, y)
        print('after csu')
        '''ds = []
        alphas = []
        for i in range(len(d_alpha_t[0])):
            for j in range(len(d_alpha_t[0][0])):
                if d_alpha_t[0][i][j]>dmax:
                    pass
                else:
                    ds.append(d_alpha_t[0][i][j])
                    alphas.append(d_alpha_t[1][i][j])
        prrint('runed alphas')'''
        #wave = np.arange(4849.58349609375,5098.33349609375+0.125, w_pix)
        #vels_wave = (const.c.to('km/s').value * ((wave/ (2796.35 * (1 + zabs))) - 1))


        z_median = np.median(z_gal_magiicat)
        R_vir_min = np.min(R_vir_magiicat)
        R_vir_max = np.max(R_vir_magiicat)

        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        H = cosmo.H(z_median)
        vel_min = R_vir_min * u.kpc * H / 0.1
        vel_min = vel_min.to(u.km/u.second).value

        vel_max = R_vir_max * u.kpc * H / 0.1
        vel_max = vel_max.to(u.km/u.second).value

        vels = np.linspace(vel_min,vel_max,1000)

        vels_dist = rot_vel_dist(vels,0.061,10**2.06, 0.66, 2.10)
        fN_v = RanDist(vels, vels_dist)

        d_alpha = list(zip(ds,alphas))

        random_nr_clouds = []
        random_specs = []
        random_alphas = []
        random_im_par = []
        random_vels = []
        random_b = []
        random_inclis = []
        random_r_vir = []
        random_equi_wid = []


        alpha_i = random.choices(alphas, k=sample_size)
        #d_i = random.choices(ds, k=sample_size)
        d_i = f_D_C.random(sample_size)


        random_vels_i = f_v.random(sample_size)

        #random_vels_i = fN_v.random(sample_size)
        random_r_vir_i = (random_vels_i * u.km /u.second)*0.1/H
        random_r_vir_i = random_r_vir_i.to(u.kpc).value


        random_inclis_i = fN.random(sample_size)
        random_inclis_i = np.degrees(np.arcsin(random_inclis_i))
        random_nr_clouds_pow_i = []
        random_specs_pow_i = []
        random_equi_wid_pow_i =[]
        print('before loop')
        for i in range(sample_size):
            print('running samile i')
            d = d_i[i]
            alpha = alpha_i[i]

            model = cgm.Disco(h, random_inclis_i[i], Rcore=0.1)
            cells = get_cells(model,d,alpha,csize, random_r_vir_i[i],prob_r_cs,random_vels_i[i],hv,self.filling_factor,  rmax, por_r_vir)
            results = [0]*1
            results = [get_clouds(cells[0],cells[1],cells[2],cells[3]) for x in results]
            results_nr = csu.nr_clouds(results, 6.6)
            b = fNb.random(len(results[0]))
            speci = averagelos(model, d, alpha, wave, 1,1, zabs, csize, b, 0, random_vels_i[i], hv, 0, results)
            random_specs.append(speci)
            random_nr_clouds.append(results_nr[0])
            equi_wid_i = csu.eq_w(speci, vels_wave, random_vels_i[i]+20, zabs,  w_pix)
            random_equi_wid.append(equi_wid_i)
            #print(i)
        print('after loop')

        return(np.asarray([np.asarray(random_nr_clouds),
        np.asarray(random_specs),
        np.asarray(alpha_i),
        np.asarray(d_i),
        np.asarray(random_vels_i),
        np.asarray(random_b),
        np.asarray(random_inclis_i),
        np.asarray(random_r_vir_i),
        np.asarray(random_equi_wid)]))


'''def Chen_sample(self, prob_r_cs, rmax, por_r_vir):
    dmax = self.dmax
    filling_factor = self.filling_factor
    dmax = self.dmax
    h = self.h
    w_pix = self.w_pix
    zabs = self.zabs
    csize = self.csize
    hv = self.hv
    sample_size = self.sample_size

    xs = np.linspace(-dmax,dmax,2*dmax)
    ys = np.linspace(-dmax,dmax,2*dmax)
    x, y = np.meshgrid(xs, ys)
    d_alpha_t = csu.xy2alpha(x, y)

    ds = []
    alphas = []
    for i in range(len(d_alpha_t[0])):
        for j in range(len(d_alpha_t[0][0])):
            if d_alpha_t[0][i][j]>dmax:
                pass
            else:
                ds.append(d_alpha_t[0][i][j])
                alphas.append(d_alpha_t[1][i][j])

    wave = np.arange(4849.58349609375,5098.33349609375+0.125, w_pix)
    vels_wave = (const.c.to('km/s').value * ((wave/ (2796.35 * (1 + zabs))) - 1))


    z_median = np.median(z_gal_magiicat)
    R_vir_min = np.min(R_vir_magiicat)
    R_vir_max = np.max(R_vir_magiicat)

    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    H = cosmo.H(z_median)
    vel_min = R_vir_min * u.kpc * H / 0.1
    vel_min = vel_min.to(u.km/u.second).value

    vel_max = R_vir_max * u.kpc * H / 0.1
    vel_max = vel_max.to(u.km/u.second).value

    vels = np.linspace(vel_min,vel_max,1000)

    vels_dist = rot_vel_dist(vels,0.061,10**2.06, 0.66, 2.10)


    fN_v = RanDist(vels, vels_dist)





    d_alpha = list(zip(ds,alphas))

    random_nr_clouds = []
    random_specs = []
    random_alphas = []
    random_im_par = []
    random_vels = []
    random_b = []
    random_inclis = []
    random_r_vir = []
    random_equi_wid = []


    alpha_i = random.choices(alphas, k=sample_size)
    #d_i = random.choices(ds, k=sample_size)
    d_i = f_D_chen.random(sample_size)


    random_vels_i = f_v.random(sample_size)

    #random_vels_i = fN_v.random(sample_size)
    random_r_vir_i = (random_vels_i * u.km /u.second)*0.1/H
    random_r_vir_i = random_r_vir_i.to(u.kpc).value


    random_inclis_i = fN.random(sample_size)
    random_inclis_i = np.degrees(np.arcsin(random_inclis_i))
    random_nr_clouds_pow_i = []
    random_specs_pow_i = []
    random_equi_wid_pow_i =[]






    for i in range(sample_size):
        d = d_i[i]
        alpha = alpha_i[i]

        model = cgm.Disco(h, random_inclis_i[i], Rcore=0.1)
        cells = get_cells(model,d,alpha,csize, random_r_vir_i[i],prob_r_cs,random_vels_i[i],hv,self.filling_factor,  rmax, por_r_vir)
        results = [0]*1
        results = [get_clouds(cells[0],cells[1],cells[2],cells[3]) for x in results]
        results_nr = csu.nr_clouds(results, 6.6)
        b = fNb.random(len(results[0]))
        speci = averagelos(model, d, alpha, wave, 1,1, zabs, csize, b, 0, random_vels_i[i], hv, 0, results)
        random_specs.append(speci)
        random_nr_clouds.append(results_nr[0])
        equi_wid_i = csu.eq_w(speci, vels_wave, random_vels_i[i], zabs,  w_pix)
        random_equi_wid.append(equi_wid_i)
        #print(i)

    return(np.asarray([np.asarray(random_nr_clouds),
    np.asarray(random_specs),
    np.asarray(alpha_i),
    np.asarray(d_i),
    np.asarray(random_vels_i),
    np.asarray(random_b),
    np.asarray(random_inclis_i),
    np.asarray(random_r_vir_i),
    np.asarray(random_equi_wid)]))'''
