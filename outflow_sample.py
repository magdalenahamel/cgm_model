import numpy as np
from math import sqrt
from cgmspec import utils as csu
from astropy import constants as const
import matplotlib.pyplot as plt
import astropy.units as u
from mpdaf.obj import Cube, WCS, WaveCoord
from astropy import coordinates as coord
import pandas as pd

def resolver_ecuacion_cuadrada(a, b, c):
    soluciones = [] # creamos una lista vacía para las soluciones

    discriminante = b * b - 4 * a * c

    if discriminante >= 0:  # comprobamos si existen soluciones reales
        raiz = sqrt(discriminante)
        soluciones.append((-b + raiz) / (2 * a))# calculamos una primera solución
        if discriminante != 0:
           soluciones.append((-b - raiz) / (2 * a)) # calculamos la segunda solución si existe

    return soluciones


def out_los_vel(y, z, theta, D, alpha, incli, vel, r0):
    x0 = D * np.cos(np.radians(alpha))
    #rho_t, theta_t, phi_t = coord.cartesian_to_spherical(x0, y, z)

    a = np.tan(np.radians(theta))
    z0 = r0 / a

    z_t = np.asarray([i+z0 if i>=0 else -z0 for i in z])
    #z_t = z + z0

    losy = -np.sin(np.radians(incli))
    losz = np.cos(np.radians(incli))


    #print(vy,vz)

    af = np.sqrt((vel**2)/(x0*x0 + y*y +z_t*z_t))

    return(af*(losy*y + losz*z_t))
    ''' uv = -y0*np.sin(theta_t)*np.sin(phi_t) + n*np.cos(theta_t)
    magv = np.sqrt(y0*y0 + n*n)

    uvmag = uv/(magv*magv)

    return((z/np.abs(z))*vel* np.sqrt(uvmag*uvmag*((y0**2)+n**2)))'''
    #return((z/np.abs(z))*100 * np.ones(len(y)))

def los(y, incli, D, alpha):
    m = np.tan(np.radians(90)+np.radians(incli))
    y0 = -D*np.sin(np.radians(alpha))/np.cos(np.radians(incli))
    n = -m*y0
    return(y*m +n)

def para_pos(y,r0, theta, D, alpha):
    x0 = D * np.cos(np.radians(alpha))
    a = np.tan(np.radians(theta))
    z0 = r0 / a
    return(np.sqrt((x0*x0)/(a*a)+((y*y)/(a*a))) - z0)


def para_neg(y,r0, theta, D, alpha):
    x0 = D * np.cos(np.radians(alpha))
    a = np.tan(np.radians(theta))
    z0 = r0 / a
    return(-np.sqrt((x0*x0)/(a*a)+((y*y)/(a*a))) + z0)

def inter_los_par_pos (r0, theta, D,alpha,incli, zmax):
    x0 = D * np.cos(np.radians(alpha))
    a = np.tan(np.radians(theta))
    m = np.tan(np.radians(90)+np.radians(incli))
    y0 = -D*np.sin(np.radians(alpha))/np.cos(np.radians(incli))
    n = -m*y0
    z0 = r0 / a

    a_t = (1/(a*a))-(m*m)
    b_t = -2*m*n  - 2*m*z0
    c_t = (x0*x0)/(a*a) - 2*n*z0 -(n*n) -(z0*z0)

    #print(a_t,b_t,c_t)

    y_sol = resolver_ecuacion_cuadrada(a_t, b_t, c_t)
    y_sol = np.asarray(y_sol)

    #print(y_sol)
    z_sol = y_sol*m +n

   # print(z_sol)
    condz = (z_sol > 0) & (z_sol < zmax)
    y_sol_t = y_sol[condz]
    z_sol_t = z_sol[condz]
    if len(y_sol_t) == 1:
        #print(1)
        z_sol_2 = zmax
        y_sol_2 = (z_sol_2-n)/m
        return(np.array([y_sol_2, y_sol_t[0]]),np.array([z_sol_2, z_sol_t[0]]) )
    elif len(y_sol_t) == 0:
        pass
    else:
        #print(3)
        return(y_sol[condz], z_sol[condz])


def inter_los_par_neg (r0, theta, D,alpha,incli, zmax):
    x0 = D * np.cos(np.radians(alpha))
    a = np.tan(np.radians(theta))
    m = np.tan(np.radians(90)+np.radians(incli))
    y0 = -D*np.sin(np.radians(alpha))/np.cos(np.radians(incli))
    n = -m*y0
    z0 = r0 / a
    a_t = (1/(a*a))-(m*m)
    b_t = -2*m*n  + 2*m*z0
    c_t = (x0*x0)/(a*a) + 2*n*z0 -(n*n) -(z0*z0)
    #print(a_t, b_t, c_t)
    y_sol = resolver_ecuacion_cuadrada(a_t, b_t, c_t)
   # print(y_sol)
    y_sol = np.asarray(y_sol)
    z_sol = y_sol*m +n
   # print(z_sol)
    condz = (z_sol < 0) & (z_sol > -zmax)
    y_sol_t = y_sol[condz]
    z_sol_t = z_sol[condz]
    if len(y_sol_t) == 1:
        z_sol_2 = -zmax
        y_sol_2 = (z_sol_2-n)/m
        return(np.array([y_sol_2, y_sol_t[0]]),np.array([z_sol_2, z_sol_t[0]]) )
    elif len(y_sol_t) == 0:
        pass

    else:
        return(y_sol[condz], z_sol[condz])

def plot_test_inter(y,r0,theta,D,alpha,incli, zmax, size):
    z_los = los(y, incli, D, alpha)
    z_par_pos = para_pos(y,r0, theta, D, alpha)
    z_par_neg = para_neg(y,r0,theta,D,alpha)
    inter_pos = inter_los_par_pos(r0, theta, D,alpha,incli, zmax)
    inter_neg = inter_los_par_neg(r0, theta, D,alpha,incli, zmax)

    cloud_pos = get_cels(r0, theta, D,alpha,incli, zmax, size)


    plt.plot(y, z_los)
    plt.plot(y, z_par_pos,'r')
    plt.plot(y, z_par_neg,'b')
    if cloud_pos:
        plt.scatter(cloud_pos[0], cloud_pos[1], c='g')
    if inter_pos:
        plt.scatter(inter_pos[0][0],inter_pos[1][0],c='r')
        plt.scatter(inter_pos[0][1],inter_pos[1][1], c='r')
    if inter_neg:
        plt.scatter(inter_neg[0][0],inter_neg[1][0], c='b')
        plt.scatter(inter_neg[0][1],inter_neg[1][1], c='b')

    major_ticks = np.arange(-200, 201, size)
    plt.xticks(major_ticks)
    plt.yticks(major_ticks)

    plt.ylim(-zmax-5, zmax+5)
    #plt.grid(True, which='major', axis='both')
    plt.show()


def prob(r0, ypos, zpos, theta, alpha, D):
    x0 = D * np.cos(np.radians(alpha))
    r = np.sqrt(x0*x0 + ypos*ypos + zpos*zpos)
    #r = np.abs(zpos * np.tan(np.radians(theta)))
    #print('r', r)
    return(100/(r**2))
    #return(np.ones(len(ypos)) * 100)



def get_cells_pos(r0, theta, D,alpha,incli, zmax, size, vel):

    m = np.tan(np.radians(90)+np.radians(incli))
    y0 = -D*np.sin(np.radians(alpha))/np.cos(np.radians(incli))
    n = -m*y0
    inter_pos = inter_los_par_pos(r0, theta, D,alpha,incli, zmax)
   # print(inter_pos)
    #inter_neg = inter_los_par_neg(r0, theta, D,alpha,incli, zmax)
    #print(inter_pos)

    if inter_pos:
        #print('pos',1)
        y2_pos = inter_pos[0][1]
        y1_pos = inter_pos[0][0]

        z2_pos = inter_pos[1][1]
        z1_pos = inter_pos[1][0]

        y_pos = np.array([y1_pos,y2_pos])
        z_pos = np.array([z1_pos,z2_pos])

        inds = y_pos.argsort()
        z1_pos, z2_pos = z_pos[inds]
        y1_pos, y2_pos = y_pos[inds]


        #print(y1_pos, y2_pos, z1_pos, z2_pos)
        mindis = np.sqrt(2*(size**2))/2
        #print(mindis)
        zgrid = np.arange(z2_pos + (size/2), z1_pos + (size/2), size)
        #print(z1_pos, z2_pos)
        #print(z1_pos + (size/2),z2_pos + (size/2) )
        #print(zgrid)
        ymin = int(y1_pos/size) * size + (size/2)
        ymax = int(y2_pos/size)*size +(size/2)
        #print(ymin, ymax)
        b=-1

        ygrid = np.arange(ymin,ymax,size)
        #print(ygrid)
        points = abs((m * (ygrid) + b * zgrid[:,None] + n)) / (np.sqrt(m * m + b * b))
        #print(points)
        selected = points <= mindis
        yv, zv = np.meshgrid(ygrid, zgrid)
        ypos_pos = np.asarray(yv[selected])
        zpos_pos = np.asarray(zv[selected])
        if len(zpos_pos) == 0:
            return(False)
        #print(zpos_pos)
        return(ypos_pos,zpos_pos)
    else:
        #print('pos',2)
        return(False)


def get_cells_neg(r0, theta, D,alpha,incli, zmax, size, vel):

    m = np.tan(np.radians(90)+np.radians(incli))
    y0 = -D*np.sin(np.radians(alpha))/np.cos(np.radians(incli))
    n = -m*y0
    inter_neg = inter_los_par_neg(r0, theta, D,alpha,incli, zmax)

    if inter_neg:
        #print('neg',1)
        y2_neg = inter_neg[0][0]
        y1_neg = inter_neg[0][1]

        z2_neg = inter_neg[1][0]
        z1_neg = inter_neg[1][1]

        y_neg = np.array([y1_neg,y2_neg])
        z_neg = np.array([z1_neg,z2_neg])

        inds = y_neg.argsort()
        z1_neg, z2_neg = z_neg[inds]
        y1_neg, y2_neg = y_neg[inds]

        #print('aaa', y1_neg, y2_neg, z1_neg, z2_neg)

        mindis = np.sqrt(2*(size**2))/2

        zgrid = np.arange(z2_neg + (size/2), z1_neg + (size/2), size)
        ymin = int(y1_neg/size) * size + (size/2)
        ymax = int(y2_neg/size)*size +(size/2)
        b=-1

        ygrid = np.arange(ymin,ymax,size)
        points = abs((m * (ygrid) + b * zgrid[:,None] + n)) / (np.sqrt(m * m + b * b))
        selected = points <= mindis
        yv, zv = np.meshgrid(ygrid, zgrid)
        ypos = np.asarray(yv[selected])
        zpos = np.asarray(zv[selected])
        return(ypos, zpos)
    else:
        #print('neg',2)
        return(False)



def get_cells(r0, theta,theta_min, D,alpha,incli, zmax, size, vel, neg_flow=False):

    yz_pos = get_cells_pos(r0, theta, D,alpha,incli, zmax, size, vel)
    yz_neg = get_cells_neg(r0, theta, D,alpha,incli, zmax, size, vel)
    #print('aaa', yz_pos)

    if neg_flow:

        if yz_pos and yz_neg:
        #print('cells',yz_pos[0],yz_pos[1] ,yz_neg[0],yz_neg[1])
            ypos_t = np.concatenate((yz_pos[0],yz_neg[0]))
            zpos_t = np.concatenate((yz_pos[1],yz_neg[1]))

        elif yz_pos:
            ypos_t = yz_pos[0]
            zpos_t = yz_pos[1]
            print('pos')

        elif yz_neg:
            print('neg')
            ypos_t = yz_neg[0]
            zpos_t = yz_neg[1]

        else:
            return(np.array([]), np.array([]), np.array([]), np.array([]))

    else:

        if yz_pos:
            print('yes')
            ypos_t = yz_pos[0]
            zpos_t = yz_pos[1]
        else:
            print('no')
            return(np.array([]), np.array([]), np.array([]), np.array([]))

    #print(ypos_t, zpos_t)

    x0 = D * np.cos(np.radians(alpha))
    a = np.tan(np.radians(theta))

    '''z0_max = r_max/a

    y_max = np.sqrt(a*a*(zpos_t + z0_max)*(zpos_t + z0_max) - x0*x0)

    z_max = np.sqrt(x0*x0/(a*a) + (ypos_t*ypos_t)/(a*a)) - z0_max
    z_max_2 = -np.sqrt(x0*x0/(a*a) + (ypos_t*ypos_t)/(a*a)) - z0_max
    print(z_max)'''

    '''cond_y = np.abs(ypos_t) >= np.abs(y_max)

    #cond_z = ypos_t>=0 & (zpos_t<=z_max)
    #cond_z_2 = ypos_t<0 & (zpos_t<=z_max_2)

    #cond_z_t = cond_z | cond_z_2

    ypos_t = ypos_t[cond_y]
    zpos_t = zpos_t[cond_y]'''

    x0 = D * np.cos(np.radians(alpha))
    a = np.tan(np.radians(theta))
    z0 = r0/a

    rho_t, theta_t, phi_t = coord.cartesian_to_spherical(x0, ypos_t, zpos_t)

    #print(theta_t)


    cond_theta = np.abs(theta_t.value) < (np.radians(90) - np.radians(theta_min))
    ypos_t = ypos_t[cond_theta]
    zpos_t = zpos_t[cond_theta]
   # print(y_pos, y_neg, z_pos, z_neg)
    #print(ypos_t, zpos_t)
    print(len(ypos_t))

    probs = prob(r0, ypos_t, zpos_t, theta, alpha, D)
    velos = out_los_vel(ypos_t, zpos_t, theta, D, alpha, incli, vel, r0)
    #print(velos)
    return(ypos_t,zpos_t,probs,velos)

from cgmspec.sampledist import RanDist
from scipy import stats
from scipy.special import wofz
from scipy.special import gamma


df = 6.71701 # parametro de forma.
chi2 = stats.chi2(df)
x = np.linspace(0,20)
fp = chi2.pdf(x) # Función de Probabilidad

bvals = np.linspace(3,20,100)
fNb = RanDist(bvals, chi2.pdf(bvals))

def ndist(n, beta = 1.5 ):
    return n**-beta

def get_clouds(ypos,zpos,probs,velos):
        randomnum = np.random.uniform(0, 100, len(probs))
        selected = probs >= randomnum
        return(velos[selected])

def losspec(lam,velos,X, N,b,z):
        #nvals = np.logspace(12.6, 16, 1000)
        #fN = RanDist(nvals, ndist(nvals))

        #Ns = fN.random(len(velos))
        #print(Ns)
       # N = np.empty([len(velos), 1])
       # for i in range(len(Ns)):
        #    N[i,0]=Ns[i]
        taus = Tau(lam,velos,X,N,b,z)
        tottau = np.sum(taus,axis=0)
        return(np.exp(-tottau))

def averagelos(r0, theta,theta_min, D,alpha,incli, zmax, size, vel, itera, X, N, z, lam, neg_flow=False):
        cells = get_cells(r0,theta, theta_min, D,alpha,incli, zmax, size, vel, neg_flow)
        #print(len(cells[0]))
        #List of 4 params:
        #cells= [ypos,zpos, probs, velos]

        results = [0]*itera

        results = [get_clouds(cells[0],cells[1],cells[2],cells[3]) for x in results]
        #list of len(iter) velocities satisfying prob_hit
        results = np.asarray(results)

        b = fNb.random(len(results[0]))
        #print(len(results))



        fluxes = [0]*itera
        fluxtoaver = [losspec(lam,results[x],X,N,b,z) for x in fluxes]
        fluxtoaver = np.asarray(fluxtoaver)
        totflux = np.median(fluxtoaver, axis=0)

        return(totflux, len(results[0]))

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

def eq_w(spec, vel, rang,z, w_pix):
    cond = np.abs(vel)<rang
    flux = spec[cond]
    a = (np.sum(1-flux))*w_pix
    #print(a)
    w = a/(1+z)
    return(w)

from astropy.convolution import convolve, Gaussian1DKernel
def filtrogauss(R, spec_res, lam_0, flux):
    del_lam = lam_0/R
    del_lam_pix = del_lam/spec_res
    gauss_kernel = (Gaussian1DKernel(del_lam_pix))
    gausflux = convolve(flux, gauss_kernel)
    return(gausflux)


lam1 = 6382
lam2 = 6425



lam = np.arange(lam1,lam2,0.05)
#(10, 20, 2,80,30, 200, 1, 200)
bouche_2012 = pd.read_csv('bouche_2012.txt', error_bad_lines=False, delim_whitespace=True)
bouche_2012.head
EW_bouche_2012 = bouche_2012['EW'].to_numpy()
D_bouche_2012 = bouche_2012['D'].to_numpy()

minor_tpcf = pd.read_csv('2edgemin.txt', delimiter='     ', engine='python')
minor_vel = minor_tpcf['vel'].to_numpy()
minor_tpcf_val = minor_tpcf['TPCF'].to_numpy()
minor_error = np.abs(minor_tpcf['minus_error'].to_numpy() - minor_tpcf['plus_error'].to_numpy())

vels_wave = (const.c.to('km/s').value * ((lam/ (2796.35 * (1 + 1.29))) - 1))

def get_sample(N,theta_max,theta_min,r_0,size,vel):
    ews_empty = []
    Ds_empty = []
    speci_empty = []
    nr_clouds = []
    alphas = np.random.uniform(low=-90, high=-45, size=(500,))
    ds = np.random.uniform(low=1, high=100, size=(500,))
    inclis = np.random.uniform(low=70, high=89, size=(500,))

    for i in range(len(alphas)):

        #(r0, theta, D,alpha,incli, zmax, size, vel, itera, X, N, b, z, lam)
        #(r0, theta, D,alpha,incli, zmax)
        #bla = inter_los_par_neg (10, 30, 30,5,5, 100)
        bla = averagelos(r_0,theta_max,theta_min, ds[i] ,alphas[i] ,inclis[i],200,size, vel, 1, 1, N,1.29, lam, neg_flow=False)
        bla1 = bla[0]
        nr = bla[1]
        ew = eq_w(bla1, vels_wave, 1000, 1.29, 0.05)
        ews_empty.append(ew)
        speci_empty.append(bla1)
        nr_clouds.append(nr)
        
    return(ews_empty,speci_empty,nr_clouds,ds)
    #Ds.append(result[0][i][j])
    #MyData[:,j,i] = bla

from itertools import combinations

def TPCF(speci_empty):
    cond = np.asarray(nr_clouds) == 0
    gauss_specs = []
    abs_specs = []
    vels_abs = []
    speci_empty_t = np.asarray(speci_empty)[~cond]
    print(len(speci_empty_t))
    for m in range(len(speci_empty_t)):
        print(m)
        gauss_specj = filtrogauss(45000,0.03,2796.35,speci_empty_t[m])
        gauss_specs.append(gauss_specj)
        zabs=1.29

        cond_abs1 = gauss_specj < 0.98
        cond_abs2 = np.abs(vels_wave) < 400
        abs_gauss_spec_major = vels_wave[cond_abs1 & cond_abs2]
        abs_specs.append(abs_gauss_spec_major)
    #vels_abs_major_i = [abs(i-j) for i in abs_gauss_spec_major for j in abs_gauss_spec_major if i != j]
    #vels_abs.append(vels_abs_major_i)

# Convert input list to a numpy array
    abs_specs_f = np.concatenate(np.asarray(abs_specs))
    bla = [abs(a -b) for a, b in combinations(abs_specs_f, 2)]
    bins1 = np.arange(0,np.max(bla),5)
    bla2 = np.histogram(bla,bins=bins1)
    return(bla2)


Ns = [10**12, 10**14, 10**16]
theta_maxs = 80
r_0 = 10
size = 0.1
vel = 200

EWs = []
specs = []
nr_clouds = []
ds_s = []

for i in range(len(Ns)):
    bla = get_sample(Ns[i],theta_maxs,0,r_0,size,vel)
    EWs.append(bla[0])
    specs.append(bla[1])
    nr_clouds.append(bla[2])
    ds_s.append(bla[3])

np.save('out_EW_2', EWs)
np.save('out_specs_2', specs)
np.save('out_nr_clouds_2', nr_clouds)
np.save('out_ds_2', ds_s)
