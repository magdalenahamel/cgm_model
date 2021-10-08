
import numpy as np
from cgmspec import utils as csu
from matplotlib import pyplot as plt
from matplotlib import patches
from astropy import constants as const
from matplotlib.pyplot import cm
from timeit import default_timer as timer
from cgmspec.sampledist import RanDist



"""This Class Disco represents the CGM of a galaxy from a disc model"""

#logN = np.arange(12, 16, 0.1)
#logN_PDF = logN**(-0.4)
#logN_dist = RanDist(logN, logN_PDF)

class Disco:
    """Represents the CGM of a idealized disc projected in the sky"""

    def __init__(self,h, incl, Rcore=0.1):
        """
        :param h: float, height of disc in kpc
        :param incl: float, inclination angle of disc in degrees
        :param Rcore: float, radius of disk core in kpc, where the probability is maximum
        """

        self.Rcore = Rcore
        self.h = h
        self.incl = incl
        self.incl_rad = np.radians(self.incl)

    def get_cells(self,D,alpha,size,r_0, vR,hv):


        h = self.h
        incli = self.incl

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
        probs = self.prob_hit(radios,r_0)
        velos = self.los_vel(ypos, D, alpha, vR, hv)
        return(ypos,zpos, probs, velos)



    def prob_hit(self, r, r_0, prob_rmin=100):
        """
        Probability of hitting a cloud at distance r in the plane xy of a disc of radius rmax. For the moment is a power law.

        :param r: np.array, array with distances to the center of disc in plane xy in kpc
        :param r_0: float, the characteristicradius of the power law
        :param prob_rmin: float, probability at Rcore or below, default 100% probability of crossing a cloud
        :return: float, probability of hitting a cloud
        """
        rmin = self.Rcore
        ind = np.log(prob_rmin)/(np.log(rmin)-np.log(r_0))
        A = r/r_0
        prob = A**(ind)
        prob = np.where(r<=rmin, prob_rmin, prob)
        return prob

    '''def prob_hit(self, r, r_0):
        """
        :param r: float, distance to the center of disc in plane xy in kpc
        :param r_0: float, characteristic
        :return: float, covering fraction at distance r
        """

        return csu.prob_hit(r, self.Rcore, r_0)'''

    def los_vel(self, y, D, alpha, vR, hv, v_inf=0):
        """
        line of sight velocity of a cloud in a disc. See Ho et al. 2017

        :param y: np.array, distance of the line-of-sight to the semi-major axis (x-axis) along the y-axis
        :param D: float, impact parameter in kpc
        :param alpha: float, angle between the major axis and the line-of-sight, clockwise, in degrees
        :param vR: maximum velocity of rotation of the disk in km/s
        :param hv: velocity scale height in kpc
        :param v_inf: the infall velocity, for the moment i'm not using it.
        :return: line-of-sight velocity for a cloud in the given position

        """
        v_los_inf = (v_inf * np.sin(self.incl_rad)) * (y/(np.sqrt((y**2) + D**2)))
        al_rad = np.radians(alpha)

        R = D * np.sqrt(1+(np.sin(al_rad)**2)*np.tan(self.incl_rad)**2)
        vrot = (2/np.pi)*np.arctan2(R,1)

        x0 = D * np.cos(al_rad)  # this is p in Ho et al. 2019, fig 10.
        #print('al,x0',al_rad,x0)
        y0 = D * np.sin(al_rad) / np.cos(self.incl_rad)  # this is y0 in the same fig.
        if x0>=0:
              a = np.sin(self.incl_rad) / np.sqrt(1 + (y/x0)**2)
        else:
            a = -np.sin(self.incl_rad) / np.sqrt(1 + (y/x0)**2)

        b = np.exp(-np.fabs(y - y0) / hv * np.tan(self.incl_rad))
        #print(b)
        vr = (vR*vrot*a*b) + v_los_inf

        #print('vel', vr)
        return(vr)




    def get_clouds(self,ypos,zpos,probs,velos):
        randomnum = np.random.uniform(0, 100, len(probs))
        selected = probs >= randomnum
        return(velos[selected])



    def losspec(self,lam,velos,X, N,b,z):
        taus = csu.Tau(lam,velos,X,N,b,z)
        tottau = np.sum(taus,axis=0)
        return(np.exp(-tottau))




    def averagelos(self, D, alpha, lam, iter,X, z, grid_size, N, b, r_0, v_max, h_v, v_inf):
        h = self.h
        incli = self.incl

        cells = self.get_cells(D,alpha,grid_size, r_0,v_max,h_v)
        #List of 4 params:
        #cells= [ypos,zpos, probs, velos]

        results = [0]*iter

        results = [self.get_clouds(cells[0],cells[1],cells[2],cells[3]) for x in results]
        #list of len(iter) velocities satisfying prob_hit
        results = np.asarray(results)



        fluxes = [0]*iter
        fluxtoaver = [self.losspec(lam,results[x],X,N,b,z) for x in fluxes]
        fluxtoaver = np.asarray(fluxtoaver)
        totflux = np.median(fluxtoaver, axis=0)

        return(totflux)
