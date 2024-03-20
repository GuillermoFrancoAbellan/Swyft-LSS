import numpy  as np
import pandas as pd
import DMemu

from scipy.interpolate import interp1d
from scipy.integrate   import trapz

from itertools import product
from time      import time


import warnings
warnings.filterwarnings('ignore')

class get_obs:

    def __init__(self,ni,cosmo,use_obs,ells,feedback,ddm):

        self.ni       = ni
        self.kernels  = {}
        self.use_obs  = use_obs
        self.Nbins    = len(self.ni)

        if ddm == True:
            self.emul = DMemu.TBDemu()

        tini = time()
        if use_obs['GC']:
            self.kernels.update({'g'+str(i+1): self.gal_window(cosmo,i) for i in range(len(self.ni))})
        if use_obs['WL']:
            self.leff = [np.array([self.lens_eff(cosmo,z,i) for z in cosmo['z']]) for i in range(len(self.ni))]
            self.kernels.update({'d'+str(i+1): self.lens_window(cosmo,i) for i in range(len(self.ni))})
            self.kernels.update({'i'+str(i+1): self.IA_window(cosmo,i) for i in range(len(self.ni))})
        tend = time()
        if feedback == True:
            print('Kernels computed in {:.1f} s'.format(tend-tini))

        tini = time()
        self.Cls = self.get_cls(cosmo,ells,ddm)
        self.sigma8 = cosmo['sig8']
        self.Omega_m = cosmo['Omm']
        tend = time()
        if feedback:
            print('Cls computed in {:.1f} s'.format(tend-tini))

    def gal_window(self,cosmo,i):

        Wgal = np.array([self.ni[i](z)*cosmo['H_Mpc'](z) for z in cosmo['z']])

        return Wgal

    def lens_window(self,cosmo,i):

        Wlens = np.array([(3/2)*cosmo['H0_Mpc']**2.*cosmo['Omm']*(1+z)*cosmo['comov_dist'](z)*self.leff[i][ind] for ind,z in enumerate(cosmo['z'])])

        return Wlens

    def IA_window(self,cosmo,i):

        WIA = np.array([self.ni[i](z)*cosmo['H_Mpc'](z) for z in cosmo['z']])

        return WIA

    def lens_eff(self,cosmo,z,i):

        zp = np.linspace(z,cosmo['z'][-1],100)
        leff = trapz(self.ni[i](zp)*(1-cosmo['comov_dist'](z)/cosmo['comov_dist'](zp)),x=zp)

        return leff

    def get_Pell(self,cosmo,ell,z,ddm):

        kappa = (ell+0.5)/cosmo['comov_dist'](z)

        if kappa > cosmo['kmax']:
            pk_m = 0.
        else:
            if ddm == True:
                if z > 2.35:
                    ddm_suppression = 1. # we cannot extrapolate the emulator beyond the training domain for z
                    # Extrapolation for kappa > 6 h/Mpc is done by adding a constant suppression continuously attached
                    # to the one provided by an emulator
                else:
                    velocity_kick = 10**(cosmo['log10_vk']) # in km/s, has to be in range (0,5000) km/s
                    gamma_decay = 10**(cosmo['log10_Gamma']) # in 1/Gyr, has to be in range (0,1/13.5) Gyr^-1
                    fraction = 1.0
                    ddm_suppression=self.emul.predict(np.asarray(kappa).reshape(-1),np.asarray(z).reshape(-1),fraction,velocity_kick,gamma_decay)
            else:
                ddm_suppression = 1.
            pk_m = cosmo['Pk_delta'].pk(kappa,z)*ddm_suppression

        Pell = {'Pgg': cosmo['bias'](z)**2.*pk_m,
                'Pgi': cosmo['bias'](z)*cosmo['IA_term'](z)*pk_m,
                'Pig': cosmo['bias'](z)*cosmo['IA_term'](z)*pk_m,
                'Pii': cosmo['IA_term'](z)**2*pk_m}
        Pell.update({'Pdd': pk_m,
                     'Pdi': cosmo['IA_term'](z)*pk_m,
                     'Pid': cosmo['IA_term'](z)*pk_m,
                     'Pdg': cosmo['bias'](z)*pk_m,
                     'Pgd': cosmo['bias'](z)*pk_m})

        return Pell

    def get_cls(self,cosmo,ells,ddm):

        integrand = np.array([1/(cosmo['H_Mpc'](z)*cosmo['comov_dist'](z)**2) for z in cosmo['z']])

        pspectra = np.array([[self.get_Pell(cosmo,ell,z,ddm) for z in cosmo['z']] for ell in ells])

        Cls = {}

        def mysplit(s):
            head = s.rstrip('0123456789')
            tail = s[len(head):]
            return head, tail

        zint = cosmo['z']
        for n1,w1 in self.kernels.items():
            for n2,w2 in self.kernels.items():
                obs1,bin1 = mysplit(n1)
                obs2,bin2 = mysplit(n2)
                Cls.update({n1+'x'+n2: np.array([trapz([w1[zind]*w2[zind]*integrand[zind]*pspectra[ellind,zind]['P'+obs1+obs2]
                                                        for zind,z in enumerate(zint)],x=zint)
                                                 for ellind,ell in enumerate(ells)])})


        #Here build the final Cls
        WLcols = ['L{}xL{}'.format(i,j) for i in range(1,self.Nbins+1) for j in range(i,self.Nbins+1)]
        XCcols = ['G{}xL{}'.format(i,j) for i in range(1,self.Nbins+1) for j in range(1,self.Nbins+1)]
        GCcols = ['G{}xG{}'.format(i,j) for i in range(1,self.Nbins+1) for j in range(i,self.Nbins+1)]

        final_Cls = pd.DataFrame(columns=['ells']+WLcols+XCcols+GCcols,dtype=object)
        final_Cls['ells'] = ells

        if self.use_obs['GC']:
            for bin1 in range(1,self.Nbins+1):
                for bin2 in range(bin1,self.Nbins+1):
                    final_Cls['G{}xG{}'.format(bin1,bin2)] = Cls['g{}xg{}'.format(bin1,bin2)]

        if self.use_obs['WL']:
            for bin1 in range(1,self.Nbins+1):
                for bin2 in range(bin1,self.Nbins+1):
                    final_Cls['L{}xL{}'.format(bin1,bin2)] = Cls['d{}xd{}'.format(bin1,bin2)]+Cls['i{}xd{}'.format(bin1,bin2)]+Cls['i{}xi{}'.format(bin1,bin2)]

        if self.use_obs['WL'] and self.use_obs['GC']:
            for bin1 in range(1,self.Nbins+1):
                for bin2 in range(1,self.Nbins+1):
                    final_Cls['G{}xL{}'.format(bin1,bin2)] = Cls['g{}xd{}'.format(bin1,bin2)]+Cls['g{}xi{}'.format(bin1,bin2)]

        return final_Cls
