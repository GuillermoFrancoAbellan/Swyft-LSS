from montepython.likelihood_class import Likelihood
import os
import numpy  as np
import pandas as pd
import DMemu

from scipy.interpolate import interp1d
from scipy.integrate   import trapz

from astropy           import constants as const
from copy              import deepcopy

import warnings
warnings.filterwarnings('ignore')


class get_observables:

    def __init__(self,ni,cosmo,use_obs,ells,ddm):

        self.ni       = ni
        self.kernels  = {}
        self.use_obs  = use_obs
        self.Nbins    = len(self.ni)

        if ddm == True:
            self.emul = DMemu.TBDemu()

        if use_obs['GC']:
            self.kernels.update({'g'+str(i+1): self.gal_window(cosmo,i) for i in range(len(self.ni))})
        if use_obs['WL']:
            self.leff = [np.array([self.lens_eff(cosmo,z,i) for z in cosmo['z']]) for i in range(len(self.ni))]
            self.kernels.update({'d'+str(i+1): self.lens_window(cosmo,i) for i in range(len(self.ni))})
            self.kernels.update({'i'+str(i+1): self.IA_window(cosmo,i) for i in range(len(self.ni))})

        self.Cls = self.get_cls(cosmo,ells,ddm)

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

    def get_Pell(self,cosmo,ell,z, ddm):

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
        GCcols = ['G{}xG{}'.format(i,j) for i in range(1,self.Nbins+1) for j in range(i,self.Nbins+1)]
        XCcols = ['G{}xL{}'.format(i,j) for i in range(1,self.Nbins+1) for j in range(1,self.Nbins+1)]

        final_Cls = pd.DataFrame(columns=['ells']+WLcols+XCcols+GCcols,dtype=object)
        final_Cls['ells'] = ells

        if self.use_obs['WL']:
            for bin1 in range(1,self.Nbins+1):
                for bin2 in range(bin1,self.Nbins+1):
                    final_Cls['L{}xL{}'.format(bin1,bin2)] = Cls['d{}xd{}'.format(bin1,bin2)]+Cls['i{}xd{}'.format(bin1,bin2)]+Cls['i{}xi{}'.format(bin1,bin2)]

        if self.use_obs['GC']:
            for bin1 in range(1,self.Nbins+1):
                for bin2 in range(bin1,self.Nbins+1):
                    final_Cls['G{}xG{}'.format(bin1,bin2)] = Cls['g{}xg{}'.format(bin1,bin2)]


        if self.use_obs['WL'] and self.use_obs['GC']:
            for bin1 in range(1,self.Nbins+1):
                for bin2 in range(1,self.Nbins+1):
                    final_Cls['G{}xL{}'.format(bin1,bin2)] = Cls['g{}xd{}'.format(bin1,bin2)]+Cls['g{}xi{}'.format(bin1,bin2)]
        return final_Cls


class StageIV_3x2pt(Likelihood):

    def __init__(self, path, data, command_line):

        Likelihood.__init__(self, path, data, command_line)

        fid_file_path = os.path.join(self.data_directory, self.fiducial_file)
        dataset = np.load(fid_file_path,allow_pickle=True).item()

        self.datadict      = dataset['Cls']
        self.specs         = dataset['specs']
        self.covmats_dict  = dataset['covmat']
        self.ni            = dataset['nz']
        self.lumfunc       = dataset['lumfunc']

        self.invcov = {key: np.linalg.inv(mat.values) for key,mat in self.covmats_dict.items()}

        lmin = np.log10(self.specs['lmin'])
        lmax = np.log10(self.specs['lmax'])
        N    = self.specs['Nbin_ell']

        ell_lims    = np.logspace(lmin,lmax,N)
        self.ells   = 0.5*(ell_lims[:-1]+ell_lims[1:])
        self.deltas = (ell_lims[1:]-ell_lims[:-1])

        z_wmax = 4.0
        self.z_wsamp = 100
        self.z_win = np.logspace(np.log10(self.specs['zmin']),np.log10(z_wmax),self.z_wsamp)

        self.k_max_Boltzmann = self.specs['kmax']

        self.need_cosmo_arguments(data, {'z_max_pk': max(self.z_win)})
        self.need_cosmo_arguments(data, {'P_k_max_1/Mpc': self.k_max_Boltzmann})

        self.data_Cls = deepcopy(self.datadict['fiducial'])

    def loglkl(self, cosmo, data):

        cosmo_ba = cosmo.get_background()

        A_IA = data.mcmc_parameters['A_IA']['current']*data.mcmc_parameters['A_IA']['scale']
        eta_IA = data.mcmc_parameters['eta_IA']['current']*data.mcmc_parameters['eta_IA']['scale']
        beta_IA = 0.

        cosmo_dict = {'z': self.z_win,
                      'Omm': cosmo.Omega_m(),
                      'H_Mpc': interp1d(cosmo_ba['z'],cosmo_ba['H [1/Mpc]']),
                      'comov_dist': interp1d(cosmo_ba['z'],cosmo_ba['comov. dist.']),
                      'H0_Mpc': data.mcmc_parameters['H0']['current']*data.mcmc_parameters['H0']['scale']/const.c.to('km/s').value,
                      'Pk_delta': cosmo,
                      'kmax': self.k_max_Boltzmann}

        if self.specs['ddm'] == True:
            cosmo_dict.update({'log10_Gamma':data.mcmc_parameters['log10_Gamma']['current']*data.mcmc_parameters['log10_Gamma']['scale'],
                               'log10_vk':data.mcmc_parameters['log10_vk']['current']*data.mcmc_parameters['log10_vk']['scale'] })

        zmid = [0.20922009, 0.48837479, 0.61809265, 0.73208249, 0.84293749, 0.95803879, 1.08517926, 1.23723134, 1.44728054, 2.28646864]
        b1 = data.mcmc_parameters['b_1']['current']*data.mcmc_parameters['b_1']['scale']
        b10 = data.mcmc_parameters['b_10']['current']*data.mcmc_parameters['b_10']['scale']
        cosmo_dict.update({'bias': interp1d(zmid,[data.mcmc_parameters['b_'+str(i)]['current']*data.mcmc_parameters['b_'+str(i)]['scale'] for i in range(1,11)],bounds_error=False,fill_value=(b1,b10))})

        ks = 0.001
        P_z_k = []
        for zz in cosmo_dict['z']:
            P_z_k.append(cosmo.pk(ks,zz))
        cosmo_dict['Dz'] = np.sqrt(np.array(P_z_k)/cosmo.pk(ks,0.001))
        cosmo_dict['IA_term'] = interp1d(cosmo_dict['z'],-A_IA*0.0134*cosmo_dict['Omm']*(1+cosmo_dict['z'])**eta_IA/cosmo_dict['Dz']*self.lumfunc(cosmo_dict['z'])**beta_IA)

        obs   = get_observables(self.ni,
                                cosmo_dict,
                                self.specs['use_obs'],
                                self.ells,
                                self.specs['ddm'])

        loglike = 0
        for ind,ell in enumerate(self.ells):
            bin_combos = self.covmats_dict[str(int(ell))].columns
            diffvec = np.array([self.data_Cls[col][ind]-obs.Cls.iloc[ind][col] for col in bin_combos])
            loglike += -0.5*np.dot(diffvec,np.dot(self.invcov[str(int(ell))],diffvec))

        return loglike
