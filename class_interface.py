import numpy as np
from astropy import constants as const

from compute_obs      import get_obs
from covariance_utils import get_covmat

from copy              import deepcopy
from classy import Class
from scipy.interpolate import interp1d
from scipy.integrate   import trapz
from itertools         import product

import pandas as pd
from time      import time

class Main:

    def __init__(self,specs,survey):

        if specs['ddm'] == True:
            self.nuisance_list = ['A_IA','eta_IA','beta_IA','b_1','b_2','b_3','b_4','b_5','b_6','b_7','b_8','b_9','b_10','log10_Gamma','log10_vk']
        else:
            self.nuisance_list = ['A_IA','eta_IA','beta_IA','b_1','b_2','b_3','b_4','b_5','b_6','b_7','b_8','b_9','b_10']

        galdist = pd.read_csv(survey['nz'],skiprows=1,header=None,sep='\s+')
        if 'ISTF' in survey['nz']:
            Nbins = 10
        elif 'SPV3' in survey['nz']:
            Nbins = 13
        galdist.columns = ['z']+['n{}'.format(ind) for ind in range(1,Nbins+1)]

        self.ni = [interp1d(galdist['z'],galdist[col]/trapz(galdist[col],x=galdist['z']),fill_value=(0,0),bounds_error=False) for col in galdist.columns if col != 'z']

        luminosity = pd.read_csv(survey['luminosity'],skiprows=1,header=None,sep='\s+')
        luminosity.columns = ['z','lum']
        self.lumfunc = interp1d(luminosity['z'],luminosity['lum'],fill_value='extrapolate')

        self.specs = deepcopy(specs)

        self.specs['Nbins']  = len(galdist.columns)-1

        self.specs['ngalbin'] = (specs['gal_per_arcmin']/self.specs['Nbins'])*3600*(180/np.pi)**2

        lmin = np.log10(specs['lmin'])
        lmax = np.log10(specs['lmax'])
        N    = specs['Nbin_ell']

        ell_lims    = np.logspace(lmin,lmax,N)
        self.ells   = 0.5*(ell_lims[:-1]+ell_lims[1:])
        self.deltas = (ell_lims[1:]-ell_lims[:-1])

        self.specs['deltas'] = self.deltas
        z_wmax = 4.0
        self.z_wsamp = 100
        self.z_win = np.logspace(np.log10(specs['zmin']),np.log10(z_wmax),self.z_wsamp)

        self.k_max_Boltzmann = specs['kmax']

    def get_cosmo_dict(self,params):

        cosmo_pars = {par:val for par,val in params.items() if par not in self.nuisance_list}

        M = Class()
        M.set(cosmo_pars)
        M.set({'z_max_pk':max(self.z_win),'P_k_max_1/Mpc':self.k_max_Boltzmann})
        tini = time()
        M.compute()
        tend = time()
        if self.specs['feedback'] == True:
            print('CLASS calculation done in {:.1f} s'.format(tend-tini))
        Mba = M.get_background()

        cosmo_dict = {'z': self.z_win,
                      'Omm': M.Omega_m(),
		      'sig8': M.sigma8(),
                      'H_Mpc': interp1d(Mba['z'],Mba['H [1/Mpc]']),
                      'comov_dist': interp1d(Mba['z'],Mba['comov. dist.']),
                      'H0_Mpc': params['H0']/const.c.to('km/s').value,
                      'Pk_delta': M,
                      'kmax': self.k_max_Boltzmann}

        if self.specs['ddm'] == True:
            cosmo_dict.update({'log10_Gamma':params['log10_Gamma'],'log10_vk':params['log10_vk']})

        zmid = [0.20922009, 0.48837479, 0.61809265, 0.73208249, 0.84293749, 0.95803879, 1.08517926, 1.23723134, 1.44728054, 2.28646864]
        cosmo_dict.update({'bias': interp1d(zmid,[params['b_'+str(i)] for i in range(1,11)],bounds_error=False,fill_value=(params['b_1'],params['b_10']))})

        ks = 0.001
        P_z_k = []
        for zz in cosmo_dict['z']:
            P_z_k.append(M.pk(ks,zz))
        cosmo_dict['Dz'] = np.sqrt(np.array(P_z_k)/M.pk(ks,0.001))
        cosmo_dict['IA_term'] = interp1d(cosmo_dict['z'],-params['A_IA']*0.0134*cosmo_dict['Omm']*(1+cosmo_dict['z'])**params['eta_IA']/cosmo_dict['Dz']*self.lumfunc(cosmo_dict['z'])**params['beta_IA'])

        return cosmo_dict


    def get_calculations(self,params):

        cosmo_dict = self.get_cosmo_dict(params)
        calc_obs = get_obs(self.ni,cosmo_dict,
                           self.specs['use_obs'],
                           self.ells,
                           self.specs['feedback'],
                           self.specs['ddm'])

        return calc_obs


    def get_noisy_Cls(self,Cls,covmat=None):

        tini = time()
        covmat_dict,all_cols,Nell = get_covmat(Cls,self.specs,covmat=covmat)
        tend = time()
        if self.specs['feedback'] == True:
            print('Covmat computed in {:.1f} s'.format(tend-tini))

        Cls = Cls.drop([col for col in Cls.columns if col not in all_cols],axis=1)
        Cls = Cls[all_cols]

        fact   = 1/((2*self.ells+1)*self.deltas*self.specs['fsky'])

        noisy_cls = {'ell': self.ells,
                     'delta_ell': self.deltas,
                     'fiducial': Cls.to_dict('series'),
                     'errs': {win: np.array([np.sqrt(covmat.loc[win][win]) for covmat in covmat_dict.values()]) for win in all_cols}}

        noisy_cls['CV']    = {key: np.sqrt(2*fact*(np.array(val))**2) for key,val in noisy_cls['fiducial'].items()}
        noisy_cls['noise'] = {key: np.sqrt(2*fact*(np.array(val))**2) for key,val in Nell.items()}
        noisy_cls['vals']  = self.get_realization(noisy_cls['fiducial'],covmat_dict,all_cols)

        return covmat_dict,noisy_cls

    def get_realization(self,fiducial,covmats,cols):

        tini = time()
        ell_realization = []

        for ind,ell in enumerate(self.ells):
            covmat = covmats[str(int(ell))]
            if fiducial is None:
                means  = np.zeros(len(covmat.values))
            else:
                means  = [fiducial[col][ind] for col in cols]
            sample = np.random.multivariate_normal(means,covmat.values)
            ell_realization.append({key: sample[ind] for ind,key in enumerate(cols)})


        realization = {key: np.array([d[key] for d in ell_realization]) for key in ell_realization[0].keys()}
        tend = time()
        if self.specs['feedback'] == True:
            print('noisy realization around fiducial C_ell computed in {:.1f} s'.format(tend-tini))

        return realization
