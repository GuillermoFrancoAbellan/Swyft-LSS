import re
import numpy  as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

def covariance_einsum(cl_5d, noise_5d, f_sky, ell_values, delta_ell, return_only_diagonal_ells=False):
    """
    computes the 10-dimensional covariance matrix, of shape
    (n_probes, n_probes, n_probes, n_probes, nbl, (nbl), zbins, zbins, zbins, zbins). The 5-th axis is added only if
    return_only_diagonal_ells is True. *for the single-probe case, n_probes = 1*

    In np.einsum, the indices have the following meaning:
        A, B, C, D = probe identifier. 0 for WL, 1 for GCph
        L, M = ell, ell_prime
        i, j, k, l = redshift bin indices

    cl_5d must have shape = (n_probes, n_probes, nbl, zbins, zbins) = (A, B, L, i, j), same as noise_5d

    :param cl_5d:
    :param noise_5d:
    :param f_sky:
    :param ell_values:
    :param delta_ell:
    :param return_only_diagonal_ells:
    :return: 10-dimensional numpy array of shape
    (n_probes, n_probes, n_probes, n_probes, nbl, (nbl), zbins, zbins, zbins, zbins), containing the covariance.

    """
    assert cl_5d.shape[0] == cl_5d.shape[1], 'cl_5d must be an array of shape (n_probes, n_probes, nbl, zbins, zbins)'
    assert cl_5d.shape[-1] == cl_5d.shape[-2], 'cl_5d must be an array of shape (n_probes, n_probes, nbl, zbins, zbins)'
    assert noise_5d.shape == cl_5d.shape, 'noise_5d must have shape the same shape as cl_5d, although there ' \
                                          'is no ell dependence'

    nbl = cl_5d.shape[2]

    prefactor = 1 / ((2 * ell_values + 1) * f_sky * delta_ell)

    # considering ells off-diagonal (wrong for Gauss: I am not implementing the delta)
    # term_1 = np.einsum('ACLik, BDMjl -> ABCDLMijkl', cl_5d + noise_5d, cl_5d + noise_5d)
    # term_2 = np.einsum('ADLil, BCMjk -> ABCDLMijkl', cl_5d + noise_5d, cl_5d + noise_5d)
    # cov_10d = np.einsum('ABCDLMijkl, L -> ABCDLMijkl', term_1 + term_2, prefactor)

    # considering only ell diagonal
    term_1 = np.einsum('ACLik, BDLjl -> ABCDLijkl', cl_5d + noise_5d, cl_5d + noise_5d)
    term_2 = np.einsum('ADLil, BCLjk -> ABCDLijkl', cl_5d + noise_5d, cl_5d + noise_5d)
    cov_9d = np.einsum('ABCDLijkl, L -> ABCDLijkl', term_1 + term_2, prefactor)

    if return_only_diagonal_ells:
        warnings.warn('return_only_diagonal_ells is True, the array will be 9-dimensional, potentially causing '
                      'problems when reshaping or summing to cov_SSC arrays')
        return cov_9d

    n_probes = cov_9d.shape[0]
    zbins = cov_9d.shape[-1]
    cov_10d = np.zeros((n_probes, n_probes, n_probes, n_probes, nbl, nbl, zbins, zbins, zbins, zbins))
    cov_10d[:, :, :, :, np.arange(nbl), np.arange(nbl), ...] = cov_9d[:, :, :, :, np.arange(nbl), ...]

    return cov_10d

def split_num(s):
    head = s.rstrip('0123456789')
    tail = s[len(head):]
    return head, tail

def get_covmat(Cls,specs,covmat=None):

    Nbins   = specs['Nbins']
    ells    = Cls['ells'].values
    ngalbin = specs['ngalbin']

    Nell = {k: [0.]*len(ells) for k in Cls.columns}
    for i in range(1,Nbins+1):
        Nell['G{}xG{}'.format(i,i)] = [(1/ngalbin)]*len(ells)
        Nell['L{}xL{}'.format(i,i)] = [(specs['sigma_eps']**2/ngalbin)]*len(ells)

    fsky      = specs['fsky']
    Delta_ell = specs['deltas']

    WLcols = ['L{}xL{}'.format(i,j) for i in range(1,Nbins+1) for j in range(i,Nbins+1)]
    GCcols = ['G{}xG{}'.format(i,j) for i in range(1,Nbins+1) for j in range(i,Nbins+1)]
    XCcols = ['G{}xL{}'.format(i,j) for i in range(1,Nbins+1) for j in range(1,Nbins+1)]

    all_cols = WLcols+XCcols+GCcols

    if not covmat is None:
        return covmat,all_cols,Nell

    err_for_cov = np.zeros((2,2,len(ells),Nbins,Nbins))
    cls_for_cov = np.zeros((2,2,len(ells),Nbins,Nbins))

    for o1,obs1 in enumerate(['G','L']):
        for o2,obs2 in enumerate(['G','L']):
            for i in range(Nbins):
                for j in range(Nbins):
                    for ell_ind,ell in enumerate(ells):
                        if obs1 == obs2 and j<i:
                            Nell[obs1+str(i+1)+'x'+obs2+str(j+1)] = Nell[obs1+str(j+1)+'x'+obs2+str(i+1)]
                            Cls[obs1+str(i+1)+'x'+obs2+str(j+1)]  = Cls[obs1+str(j+1)+'x'+obs2+str(i+1)]
                        if obs1 != obs2: # GFA: extra lines added here, we use that GixLj = LjxGi
                            Nell[obs2+str(i+1)+'x'+obs1+str(j+1)] = Nell[obs1+str(j+1)+'x'+obs2+str(i+1)]
                            Cls[obs2+str(i+1)+'x'+obs1+str(j+1)]  = Cls[obs1+str(j+1)+'x'+obs2+str(i+1)]

                        err_for_cov[o1,o2,ell_ind,i,j] = Nell[obs1+str(i+1)+'x'+obs2+str(j+1)][ell_ind]
                        cls_for_cov[o1,o2,ell_ind,i,j] = Cls[obs1+str(i+1)+'x'+obs2+str(j+1)][ell_ind]

    raw_covmat = covariance_einsum(cls_for_cov,err_for_cov,fsky,ells,Delta_ell,return_only_diagonal_ells=True)

    zpair_auto = Nbins*(Nbins+1)//2
    zpair_tot  = 2*zpair_auto+Nbins**2

    def str_to_ind(obs):

        if obs == 'G':
            ind = 0
        elif obs == 'L':
            ind = 1

        return ind


    covmat_dict = {}

    for ellind,ell in enumerate(ells):

        packed_covmat = np.zeros((len(all_cols),len(all_cols)))

        for ind1,col in enumerate(all_cols):
            bin1,bin2 = re.split('x',col)
            oi1,i1 = split_num(bin1)
            oj1,j1 = split_num(bin2)

            for ind2,row in enumerate(all_cols):
                bin1,bin2 = re.split('x',row)
                oi2,i2 = split_num(bin1)
                oj2,j2 = split_num(bin2)

                packed_covmat[ind1,ind2] = raw_covmat[str_to_ind(oi1),str_to_ind(oj1),str_to_ind(oi2),str_to_ind(oj2),
                                                      ellind,int(i1)-1,int(j1)-1,int(i2)-1,int(j2)-1]

        covmat_dict[str(int(ell))] = pd.DataFrame(packed_covmat,columns=all_cols,index=all_cols)


    return covmat_dict,all_cols,Nell
