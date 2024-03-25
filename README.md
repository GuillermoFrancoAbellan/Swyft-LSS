# Swyft-LSS

This repository allows one to reproduce the results presented in "Fast likelihood-free inference in the LSS Stage IV era", [https://arxiv.org/abs/2403.14750](https://arxiv.org/abs/2403.14750). We mainly rely on **Swyft** ([https://github.com/undark-lab/swyft](https://github.com/undark-lab/swyft)) to perform parameter inference with Marginal Neural Ratio Estimation (**MNRE**), and the Boltzmann solver **CLASS** ([https://github.com/lesgourg/class_public](https://github.com/lesgourg/class_public)) to define our simulator of **Stage IV 3x2pt photometric probes**.

Additionally, the code relies on a few other packages:

- **PyTorch** ([https://pypi.org/project/torch/](https://pypi.org/project/torch/) and **PyTorch Lightning** ([https://pypi.org/project/pytorch-lightning/](https://pypi.org/project/pytorch-lightning/)). These are anyway needed for the Swyft installation.
- **Joblib** ([https://pypi.org/project/joblib/](https://pypi.org/project/joblib/)), in order to generate simulations in parallel.
- **DMemu** ([https://github.com/jbucko/DMemu](https://github.com/jbucko/DMemu)), in order to model the effects of dark matter decays on the non-linear matter power spectrum.
- **GetDist** ([https://pypi.org/project/getdist/](https://pypi.org/project/getdist/)), to post-process the MCMC.  
