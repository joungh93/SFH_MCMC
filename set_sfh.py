#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 10:32:19 2020

@author: jlee
"""


import numpy as np
import glob, os, copy
from astropy.io import fits
import time
from scipy.optimize import minimize
from scipy.stats import truncnorm
import emcee
from multiprocessing import Pool


# ----- Class & module ----- #
class set_sfh:


	def __init__(self, obs_name, model_list, lower_limit=1.0e-4, upper_limit=1.0e+4):

		'''
		obs_name:
		The name of the observed CMD (dtype = 'str')

		model_list:
		A list of names of model CMDs. (dtype = 'str')

		lower_limit:
		The lower limit of the weight parameter

		upper_limit:
		The upper limit of the weight parameter
		'''

		# The observed CMD
		self.obs = obs_name
		obs_cmd = fits.getdata(obs_name, header=False)
		Y = obs_cmd.copy()
		Y_sum = np.sum(Y)
		S = np.sqrt(Y)
		S[S==0] = 1.    # not this, but just placeholder
		# S = np.sqrt(S**2 + detection_error**2 + minimal_error**2)
		fac = 1./Y_sum    # normalization factor
		self.Y = Y*fac
		self.S = S*fac
		Y_shape = Y.shape

		# Model CMDs
		self.mod = model_list
		self.n_model = len(model_list)
		Xm = np.zeros((self.n_model, Y_shape[0], Y_shape[1]))
		for k in np.arange(self.n_model):
			mod_cmd = fits.getdata(model_list[k], header=False)
			Xm[k,:,:] = mod_cmd*fac
		self.Xm = Xm

		# The range of weight parameters
		self.llim = lower_limit
		self.ulim = upper_limit


	def model_linpar(self, *theta):
		x = self.Xm
		t = np.array(theta)
		t = np.minimum(self.ulim, np.maximum(self.llim, t))
		tx = np.multiply(t, x.T).T

		return np.sum(tx, axis=0)


	def model_logpar(self, *theta):
	    t = 10.**(np.array(theta))
	    return self.model_linpar(*t)


	def log_likelihood(self, model, y, yerr, *theta):
	    nll = 0.5*np.sum(((y-model(*theta))/yerr)**2. + 2.*np.log(yerr)) + 0.5*len(y)*np.log(2.*np.pi)
	    return -nll


	def log_prior(self, *theta):
	    if (np.any((np.array(theta) < self.llim) | (np.array(theta) > self.ulim)) == True):
	        return -np.inf
	    else:
	        return 0.


	def log_posterior(self, model, y, yerr, *theta):
	    return self.log_prior(*theta) + self.log_likelihood(model, y, yerr, *theta)


	def mcmc_walk(self, nwalkers=64, nsteps=5000, sigma=5.0-4):

		# Applying MLE method for initial guess
		nll = lambda x: -self.log_likelihood(self.model_logpar, self.Y, self.S, *x)
		initial = [0]*self.n_model
		soln = minimize(nll, initial)

		# Running MCMC
		log_prob = lambda x: self.log_posterior(self.model_linpar, self.Y, self.S, *x)

		ndim = self.n_model
		init = []
		for ix in soln.x:
			if ((ix > np.log10(self.llim)) & (ix < np.log10(self.ulim))):
				value = 10.0**ix
			elif (ix <= np.log10(self.llim)):
				value = self.llim
			elif (ix >= np.log10(self.ulim)):
				value = self.ulim
			init.append(value)

		pos = np.zeros((nwalkers, ndim))
		for i in np.arange(ndim):
			pos[:, i] = ((lambda m, s, l, u: truncnorm((l-m)/s, (u-m)/s, loc=m, scale=s))
                         (init[i], sigma, self.llim, self.ulim)).rvs(nwalkers)

		# with Pool() as pool:
		sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)#, self.log_prob, pool=pool)
		start_time = time.time()
		sampler.run_mcmc(pos, nsteps, progress=True)
		end_time = time.time()
		print("----- running time : {0:.3f} sec".format(end_time-start_time))

		samples = sampler.get_chain()
		np.savez_compressed('mcmc_samples.npz', samp=samples,
			                obs_name=self.obs, mod_name=self.mod,
			                obs_data=self.Y, mod_data=self.Xm)

		# flat_samples = sampler.get_chain(discard=ndiscard, flat=True)

		return sampler



# ----- Main program ----- #
if (__name__ == '__main__'):

	# Structures
	current_dir = os.getcwd()
	dir_SFH = '/data/jlee/DATA/Tests/test_python/SFH_isjang/N300_SFH/1_N300_out/'
	dir_obs = dir_SFH+'obs/'
	dir_mod = dir_SFH+'models/'

	# Loading data
	obs_cmd = dir_obs+'F456.fits'
	# model_list = glob.glob(dir_mod+'a*fehm1.60.fits')
	# model_list = sorted(model_list)
	model_list = [dir_mod+'a08.00fehm1.60.fits',
	              dir_mod+'a08.20fehm1.60.fits',
	              dir_mod+'a08.40fehm1.60.fits', 
	              dir_mod+'a08.60fehm1.60.fits',
	              dir_mod+'a08.80fehm1.60.fits',
	              dir_mod+'a09.00fehm1.60.fits',
	              dir_mod+'a09.20fehm1.60.fits',
	              dir_mod+'a09.40fehm1.60.fits',
	              dir_mod+'a09.60fehm1.60.fits',
	              dir_mod+'a09.80fehm1.60.fits',
	              dir_mod+'a10.00fehm1.60.fits']

	# MCMC walker
	s = set_sfh(obs_cmd, model_list, lower_limit=1.0e-5, upper_limit=1.0e+2)
	sampler = s.mcmc_walk(nwalkers=128, nsteps=10000, sigma=5.0e-4)
	# sampler = s.mcmc_walk(nwalkers=64, nsteps=5000, sigma=1.0e-3)

