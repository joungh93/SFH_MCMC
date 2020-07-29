#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 10:28:31 2020

@author: jlee
"""


import time
start_time = time.time()

import numpy as np
import glob, os, copy
from astropy.io import fits
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import skew
from set_sfh import set_sfh


# ----- Loading data ----- #
d = np.load('mcmc_samples.npz')
# samp, obs, mod
n1, n2, n3 = d['samp'].shape
samp_2D = d['samp'].reshape((n1*n2, n3))

age, feh = [], []
for i in np.arange(len(d['mod_name'])):
	age.append(d['mod_name'][i].split('/')[-1].split('fehm')[0].split('a')[1])
	feh.append('-'+d['mod_name'][i].split('/')[-1].split('fehm')[1].split('.fits')[0])
age = np.array(age).astype('float')
feh = np.array(feh).astype('float')


# ----- Trace plot ----- #
ndiscard0 = 1000

with PdfPages('trace1.pdf') as pdf:
	n_fig, n_row = 1, 10
	plot_scale = 100

	for k in np.arange(len(d['mod_name'])):
		if (k % n_row == 0):
			fig = plt.figure(n_fig, figsize=(10,10))
			gs = GridSpec(n_row, 1, left=0.10, bottom=0.10, right=0.85, top=0.975,
                          height_ratios=[1]*n_row, hspace=0.05)
			n_fig += 1

		ax = fig.add_subplot(gs[k % n_row, 0])
		Y = d['samp'][ndiscard0:, :, k][::plot_scale, :]
		X = np.linspace(ndiscard0, len(d['samp']), Y.shape[0]+1).astype('int')[1:]
		yl, yh = np.percentile(Y, [0.27, 99.73])
		ax.set_ylim(yl-np.std(Y), yh+np.std(Y))
		ax.tick_params(axis='both', labelsize=12.0)
		ax.plot(X, Y, '-', color='k', alpha=0.3)
		ax.text(1.01, 0.95, f"Age = {age[k]:.1f} Gyr", fontsize=12.0,
			    ha='left', va='top', transform=ax.transAxes)
		ax.text(1.01, 0.70, f"[Fe/H] = {feh[k]:.2f}", fontsize=12.0,
			    ha='left', va='top', transform=ax.transAxes)

		if ((k % n_row == n_row-1) | (k == np.arange(len(d['mod_name']))[-1])):
			ax.set_xlabel('step number', fontsize=12.0)
		else:
			ax.tick_params(labelbottom=False)
		
		if (k % n_row == n_row-1):
			pdf.savefig()
			plt.close()

	pdf.savefig()
	plt.close()


# ----- Histogram plot ----- #
ndiscard1 = 5000
skewness = []
weight_par, weight_err = [], []

with PdfPages('hist1.pdf') as pdf:
	n_fig, n_row, n_col = 1, 4, 4

	for k in np.arange(len(d['mod_name'])):
		if (k % (n_row*n_col) == 0):
			fig = plt.figure(n_fig, figsize=(10,10))
			gs = GridSpec(n_row, n_col, left=0.05, bottom=0.05, right=0.975, top=0.975,
                          height_ratios=[1]*n_row, width_ratios=[1]*n_col,
                          hspace=0.15, wspace=0.05)
			n_fig += 1

		ax = fig.add_subplot(gs[(k % (n_row*n_col)) // n_row, k % n_row])
		Y = d['samp'][ndiscard1:, :, k].flatten()
		ax.tick_params(axis='both', labelsize=11.0)
		ax.hist(Y, bins=20, histtype='step', linewidth=2.0)
		ax.tick_params(labelleft=False)

		ax.text(0.95, 0.95, f"Age = {age[k]:.1f} Gyr", fontsize=11.0,
			    ha='right', va='top', transform=ax.transAxes)
		ax.text(0.95, 0.85, f"[Fe/H] = {feh[k]:.2f}", fontsize=11.0,
			    ha='right', va='top', transform=ax.transAxes)

		g1 = skew(Y[Y > 0.])
		ax.text(0.95, 0.75, f"g1 = {g1:.2f}", fontsize=11.0, color='red',
			    ha='right', va='top', transform=ax.transAxes)
		skewness.append(g1)

		results = np.percentile(Y[Y > 0.], [50.-34.13, 50., 50.+34.13])
		if (np.abs(g1) < 0.5):
			weight_par.append(results[1])
			weight_err.append(0.5*(results[2]-results[0]))
			ax.text(0.95, 0.65, f"w = {results[1]:.2e}", fontsize=11.0, color='blue',
			        ha='right', va='top', transform=ax.transAxes)
		else:
			weight_par.append(0.0)
			weight_err.append(0.0)
			ax.text(0.95, 0.65, f"w = 0.00", fontsize=11.0, color='blue',
			        ha='right', va='top', transform=ax.transAxes)

		if (k % (n_row*n_col) == n_row*n_col-1):
			pdf.savefig()
			plt.close()

	pdf.savefig()
	plt.close()

skewness = np.array(skewness)
weight_par = np.array(weight_par)
weight_err = np.array(weight_err)


# ----- Comparison plot ----- #
s = set_sfh(d['obs_name'].item(), d['mod_name'])
scale_lo, scale_hi = np.percentile(s.Y, [2.0, 98.0])

fig = plt.figure(1, figsize=(16,8))
gs = GridSpec(1, 2, left=0.05, bottom=0.05, right=0.95, top=0.95,
	          width_ratios=[1., 1.], wspace=0.10)

# Observed CMD
ax = fig.add_subplot(gs[0,0])
ax.imshow(s.Y, origin='lower', cmap='gray_r', vmin=scale_lo, vmax=scale_hi)
ax.tick_params(length=0, labelleft=False, labelbottom=False)
ax.text(0.50, 1.025, "Observed CMD", fontsize=30.0, fontweight='bold', color='black',
        ha='center', va='bottom', transform=ax.transAxes)

# Combined model CMD
ax = fig.add_subplot(gs[0,1])
ax.imshow(s.model_linpar(*weight_par), origin='lower', cmap='gray_r', vmin=scale_lo, vmax=scale_hi)
ax.tick_params(length=0, labelleft=False, labelbottom=False)
ax.text(0.50, 1.025, "Combined model CMD", fontsize=30.0, fontweight='bold', color='black',
        ha='center', va='bottom', transform=ax.transAxes)

plt.savefig('comp1.pdf')
plt.close()


# Printing the running time
print("----- running time : {0:.3f} sec".format(time.time()-start_time))

