# SFH_MCMC
(updated on 2020. 07. 29.)


## Description
The test Python codes for deriving star-formation history (SFH) from the synthetic color-magnitude diagram (CMD) using the MCMC algorithm


## Prerequisites
* The following Python modules should be installed.
  * ``numpy >= 1.18.5``
  * ``matplotlib >= 3.2.2``
  * ``scipy >= 1.5.0``
  * ``astropy >= 4.0.0``
  * ``emcee >= 3.0.2`` ([Reference link](https://emcee.readthedocs.io/en/stable/))

  
## Test history
* [Note 1 (SFH_200616.ipynb)](https://nbviewer.jupyter.org/gist/joungh93/e4a32ee7a62c34d0ed352f564f6e114b): 2020. 06. 16. revised
  * Test data: F456
  * Test models: 4 model CMDs with age = 7.0 - 10.0 Gyr (1.0 Gyr interval) and [Fe/H] = -1.60
  * MCMC walkers & samples: 32, 1000
  * Running time: 6.833 sec
  * Each CMD is normalized to make its sum of pixel values to 1. (wrong)
* [Note 2 (SFH_200724.ipynb)](https://nbviewer.jupyter.org/gist/joungh93/5c5a4be8025a0297d536ee4eb253ba8a): 2020. 07. 24. revised
  * Test data: F456
  * Test models: 7 model CMDs with age = 7.0 - 10.0 Gyr (0.5 Gyr interval) and [Fe/H] = -1.60
  * MCMC walkers & samples: 128, 5000
  * Running time: 146.245 sec
  * The normalization problem is revised.
* ``set_sfh.py`` is a class that runs the MCMC algorithm with the input data (an observed CMD and model CMDs) including a main program that actually executes the class with the following information.
  * Test data: F456
  * Test models: 11 model CMDs with age = 8.0 - 10.0 Gyr (0.2 Gyr interval) and [Fe/H] = -1.60
  * MCMC walkers & samples: 128, 10000
  * Running time: 7 min
* ``test1.py`` generates the following plots.
  * ``trace1.pdf`` - trace plots
  * ``hist1.pdf`` - histograms that show the distribution of each weight parameter after discarding the non-convergent regions.
  * ``comp1.pdf`` - comparison between the observed CMD and the synthesized CMD using the derived weight paramters


## Acknowledgements
* I.S. Jang provided the model CMDs using [Padova ischrone](http://stev.oapd.inaf.it/cgi-bin/cmd) and the observed CMDs of NGC 300 obtained by Hubble Space Telescope. :blush:

