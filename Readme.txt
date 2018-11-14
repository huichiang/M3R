Hi! This repo contains the final report and the code used in my Year 3 Individual Research Project, 'Bayesian Inference with Applications to Censored Data', I did at Imperial College London, under the supervision of Dr. Daniel Mortlock.

You will find that the 'M3R.py' file contains functions for:
- Sampling from a negative binomial distribution 
- Computing the binomial coefficient ('choose')
- Sampling from a negative hypergeometric distribution
- Gibbs sampler for a 'fishing' experiment, outlined in the report
- Sampling from an inverse chi-squared distribution
- *Gibbs sampler for linear regression*
- Assessing convergence of the Gibbs sampler using the Gelman-Rubin Diagnostic

** The Gibbs sampler for linear regression simulates censored data generated from a mixture model of normal distributions. It is outlined in detail in report, and was originally taken from the following paper:

Kelly, B. (2007). Some Aspects of Measurement Error in Linear Regression of Astronomical Data. The Astrophysical Journal. 665. 10.1086/519947. 