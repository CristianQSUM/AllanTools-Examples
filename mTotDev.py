import numpy as np
import allantools as allan
from scipy.stats import chi2

def AllanDev(yList, taus='all', rate=1, ADevTyp='Total', tauMax=0.4, ComputeErr=True):
    '''
    Computing Allan deviation using allantools.mtotdev(data, rate, data_type=phase, taus='all')
    Return format (Tau, ADev, ADevSigL, ADevSigU)
    '''

    if ADevTyp == 'Total':
        (tau, ADev, ADevError, ADevN) = allan.mtotdev(yList, rate=rate, data_type='freq', taus=taus)
        tauMax = min(abs(tauMax), 1.0)
    
    else:
        (tau, ADev, ADevError, ADevN) = allan.oadev(yList, rate=rate, data_type='freq', taus=taus)
        tauMax = min(abs(tauMax), 1.0)
    
    N = len(yList)
    ntau = len(tau)

    if taus == 'all':
        ntau = int(round(tauMax*ntau))
    elif taus == 'octave' or taus == 'decade':
        itau = 0
        while tau[itau] <= tauMax*tau[-1]:
            itau += 1
        ntau = itau + 1
    tau = tau[:ntau]
    ADev = ADev[:ntau]
    ADevErr = ADevError[:ntau]
    ADevN = ADevN[:ntau]

    if ComputeErr:
        ADevErrL, ADevErrU = ComputeADevErrors(N, ntau, rate, tau, ADev, ADevErr, ADevTyp)[:2]
        return (tau, ADev, ADevErrL, ADevErrU)
    else:
        return (tau, ADev, ADevErr, ADevErr)


def ComputeADevErrors(N, ntau, rate, tau, ADev, ADevErr, ADevType='Total', ModelType='chi2'):
	"""Compute 1-sigma confidence intervals and one- or two-sided uncertainties for Allan deviations.
	ARGUMENTS:
	N             (int) - Total number of data points
	ntau          (int) - Number of averaging times contained in 'tau'
	rate        (float) - Sampling rate of input data (Hz)
	tau      (np.array) - Averaging times
	ADev     (np.array) - Allan deviation
	ADevErr  (np.array) - One-sided uncertainty output by allantools
	ADevType      (str) - Type of Allan deviation
	ModelType     (str) - Type of model to use to compute uncertainty
	RETURN FORMAT: (ADevErrL, ADevErrU, ADevCIL, ADevCIU)
	ADevErrL (np.array) - Lower bound of ADev uncertainty
	ADevErrU (np.array) - Upper bound of ADev uncertainty
	ADevCIL  (np.array) - Lower bound of ADev confidence interval
	ADevCIU  (np.array) - Upper bound of ADev confidence interval
	"""

	if ModelType == 'chi2':
		## Compute Allan deviation confidence intervals based on chi2-distribution
		ADevErrL = np.zeros(ntau)
		ADevErrU = np.zeros(ntau)
		ADevCIL  = np.zeros(ntau)
		ADevCIU  = np.zeros(ntau)

		for i in range(ntau):
			## Averaging factor tau = m*tau0 = m/rate
			m = rate*tau[i]
			if ADevType == 'Total':
				## Equivalent degrees of freedom for total ADev (white frequency noise case)
				edf = 1.5*float(N)/m
			else:
				## Equivalent degrees of freedom for overlapped ADev (white frequency noise case)
				edf = (3*(float(N)-1)/(2*m) - 2*(float(N)-2)/float(N))*4*m**2/(4*m**2 + 5)
			## Chi-Squared values corresponding to +/- sigma
			(chi2L, chi2U) = ChiSquaredModel(edf)
			## Estimate two-sided confidence interval
			ADevCIL[i]  = np.sqrt(edf/chi2U)*ADev[i]
			ADevCIU[i]  = np.sqrt(edf/chi2L)*ADev[i]
			ADevErrL[i] = np.abs(ADevCIL[i] - ADev[i])
			ADevErrU[i] = np.abs(ADevCIU[i] - ADev[i])
	else:
		## Simple one-sided confidence interval output by allantools (typically an underestimate of true confidence interval)
		##   ADevErr = ADev/sqrt(n), where n is the number of data pairs used to compute each ADev
		ADevErrL = ADevErr
		ADevErrU = ADevErr
		ADevCIL  = ADev - ADevErr
		ADevCIU  = ADev + ADevErr

	return (ADevErrL, ADevErrU, ADevCIL, ADevCIU)

################### End of ComputeADevErrors() ######################
#####################################################################

def ChiSquaredModel(dof, p=0.683):
	"""Compute chi-squared distribution for a given number of degrees of freedom (dof) at confidence level p.
	ARGUMENTS:
	dof   (float) - Number of degrees of freedom
	p     (float) - Value between 0 and 1 corresponding to confidence level.
					Defaults to 0.683 corresponding to 1 sigma
	RETURN FORMAT: (chi2U, chi2L)
	chi2U (float) - Upper limit of chi2 distribution
	chi2L (float) - Lower limit of chi2 distribution
	"""

	chi2L = chi2.ppf(0.5*(1-p), dof) ## Percent point function (inverse of cdf)
	chi2U = chi2.ppf(0.5*(1+p), dof) 

	return (chi2L, chi2U)

#################### End of ChiSquaredModel() #######################
