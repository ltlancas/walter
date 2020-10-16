## for calculating the mean luminosity of a stellar population
## by LTL

import numpy as np

if __name__ == '__main__':
	# First specify the properties of the populations you'd like to calculate f_detect 
	# for by giving lists of metallicities and ages. These should be linearly spaced 
	metlist = np.array([0.5,0.25,0.0,-0.25,-0.5,-0.75,-1.0,-1.25,-1.5,-1.75,-2.0,-2.25,-2.5,-2.75,-3.,-3.25])
	agelist = np.array([8.95,9.,9.05,9.1,9.15,9.2,9.25,9.3,9.35,9.4,9.45,9.5,9.55,9.6,9.65,9.7,9.75,9.8,9.85,9.9,9.95,10.,10.05,10.1,10.15])
	nmet = len(metlist) # number of metallicities
	nage = len(agelist) # number of ages
	dage = agelist[1] - agelist[0] # logarithmic difference in  age bins, for selection below


	## specify directory where isochrones are stored, this is set up for the 
	## MIST Roman Isocrones, changing that would involve changing the below
	iso_dir = "../mist_isos/"
	## name od output file
	output_name = "data/tot_lum" # it will have a .npy at the end

	## define the IMF
	## specify the power-law index for the IMF that you would like to use
	mpow = -1.3
	## specify the number of mass samples you would like for integrals to be 
	## numericall performed below
	nmass_samp = 10000000
	# mass range in solar masses
	(mMin,mMax) = (0.1,10)
	# masses spaced linearly throughout imf space
	mlin = np.linspace(mMin,mMax,nmass_samp)
	# difference between the masses, calculated so that the integral 
	# over the 'pre-image' can be calculated accurately
	dm = mlin[1] - mlin[0]
	# the IMF
	xim = mlin**mpow
	# the IMF, normalized over the mass range of interest
	xim = xim/np.trapz(xim,mlin)

	# specify the number of filters. The default is 6 (Z,Y,J,H,F). Changing this 
	# number will require some hard code changes below
	nfilts = 5

	AB_Vega = np.array([0.487, 0.653, 0.958, 1.287, 1.552]) #This is m_AB - m_Vega

	totdat = np.zeros((nmet,nage,nfilts))
	# load isochrones from mist
	for (l,met) in enumerate(metlist):
		print(met)
		if met>-0.1:
			(age,imass,T,L) = np.loadtxt("%smist_fehp%1.2f.iso.cmd"%(iso_dir,abs(met)),usecols=[1,2,4,6]).T
			(Z,Y,J,H,F) = np.loadtxt("%smist_fehp%1.2f.iso.cmd"%(iso_dir,abs(met)),usecols=[10,11,12,14,15]).T
		else:
			(age,imass,T,L) = np.loadtxt("%smist_fehm%1.2f.iso.cmd"%(iso_dir,abs(met)),usecols=[1,2,4,6]).T
			(Z,Y,J,H,F) = np.loadtxt("%smist_fehm%1.2f.iso.cmd"%(iso_dir,abs(met)),usecols=[10,11,12,14,15]).T

		(Zage,Yage,Jage,Hage,Fage) = (np.zeros((nage)),np.zeros((nage)),np.zeros((nage)),np.zeros((nage)),np.zeros((nage)))
		# select a single isochrone
		for (agej,tau) in enumerate(agelist):
			sids = np.intersect1d(np.where(age>tau-0.01),np.where(age<tau+0.01))

			Z_out = np.interp(mlin,imass[sids],Z[sids],right=100) + AB_Vega[0]
			Y_out = np.interp(mlin,imass[sids],Y[sids],right=100) + AB_Vega[1]
			J_out = np.interp(mlin,imass[sids],J[sids],right=100) + AB_Vega[2]
			H_out = np.interp(mlin,imass[sids],H[sids],right=100) + AB_Vega[3]
			F_out = np.interp(mlin,imass[sids],F[sids],right=100) + AB_Vega[4]

			Zage[agej] = np.trapz((10**(Z_out/-2.5))*xim,mlin)
			Yage[agej] = np.trapz((10**(Y_out/-2.5))*xim,mlin)
			Jage[agej] = np.trapz((10**(J_out/-2.5))*xim,mlin)
			Hage[agej] = np.trapz((10**(H_out/-2.5))*xim,mlin)
			Fage[agej] = np.trapz((10**(F_out/-2.5))*xim,mlin)

		tsave = np.array([Zage,Yage,Jage,Hage,Fage]).T
		totdat[l] = tsave
	np.save(output_name,totdat)

