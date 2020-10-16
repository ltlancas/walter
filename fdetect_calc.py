## for calculating the fraction of a population that is detected
## for a given limiting magnitude and population based on a very well 
## sampled IMF
## by LTL

import numpy as np

def get_f(B_mu, B_cut,ms,xiofm,dmf):
	Bcut_ids = np.where(B_mu<B_cut)[0]
	dms = ms[Bcut_ids][1:] - ms[Bcut_ids][:-1]
	cuts = np.where(dms>dmf*1.1)[0]
	cuts = np.sort(np.concatenate(([0],cuts,cuts+1,[len(dms)-1])))
	fsum = 0
	for i in range(len(cuts)-1)[::2]:
		ximtint = xiofm[Bcut_ids][cuts[i]:cuts[i+1]]
		mlintint = ms[Bcut_ids][cuts[i]:cuts[i+1]]
		fsum += np.trapz(ximtint,mlintint)
	return fsum



if __name__ == '__main__':
	# First specify the properties of the populations you'd like to calculate f_detect for
	# by giving lists of metallicities and ages. These should be linearly spaced 
	metlist = np.array([0.5,0.25,0.0,-0.25,-0.5,-0.75,-1.0,-1.25,-1.5,-1.75,-2.0,-2.25,-2.5,-2.75,-3.,-3.25])
	agelist = np.array([8.95,9.,9.05,9.1,9.15,9.2,9.25,9.3,9.35,9.4,9.45,9.5,9.55,9.6,9.65,9.7,9.75,9.8,9.85,9.9,9.95,10.,10.05,10.1,10.15])
	nmet = len(metlist) # number of metallicities
	nage = len(agelist) # number of ages
	dage = agelist[1] - agelist[0] # logarithmic difference in  age bins, for selection below

	## specify directory where isochrones are stored, this is set up for the 
	## MIST Roman Isocrones, changing that would involve changing the below
	iso_dir = "../mist_isos/"
	## name od output file
	output_name = "data/tot_fdet" # it will have a .npy at the end

	## define the IMF
	## specify the power-law index for the IMF that you would like to use
	mpow = -1.3
	## specify the number of mass samples you would like for integrals to be 
	## numericall performed below
	nmass_samp = 10000000
	# mass range in solar masses
	(mMin,mMax) = (0.1,100)
	# masses spaced linearly throughout imf space
	mlin = np.linspace(mMin,mMax,nmass_samp)
	# difference between the masses, calculated so that the integral 
	# over the 'pre-image' can be calculated accurately
	dm = mlin[1] - mlin[0]
	# the IMF
	xim = mlin**mpow
	# the IMF, normalized over the mass range of interest
	xim = xim/np.trapz(xim,mlin)

	# then specify the range of *absolute* magnitudes you would like to cover
	# these are related to the exposure time of the observation and the distance 
	# to the observed population through Eq 2 of our paper
	nmags = 200
	# what are the brightest and faintest magnitudes you want to calculate for
	(minMag,maxMag) = (1,-7)
	# get the range of absolute magnitudes you would like to calculate fdet for
	Mags = np.linspace(minMag,maxMag,nmags)
	# specify the number of filters. The default is 6 (Z,Y,J,H,F). Changing this 
	# number will require some hard code changes below
	nfilts = 5

	# shape of the total output array
	totdat = np.zeros((nmet,nage,nmags,nfilts+1))
	# load isochrones from mist
	for (l,met) in enumerate(metlist):
		print(met)
		if met>-0.1:
			(age,imass,T,L) = np.loadtxt("%smist_fehp%1.2f.iso.cmd"%(iso_dir,abs(met)),usecols=[1,2,4,6]).T
			(Z,Y,J,H,F) = np.loadtxt("%smist_fehp%1.2f.iso.cmd"%(iso_dir,abs(met)),usecols=[10,11,12,14,15]).T
		else:
			(age,imass,T,L) = np.loadtxt("%smist_fehm%1.2f.iso.cmd"%(iso_dir,abs(met)),usecols=[1,2,4,6]).T
			(Z,Y,J,H,F) = np.loadtxt("%smist_fehm%1.2f.iso.cmd"%(iso_dir,abs(met)),usecols=[10,11,12,14,15]).T
		
		tsave = np.zeros((nage,nmags,nfilts + 1))
		# select a single isochrone
		for (k,a) in enumerate(agelist):
			sids = np.intersect1d(np.where(age>a-dage/2),np.where(age<a+dage/2))
			
			# interpolate what the right magnitudes are
			AB_Vega = np.array([0.487, 0.653, 0.958, 1.287, 1.552]) #This is m_AB - m_Vega
			Z_out = np.interp(mlin,imass[sids],Z[sids]) + AB_Vega[0]
			Y_out = np.interp(mlin,imass[sids],Y[sids]) + AB_Vega[1]
			J_out = np.interp(mlin,imass[sids],J[sids]) + AB_Vega[2]
			H_out = np.interp(mlin,imass[sids],H[sids]) + AB_Vega[3]
			F_out = np.interp(mlin,imass[sids],F[sids]) + AB_Vega[4]


			# distance to object
			(fZdetect,fYdetect,fJdetect,fHdetect,fFdetect) = (np.zeros(nmags),np.zeros(nmags),np.zeros(nmags),np.zeros(nmags),np.zeros(nmags))
			for (i,mag) in enumerate(Mags):
				fZdetect[i] = get_f(Z_out,mag,mlin,xim,dm)
				fYdetect[i] = get_f(Y_out,mag,mlin,xim,dm)
				fJdetect[i] = get_f(J_out,mag,mlin,xim,dm)
				fHdetect[i] = get_f(H_out,mag,mlin,xim,dm)
				fFdetect[i] = get_f(F_out,mag,mlin,xim,dm)
			tsave[k] = np.array([Mags,fZdetect,fYdetect,fJdetect,fHdetect,fFdetect]).T
		totdat[l] = tsave
	np.save(output_name,totdat)
	
