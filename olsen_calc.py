# for calculating quantities related to the accuracy of photometery 
# due to crowding. Specifically, here we want to calculate the mean 
# luminosity of a population (number weighted) and the square of 
# the luminosity of low luminosity stars, which comes up in section 
# 2.1 of our paper and in Olsen et al. 2003

import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.optimize import brentq

def get_L2_int(B_mu,B_cut,ms,xiofm,dmf):
	bcis = np.where(B_mu>B_cut)[0]
	dms = ms[bcis][1:] - ms[bcis][:-1]
	cuts = np.where(dms>dmf*1.1)[0]
	cuts = np.sort(np.concatenate(([0],cuts,cuts+1,[len(dms)-1])))
	cut_inds = range(len(cuts)-1)[::2]
	(xiofm,B_mu,ms) = (xiofm[bcis],B_mu[bcis],ms[bcis])
	ximtint_arr = [xiofm[j:cuts[i+1]] for (i,j) in enumerate(cuts[:-1])]
	L2int_arr = [10**(B_mu[j:cuts[i+1]]/(-1.25)) for (i,j) in enumerate(cuts[:-1])]
	mlintint_arr = [ms[j:cuts[i+1]] for (i,j) in enumerate(cuts[:-1])]
	fsum = np.sum(np.array([np.trapz(L2int_arr[i]*ximtint_arr[i],mlintint_arr[i]) for i in range(len(cut_inds))]))
	return fsum

"""
def get_L2_int(B_mu,B_cut,ms,xiofm,dmf):
	Bcut_ids = np.where(B_mu>B_cut)[0]
	dms = ms[Bcut_ids][1:] - ms[Bcut_ids][:-1]
	cuts = np.where(dms>dmf*1.1)[0]
	cuts = np.sort(np.concatenate(([0],cuts,cuts+1,[len(dms)-1])))
	cut_inds = range(len(cuts)-1)[::2]
	fsum = 0
	for i in cut_inds:
		ximtint = xiofm[Bcut_ids][cuts[i]:cuts[i+1]]
		L2int = 10**(B_mu[Bcut_ids][cuts[i]:cuts[i+1]]/(-1.25))
		mlintint = ms[Bcut_ids][cuts[i]:cuts[i+1]]
		#plt.plot(mlintint,ximtint,'o')
		fsum += np.trapz(L2int*ximtint,mlintint)
	#plt.show()
	return fsum
"""

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
	output_name = "data/l2" # it will have a .npy at the end

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

	# This is m_AB - m_Vega
	AB_Vega = np.array([0.487, 0.653, 0.958, 1.287, 1.552])


	ncrow = 1

	# apparent magnitude limit for a given filter and the corresponding apparent luminosity
	l2_out = np.zeros((nmet,nage,nfilts,nmags))
	for meti in range(nmet):
		print(metlist[meti])
		if metlist[meti]>-0.1:
			(age,imass,T,L) = np.loadtxt("%smist_fehp%1.2f.iso.cmd"%(iso_dir,abs(metlist[meti])),usecols=[1,2,4,6]).T
			(Z,Y,J,H,F) = np.loadtxt("%smist_fehp%1.2f.iso.cmd"%(iso_dir,abs(metlist[meti])),usecols=[10,11,12,14,15]).T
		else:
			(age,imass,T,L) = np.loadtxt("%smist_fehm%1.2f.iso.cmd"%(iso_dir,abs(metlist[meti])),usecols=[1,2,4,6]).T
			(Z,Y,J,H,F) = np.loadtxt("%smist_fehm%1.2f.iso.cmd"%(iso_dir,abs(metlist[meti])),usecols=[10,11,12,14,15]).T
		for (agej,tau) in enumerate(agelist):
			print agej
			# select a single isochrone
			sids = np.intersect1d(np.where(age>tau-dage/2),np.where(age<tau+dage/2))

			# interpolate absolute magnitudes on to the masses that we 
			# are covering and put them in the AB magnitude system
			# the right = -100 is to make sure that we don't spend time 
			# integrating over "high mass" stars which have already 
			# burnt out
			Z_out = np.interp(mlin,imass[sids],Z[sids],right=-100) + AB_Vega[0] 
			Y_out = np.interp(mlin,imass[sids],Y[sids],right=-100) + AB_Vega[1]
			J_out = np.interp(mlin,imass[sids],J[sids],right=-100) + AB_Vega[2]
			H_out = np.interp(mlin,imass[sids],H[sids],right=-100) + AB_Vega[3]
			F_out = np.interp(mlin,imass[sids],F[sids],right=-100) + AB_Vega[4]


			Filts = [Z_out,Y_out,J_out,H_out,F_out]
			for filtj in range(nfilts):
				for (k,mag) in enumerate(Mags):
					IBtZ = Filts[filtj]
					L2 = get_L2_int(IBtZ,mag,mlin,xim,dm)
					l2_out[meti][agej][filtj][k] = L2
					
	np.save(output_name,l2_out)


