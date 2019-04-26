import numpy as np 


def cross_section(arr,debug = 0):
	if debug > 0:
		print "transforms.cross_section:: 'arr' Array shape: [{0}]".format(arr.shape)
	return np.sum(arr,axis=0)

def get_timeseries(img,xmin,xmax,debug=0):
	if debug > 0:
		print "transforms.get_timeseries:: \"img\" array shape: [{0}]".format(img.shape)
		print "transforms.get_timeseries:: xmin: {0}, xmax: {1}".format(xmin,xmax)
	
	series= img[:,xmin:xmax]
	return np.sum(series,axis=1)

def detrend(xs,ys,debug = 0):
	N = xs.shape[0]
	xs_augmented = np.transpose([xs,np.ones(N)])
	lin,_,_,_ = np.linalg.lstsq(xs_augmented,ys)
	m,c = lin[0], lin[1]

	trend = m*xs + c
	if debug > 0:
		print "transforms.detrend: model:: [y=m.x+c], parameters:: m: {0}, c: {1}".format(m,c)
		
	detrended = ys - trend
	return detrended,m,c

def power_spectrum(signal,sweep_time):
	N_pixels = signal.shape[0]
	freqs = np.fft.fftshift(np.fft.fftfreq(N_pixels,sweep_time/float(N_pixels)))
	def compute_power_spectrum(signal):
		 return np.absolute(np.fft.fftshift(np.fft.fft(signal)))*2
		 
	ps = compute_power_spectrum(signal)
	return freqs, ps


def stage2_segmentation(img,debug_plot=False,label=None):
	'''
	
	Compute the intervals on the x-axis that represent the particles
	TODO - return quality scores for each particle seen?
	'''
	intervals = get_dual_peaks(img)
	print intervals
	if debug_plot:
		xsection = cross_section(img)
		xsection = xsection - np.median(xsection)

		fig,ax = plt.subplots(1)
		xs = np.arange(0,len(xsection))
		ax.plot(xs,xsection)
		for [start,stop] in intervals:
			ax.plot(xs[start:stop], xsection[start:stop],'o')

		ax.set_xlabel("Pixel index [spatial coord]")
		ax.set_ylabel("Image cross section [pixel sum]")
		ax.set_title("stage2 - segmentation\nIntervals: {}".format(intervals))
		plt.savefig("images/stage2/segmentation_{}.png".format(label))
		plt.close()

	return intervals
		