import numpy as np 
import matplotlib.pyplot as plt 
from scipy.signal import argrelmax, argrelmin
def cross_section(arr,debug = 0):
	if debug > 0:
		print "transforms.cross_section:: 'arr' Array shape: [{0}]".format(arr.shape)
	return np.sum(arr,axis=0)

def get_timeseries(img,xmin,xmax):
	series = img[:,xmin:xmax]
	return np.sum(series,axis=1)

def segment_score_threshold_level(img,x_start,x_stop,threshold=0.15):
	xsection = cross_section(img)
	threshold_level = np.mean(xsection) + threshold*np.std(xsection)
	segment_intensity = xsection[x_start:x_stop]
	#compute how far above baseline intensity we are & rescale by intensity
	score = (np.max(segment_intensity)-threshold_level)/threshold_level
	return score

def extract_segments(img,level_threshold=0.15,segment_score_threshold = 0.5,debug = 0, debug_plot=0):
	xsection = cross_section(img)
	def get_threshold_level(level_threshold):
		mu = np.mean(xsection)
		std = np.std(xsection) 
		return mu + level_threshold * std
	threshold_level = get_threshold_level(level_threshold)

	inds = np.zeros(xsection.shape)
	inds[xsection >=threshold_level] = 1
	inds[xsection < threshold_level] =-1

	def find_zero_crossings(data):
		inds = np.where(np.multiply(data[0:-1],data[1:]) < 0)
		return inds[0]

	crossings = find_zero_crossings(inds)
	# print crossings
	# if len(crossings)%2 != 0:
	# 	raise ValueError("Odd number of intervals start/end-points")

	thresholded_intervals = [(crossings[i],crossings[i+1]) for i in range(0,crossings.shape[0]-1,2)]

	segments = []
	for (start,stop) in thresholded_intervals:
		ys = np.convolve(xsection[start:stop],0.2*np.ones(5)) #smooth data with averaging filter
		maxima = argrelmax(ys)[0]
		if len(maxima) == 1:
			segments = segments + [(start,stop)]
		else:
			
			minima = [0] + list(argrelmin(ys)[0]) + [stop-start]
			for i in range(0,len(minima)-1,1):
				segments = segments + [(start + minima[i], start + minima[i+1])]

	
	segment_score_function = segment_score_threshold_level
	scores = [ segment_score_function(img,start,stop) for (start,stop) in segments]
	
	outp = zip(segments,scores)

	outp = [(segment, score) for (segment, score) in outp if score > segment_score_threshold]

	if debug_plot > 0:
		fig,axarr = plt.subplots(2,2,figsize=(9,9))
		axarr = [axarr[0][0],axarr[0][1],axarr[1][0],axarr[1][1]]

		axarr[0].plot(xsection)
		axarr[0].axhline(threshold_level,label="threshold_level ($\mu + thresold \\times \sigma$)")
		axarr[0].set_title("extract_segments - stage 1: Image cross-section")

		axarr[1].plot(xsection)
		axarr[1].axhline(threshold_level)
		for (start,stop) in thresholded_intervals:
			axarr[1].plot(range(start,stop),xsection[start:stop],"o")

		axarr[1].set_title("extract_segments - stage 2: Compute segments above threshold")

		axarr[2].plot(xsection)
		axarr[2].axhline(threshold_level)
		for (start,stop) in segments:
			axarr[2].plot(range(start,stop),xsection[start:stop],"o")
		axarr[2].set_title("extract_segments - stage 3: Compute intervals")		
		
		axarr[3].plot(xsection)
		axarr[3].axhline(threshold_level)
		for i,((start,stop),score) in enumerate(outp):
			axarr[3].plot(range(start,stop),xsection[start:stop],"o",label="segment[{3}],start:{0}, stop: {1}, score: {2}".format(start,stop,score,i))
		
		axarr[3].legend()
		plt.show()

	if debug > 0:
		print "streaking.timeseries_extraction.extract_segments..."
		for ((start,stop), score) in outp:
			print "segment [start,stop,score]: {0}, {1}, {2}".format(start,stop,score)

	return outp

def extract_timeseries(img,level_threshold=0.15,segment_score_threshold = 0.5,debug = 0, debug_plot=0):
	segments_scored = extract_segments(img, level_threshold=level_threshold,segment_score_threshold=segment_score_threshold,debug=debug-1,debug_plot=debug_plot-1)

	timeseries_scored = [(get_timeseries(img,xstart,xstop),score) for ((xstart,xstop),score) in segments_scored]
	if debug > 0:
		print "streaking.timeseries_extraction.extract_timeseries..."
		for i,(_,score) in enumerate(timeseries_scored):
			print "timeseries [{0}] score: {1}".format(i,score)
	if debug_plot > 0:
		for (ts,score) in timeseries_scored:
			print "score timeseries: {0}".format(score)
		fig,ax = plt.subplots(1)
		for i,(timeseries,score) in enumerate(timeseries_scored):
			((start,stop),_) = segments_scored[i]
			ax.plot(timeseries,label="timeseries [segment,start:{0},stop:{1}], score: {2}".format(start,stop,score))
		ax.legend()
		ax.set_title("streaking.timeseries_extraction.extract_timeseries")
		plt.show()

	return timeseries_scored

