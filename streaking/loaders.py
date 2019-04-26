import numpy as np
import matplotlib.pyplot as plt 
from PIL import Image

def load_tiff(path,plot=False, with_time_load=False):
	outp = np.array(Image.open(path))

	if with_time_load:
		with Image.open(path) as img:
			#key: 270 is the key we want - tuple of configs
			config_str = img.tag[270][0]
			config_str = config_str.split(",")
			matched = [k for k in config_str if bool(re.match("Time Range.*",k))==True]
			time_range = matched[0].split("=")[1].replace("\"","")
			time_number = time_range.split(" ")[0]
			time_units = time_range.split(" ")[1]
			# print time_number,"::", time_units
			sweep_time = float(time_number)
			if time_units == "us":
				sweep_time = sweep_time * 1e-6
			elif time_units == "ns":
				sweep_time = sweep_time * 1e-9
			elif time_units == "ps":
				sweep_time = sweep_time * 1e-12
	
	if plot:
		plt.imshow(outp)
		plt.show()
	if with_time_load :
		return outp, sweep_time
	else:
		return outp

def get_tif_metadata(path):

	
	k = 270
	with Image.open(path) as img:
		metadata = img.tag[k][0]
		metadata= metadata.replace("\r\n",",")
		bracket_open = False
		start = 0
		stop = start + 1
		segments = []
		while stop < len(metadata):
			if metadata[stop] == "," and bracket_open == False:
				segment = metadata[start:stop]
				segments = segments + [segment]
				segment = ""
				start = stop + 1
				stop = start + 1
			elif metadata[stop] == "\"":
				bracket_open = not bracket_open

			stop = stop + 1

		metadata_dict = {}
		for s in segments:
			kv = s.split("=")
			if len(kv) == 2:
				k = kv[0]
				v = kv[1].replace("\"","")
				metadata_dict.update({k:v})

		return metadata_dict


		# for v in img.tag[k][0].split("="):
		# 	print [v.replace("\"","") for v in v.split("\",")]


def get_tif_start_time(path):
	import time
	from datetime import datetime
	metadata = get_tif_metadata(path)
	print metadata.keys()
	d = metadata["Date"]
	t = metadata["Time"]
	td = datetime.strptime(d+"::"+t, "%d/%m/%Y::%H:%M:%S.%f")
	def unix_time_millis(dt):
		epoch = datetime.utcfromtimestamp(0)
		return (dt - epoch).total_seconds() * 1000.0

	return unix_time_millis(td)

if __name__ == "__main__":
	print "pass"
	