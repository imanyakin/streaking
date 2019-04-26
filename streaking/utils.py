import os
import matplotlib.pyplot as plt 

def make_folder(path):
	try:
		os.stat(path)
	except:
		os.mkdir(path)

	return


def figure_save_and_close(fig,path):
	fig.savefig(path)
	plt.close(fig)