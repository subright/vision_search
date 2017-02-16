import logging
import getopt
import sys
import time
import random
import numpy as np
import scipy.misc
import colorsys

class LogHandler:

	def __init__(self):
		self.global_log_list = []
		self.log_list = []
		self.null_index = -1

	def add(self, curr_device, curr_site, curr_freq):
		if (curr_freq > 0):
			curr_log_str = str(curr_device) + ":" + str(curr_site) + ":" + str(curr_freq)
		else:
			curr_log_str = "0:0:0" # or str(null_count)

		self.update_global_log_list(curr_log_str)
		self.log_list.append(self.global_log_list.index(curr_log_str))

		if (self.null_index < 0 and curr_log_str == "0:0:0"):
			self.null_index = self.global_log_list.index("0:0:0")

	def sneak(self, begin_index, len_log):

		output_log_string = ""
		for i in range(begin_index, begin_index + len_log):
			output_log_string += ("" if (len(output_log_string) == 0) else ",") + str(self.log_list[i])

 		return output_log_string + "\n"

	def pop(self):

		self.log_list = []

	def update_global_log_list(self, log_str):
		if (log_str not in self.global_log_list):
			self.global_log_list.append(log_str)

	def size(self):
		return len(self.global_log_list)

	def write(self, filepath):
		fout = open(filepath, 'w')

		for i, log in enumerate(self.global_log_list):
			fout.write(str(i) + "," + log + "\n")

		fout.close()

def getIndexString(digit, v):
	res_str = ""
	for i in range(digit - 1, -1, -1):
		mod_v = v % (10 ** (i + 1))
		res_str += str(mod_v / (10 ** i))

	return res_str

def main(argv):
	
	basepath = '/.../'
	logpath = basepath + 'log'
	imgpath = basepath + 'png/'
	usageInfo = ''
	
	# default
	tranition_threshold = 0.14

    nb_noise = 5
	np_suffle = 5
	nb_logs = 200
	max_len = 60
	len_log = 60
	nb_devices = 4
	nb_sites = 10
	prob_devices = norm2mcmc(np.random.rand(4)) # [0.5, 0.3, 0.15, 0.05]
	freqs = [0, 0.25, 0.5, 0.75, 1.0]
	actions = {}
	loghandler = LogHandler()
	
	# prob_transitions = 1.0 * np.tile(prob_devices, [4, 1]).transpose() #2.0 * np.identity(nb_devices)
	# get device-transition matrix
	prob_transitions = np.random.rand(nb_devices, nb_devices)
	prob_transitions = prob_transitions / prob_transitions.sum(axis=0)
	# print prob_transitions

	# prob_sites_init = norm2mcmc(np.random.rand(nb_sites))
	# prob_sites = norm2mcmc(np.random.multinomial(nb_sites * nb_devices, prob_sites_init))
	
	print "######################################"
	print "### log-generator running with....."
	print "######################################"
	
	f_log = open(logpath, 'w')
	# log instance - device:site:freq (0 - 4)

	cnt = 0
	must_iterate = True
	while (must_iterate):
	
		color_mat = np.zeros((nb_devices, max_len, 3))

		# get new probability	
		prob_devices = norm2mcmc(np.random.rand(nb_devices))
		device_m1 = np.argmax(np.random.multinomial(1, prob_devices))
		device_m2 = np.argmax(np.random.multinomial(1, prob_devices))
		prob_devices = applyMajor(prob_devices, device_m1, device_m2)

		prob_sites = norm2mcmc(np.random.rand(nb_sites))
		site_m1 = np.argmax(np.random.multinomial(1, prob_sites))
		site_m2 = np.argmax(np.random.multinomial(1, prob_sites))
		prob_sites_i = applyMajor(prob_sites, site_m1, site_m2)
		
		# sample the number of device, site		
		nb_device_i = np.argmax(np.random.multinomial(1, prob_devices)) + 1
		
		# print(prob_sites_i)

		# get currne tdevice, site, frequency
		curr_device = np.argmax(np.random.multinomial(1, prob_devices[:nb_device_i])) + 1
		curr_site = np.argmax(np.random.multinomial(1, prob_sites_i)) + 1
		curr_freq = freqs[np.random.randint(5)]

		loghandler.add(curr_device, curr_site, curr_freq)

		window_dict = {}
		
		prev_i = 0
		for i in range(0, max_len):
			color_mat_i = getColor(curr_device, curr_site, curr_freq, nb_sites, nb_devices)
			color_mat[:, i, :] = color_mat_i
			
			prev_device = curr_device
			transit_flag = np.random.rand() < tranition_threshold

			if (transit_flag):
				curr_device = np.argmax(np.random.multinomial(1, prob_transitions[:nb_device_i,nb_device_i-1])) + 1

			if (prev_device != curr_device):
				curr_site = np.argmax(np.random.multinomial(1, prob_sites_i)) + 1

			curr_freq = freqs[np.random.randint(5)]
			loghandler.add(curr_device, curr_site, curr_freq)
	
		# print color_mat
		for i in range(np_suffle):
			begin_index = np.random.randint(max_len - len_log) 

			imgfile = imgpath + getIndexString(4, cnt) + '_' + str(i) + '_logimg_' + str(nb_logs) + '_' + str(len_log) + '_' + str(nb_device_i) + '.png'
			swap_index = np.random.permutation(len(window_dict))
			f_log.write(str(cnt * 5 + i) + "," + loghandler.sneak(begin_index, len_log))
			scipy.misc.imsave(imgfile, color_mat[:, begin_index:begin_index+len_log, :])

		loghandler.pop()

		if (cnt > nb_logs):
			must_iterate = False
		else:
			cnt = cnt + 1

	loghandler.write(logpath + "_dict")
	f_log.close()

def norm2mcmc(_probs):
	return 1.0 * _probs / _probs.sum()

def getColor(curr_device, curr_site, curr_freq, nb_sites, nb_devices):
	
	site_color = 1.0 * curr_site / (nb_sites + 1)
	freq_color = curr_freq # (curr_freq / 2 + 0.5) if (curr_freq > 0.25) else 0 #0.5 * curr_freq + 0.5
	color_mat = np.tile([0, 0, 0], (4, 1))
	
	for i in range(0, nb_devices):
		if (i == curr_device - 1 and freq_color > 0):
			color_mat_i = np.array(hsv_to_rgb(site_color, freq_color, 1))
			color_mat[i, :] = color_mat_i

	return color_mat

def hsv_to_rgb(h, s, v):
	return tuple(int(i*255) for i in colorsys.hsv_to_rgb(h, s, v))

if __name__ == "__main__":
	main(sys.argv[1:])
