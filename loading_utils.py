
from PIL.Image import QUAD
import torch.optim as optim
import torch as th

import librosa

from networks import Config, EasyNetwork, FramewiseNetwork

from mido import MidiFile

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import os

import scipy
import scipy.signal

from multiprocessing import Pool


# w&b key : 8ab0a11fe5afa358866866ae39725e8bc2b150e3

ds_path = "/media/oscar/Data/workspace/sound_to_score/maestro-v3.0.0/"
worked_ds_path = "/media/oscar/Data/workspace/sound_to_score/worked-maestro-v3.0.0/"
# worked_ds_path = "/media/oscar/Data/workspace/sound_to_score/worked_phase-maestro-v3.0.0/"
# worked_ds_path = "/media/oscar/Data/workspace/sound_to_score/worked2-maestro-v3.0.0/"



def generate_spectro (audio_filename, start, duration, custom_ds_path=None): # generate_spectro_base
	cur_ds_path = ds_path if custom_ds_path is None else custom_ds_path
	wav_in, sr = librosa.load(cur_ds_path+audio_filename, offset=start, duration=duration)
	hop_length = 500
	ps = librosa.feature.melspectrogram(hop_length=hop_length, sr=sr, win_length=2048, n_fft=2048*4, y=wav_in, n_mels=429, fmin=0, fmax=3000)
	ps = librosa.power_to_db(ps, ref=1)
	return (ps.T + 25) / 35, hop_length/sr

def filter_coef (w, sr):
	A = 0.0001 * w # can go from 0.1 to 10 ?
	B = w*w
	a = [(A*sr + B + sr*sr), (-A*sr-2*sr*sr), sr*sr]
	b = [A*sr, -A*sr]
	return a, b

def butter_bandpass(lowcut, highcut, fs, order=5):
	nyq = 0.5 * fs
	low = lowcut / nyq
	high = highcut / nyq
	b, a = scipy.signal.butter(order, [low, high], btype='band')
	return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
	b, a = butter_bandpass(lowcut, highcut, fs, order=order)
	y = scipy.signal.lfilter(b, a, data)
	return y

def generate_spectro_new (audio_filename, start, duration):
	wav_in, sr = librosa.load(ds_path+audio_filename, offset=start, duration=duration)
	hop_length = 500

	# y.. + Ay. + By = Bx.
	# ce que je veux : x[n]

	# on a :
	# x.. = (x[n-2]-2x[n-1]+x[n])*sr*sr
	# x. = (x[n]-x[n-1]) * sr
	# Soit : 
	# (y[n-2]-2y[n-1]+y[n])*sr*sr + A * (y[n]-y[n-1]) * sr + B * y[n] = B(x[n]-x[n-1])*sr
	# y[n] * (A*sr + B + sr*sr) + y[n-1] * (-A*sr-2*sr*sr) + y[n-2] * sr*sr = B*sr*x[n] - B*sr*x[n]
	# a = [(A*sr + B + sr*sr), (-A*sr-2*sr*sr), sr*sr]
	# b = [B*sr, -B*sr]
	all_res = []
	min_freq = 27.5
	max_freq = 4186
	n_channels = 429
	for i in range(n_channels):
		freq = min_freq*np.power(2, i*88/12/n_channels)
		# print(freq)
		# q = 0.02 # with order = 2
		q = 0.01 # with order = 1
		low_freq = freq*(1-q)
		high_freq = freq*(1+q)
		res = butter_bandpass_filter(wav_in, low_freq, high_freq, sr, order=1)
		# print(res.shape)
		# res = res.reshape(())
		res = res[:res.shape[0]-res.shape[0]%hop_length]
		res = res.reshape((-1, hop_length))
		all_res.append(np.max(np.abs(res), axis=1))
	all_res = np.stack(all_res)
	# print(all_res.shape)
	# plt.imshow(all_res)
	# plt.show()
	ps = librosa.feature.melspectrogram(hop_length=hop_length, sr=sr, win_length=2048, n_fft=2048*4, y=wav_in, n_mels=429, fmin=0, fmax=1300)
	ps = librosa.power_to_db(ps, ref=1)
	# plt.imshow(ps, origin="lower")
	# plt.show()
	# print(ps.shape)
	# help(scipy.signal.lfilter)
	# exit()
	
	
	# ps = np.zeros((50, 50))
	return (all_res.T - np.mean(all_res)) / np.std(all_res), hop_length/sr

def generate_train_example (audio_filename, midi_filename, start, duration=None):
	ps, frame_len = generate_spectro (audio_filename=audio_filename, start=start, duration=duration)
	handler = MidiHandler (MidiFile(ds_path+midi_filename, clip=True), frame_len)
	onset_len = ps.shape[0] if len(ps.shape) == 2 else ps.shape[1]
	onsets = handler.get_onsets(onset_len, start)
	frames = handler.get_frames(onset_len, start)
	return ps, np.stack([onsets, frames])

class MidiHandler:
	def __init__ (self, mid, frame_length):
		self.mid = mid
		self.tempo = 500000
		self.ticks_per_bits = self.mid.ticks_per_beat

		self.sec_per_tick = self.tempo / 1e6 / self.ticks_per_bits

		self.frame_length = frame_length
		self.track_nb = 88
	
	def get_onsets (self, n_frame, start_time):
		to_return = []
		message_time = 0
		to_return_time = start_time
		track_on = [0 for i in range(self.track_nb)]
		for message in self.mid.tracks[1]:
			message_time += message.time  * self.sec_per_tick

			while len(to_return) < n_frame and to_return_time + self.frame_length < message_time:
				to_return.append(track_on[:])
				track_on = [0 for i in range(self.track_nb)]
				to_return_time += self.frame_length
			
			if message.type == "note_on" and message.velocity > 0 and message.note-21 < self.track_nb:
				track_on[message.note-21] = 1
			
			if to_return_time > message_time:
				track_on = [0 for i in range(self.track_nb)]
		
		while len(to_return) < n_frame:
			to_return.append(track_on)
			track_on = [0 for i in range(self.track_nb)]
			to_return_time += self.frame_length

			if to_return_time > message_time:
				track_on = [0 for i in range(self.track_nb)]
		
		return np.array(to_return)

	def get_frames (self, n_frame, start_time):
		to_return = []
		message_time = 0
		to_return_time = start_time
		track_on = [0 for i in range(self.track_nb)]
		for message in self.mid.tracks[1]:
			message_time += message.time  * self.sec_per_tick

			while len(to_return) < n_frame and to_return_time + self.frame_length < message_time:
				to_return.append(track_on[:])
				to_return_time += self.frame_length
			
			if message.type == "note_on" and message.velocity > 0 and message.note-21 < self.track_nb:
				track_on[message.note-21] = 1
			if message.type == "note_on" and message.velocity == 0 and message.note-21 < self.track_nb:
				track_on[message.note-21] = 0
			
		
		while len(to_return) < n_frame:
			to_return.append(track_on)
			track_on = [0 for i in range(self.track_nb)]
			to_return_time += self.frame_length

			if to_return_time > message_time:
				track_on = [0 for i in range(self.track_nb)]
		
		return np.array(to_return)


def get_filtered_db ():
	ret_db = pd.read_csv(ds_path+"/maestro-v3.0.0.csv")
	# ret_db = ret_db[ret_db["year"]==2004]
	train_db = ret_db[ret_db["split"]=="train"].head(100)

	test_db = ret_db[ret_db["year"]==2004]
	test_db = test_db[test_db["split"]=="test"].head(1)
	
	# ret_db = ret_db.head(5)
	ret_db = pd.concat([train_db, test_db])
	ret_db = ret_db.reset_index(drop=True)
	return ret_db

def setup_piece (audio_filename, midi_filename):
	if os.path.isfile(worked_ds_path+midi_filename+".npy") and os.path.isfile(worked_ds_path+audio_filename+".npy"):
		print("Skipping {}".format(audio_filename))
	else:
		print("Generating {}".format(audio_filename))
		inp, out = generate_train_example(audio_filename=audio_filename, midi_filename=midi_filename, start=0)
		target_dir = os.path.dirname(worked_ds_path+midi_filename)
		os.makedirs(target_dir, exist_ok=True)
		print(inp.shape, out.shape)
		print("Saving {}".format(audio_filename))
		np.save(worked_ds_path+midi_filename+".npy", out.astype(np.int16))
		np.save(worked_ds_path+audio_filename+".npy", inp)

def get_piece (audio_filename, midi_filename):
	inp = np.load(worked_ds_path+audio_filename+".npy")
	out = np.load(worked_ds_path+midi_filename+".npy")
	return inp, out

def setup_piece_raw (inp):
	setup_piece(*inp)

if __name__ == "__main__":
	db = get_filtered_db ().reset_index(drop=True)
	DEBUG = False
	if not DEBUG:
		# for index, row in db.iterrows():
		# 	setup_piece(row["audio_filename"], row["midi_filename"])
		with Pool(7) as p:
			to_feed = [(row["audio_filename"], row["midi_filename"]) for index, row in db.iterrows()]
			p.map(setup_piece_raw, to_feed)


	else:
		pass
		row = db.iloc[2]
		audio_filename = row["audio_filename"]
		midi_filename = row["midi_filename"]

		print(audio_filename)

		mid = MidiFile(ds_path+midi_filename, clip=True)
		print(mid.tracks[0])
		print(mid.tracks[1][:100])
		# print(mid.tracks[1][-300:])

		print(audio_filename)
		exit()

		start = 0
		duration = 10


		ps, onsets = generate_train_example(audio_filename, midi_filename, 20, 5)
		plt.imshow(ps.T, origin="lower")
		plt.show()
		# plt.imshow(ps[1].T, origin="lower")
		# plt.show()
		# plt.imshow(ps[2].T, origin="lower")
		# plt.show()
		plt.imshow(onsets.T, origin="lower")
		plt.show()