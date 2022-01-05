

import torch.optim as optim
import torch as th

import librosa

from networks import Config, EasyNetwork, FramewiseNetwork, FullNetwork

from mido import Message, MetaMessage, MidiFile, MidiTrack

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import os

import loading_utils

import wandb

import scipy.signal

import argparse

device = th.device('cuda' if th.cuda.is_available() else 'cpu')
print('Using device:', device)
print()
if device.type == 'cuda':
	print(th.cuda.get_device_name(0))
	print('Memory Usage:')
	print('Allocated:', round(th.cuda.memory_allocated(0)/1024**3,1), 'GB')
	print('Cached:   ', round(th.cuda.memory_reserved(0)/1024**3,1), 'GB')


class BatchLoader:
	def __init__ (self, split="train"):
		self.db = loading_utils.get_filtered_db().reset_index(drop=True)
		self.db = self.db[self.db["split"]==split]
		self.loaded_inp = [None for i in range(self.db.shape[0])]
		self.loaded_out = [None for i in range(self.db.shape[0])]

	def get_batch (self, batch_size, item_size):
		to_return = []

		n_rows, _ = self.db.shape
		for i in range(batch_size):
			rand_row_id = np.random.randint(n_rows)
			if self.loaded_inp[rand_row_id] is None:
				row = self.db.iloc[rand_row_id]
				inp, out = loading_utils.get_piece(row["audio_filename"], row["midi_filename"])
				self.loaded_inp[rand_row_id] = inp
				self.loaded_out[rand_row_id] = out
			else:
				inp = self.loaded_inp[rand_row_id]
				out = self.loaded_out[rand_row_id]
			# print(inp.shape, out.shape)
			rand_start = np.random.randint(inp.shape[0]-item_size)
			if inp.shape[0] != out.shape[1]:
				print("ai ai ai")
				print(row)
				print(inp.shape, out.shape)
			to_return.append((inp[rand_start:rand_start+item_size], out[:,rand_start:rand_start+item_size]))

		
		inps, outs = zip(*to_return)
		# for i in range(10):
		# 	inp, out = generate_train_example()
		inps = np.expand_dims(np.stack(inps), 1)
		outs = np.stack(outs)
		# print(inps.shape, outs.shape)
		return inps, outs
			
		


def main ():

	parser = argparse.ArgumentParser()
	parser.add_argument('--train', action='store_true')
	parser.add_argument('--test', action='store_true')
	parser.add_argument('--generate', action='store_true')

	args = parser.parse_args()

	if args.train:
		train()
	if args.test:
		test ()
	if args.generate:
		generate ()

	# setup ()
	pass

def train ():


	lr = 0.0005
	bs = 3
	wandb.init(project="my-test-project", entity="oscbout")
	wandb.config = {
		"learning_rate": lr,
		"epochs": 1000,
		"batch_size": bs,
		"network_type": "FramewiseNetwork"
	}

	train_batch_loader = BatchLoader(split="train")
	test_batch_loader = BatchLoader(split="test")
	
	config = Config(False, "results", "simple")
	
	network = FullNetwork().to(device)
	# config.load_model(0, network)

	optimizer = optim.Adam(network.parameters(), lr=lr)
	# all_loss = []
	all_test_loss = []

	n_epoch = 4000
	for epoch in range(n_epoch):
		optimizer.zero_grad()
		# if epoch >= 1000:
		# 	for g in optimizer.param_groups:
		# 		g['lr'] = lr/3

		for sub_epoch in range(1):
			if epoch%1 == 0:
				inp, target = train_batch_loader.get_batch(bs, 512)
				inp_th = th.Tensor(inp).to(device)
				target_th = th.Tensor(target).to(device)

		
			eps = 1e-7
			pred = network(inp_th)
			loss = -th.mean(target_th * th.log(pred+eps) + (1-target_th) * th.log(1-pred+eps), dim=(0,2,3))
			# all_loss.append(loss.cpu().detach().numpy())

			to_log = {"loss": loss[0].cpu().detach().numpy(),
						"frame_loss": loss[1].cpu().detach().numpy()}

			if (epoch+1)%10 == 0:
				test_inp, test_target = test_batch_loader.get_batch(3, 512)
				test_inp_th = th.Tensor(test_inp).to(device)
				test_target_th = th.Tensor(test_target).to(device)

				test_pred = network(test_inp_th)
				test_loss = -th.mean(test_target_th * th.log(test_pred+eps) + (1-test_target_th) * th.log(1-test_pred+eps), dim=(0,2,3))
				to_log["test_loss"] = test_loss[0].cpu().detach().numpy()
				to_log["frame_test_loss"] = test_loss[1].cpu().detach().numpy()
				all_test_loss.append(test_loss.cpu().detach().numpy())

			if (epoch+1)%100 == 0:
				config.save_model(0, network)


			print(loss.cpu().detach().numpy())
			wandb.log(to_log, step=epoch)
			th.sum(loss).backward()
		optimizer.step()
	
	# plt.plot(all_loss)
	# plt.plot(all_test_loss)
	# plt.show()
	config.save_model(0, network)

def test ():

	config = Config(False, "results", "simple")
	
	network = FullNetwork().to(device)
	config.load_model(0, network)

	print(network)

	batch_loader = BatchLoader()
	inp, target = batch_loader.get_batch(2, 512, split="train")
	# inp, target = batch_loader.get_batch(3, 512, split="test")
	inp_th = th.Tensor(inp).to(device)
	res = network(inp_th).cpu().detach().numpy()


	for i in range(inp.shape[0]):
		plt.imshow(inp[i,0].T, origin="lower")
		plt.show()

		plt.imshow(target[i, 0].T, origin="lower")
		plt.show()

		plt.imshow(res[i, 0].T, origin="lower")
		plt.show()

		plt.imshow(target[i,1].T, origin="lower")
		plt.show()

		plt.imshow(res[i,1].T, origin="lower")
		plt.show()
	

def new_midi_file ():
	mid = MidiFile()
	meta_track = MidiTrack()
	mid.tracks.append(meta_track)
	track = MidiTrack()
	mid.tracks.append(track)

	meta_track.append(MetaMessage('set_tempo', tempo=500000, time=0))
	meta_track.append(MetaMessage('time_signature', numerator=4, denominator=4, clocks_per_click=24, notated_32nd_notes_per_beat=8, time=0))
	meta_track.append(MetaMessage('end_of_track', time=1))

	track.append(Message('program_change', channel=0, program=0, time=0))
	# track.append(Message('control_change', channel=0, control=64, value=0, time=0))
	# track.append(Message('control_change', channel=0, control=67, value=124, time=0))

	# track.append(MetaMessage('end_of_track', time=1))
	return mid, track

def generate ():

	# ffmpeg -i my_video.mp4 output_audio.wav
	# ffmpeg -i gamme.mp4 gamme.wav
	# ffmpeg -i chanson.mp4 chanson.wav
	# ffmpeg -i chateau_dans_le_ciel.mp4 chateau_dans_le_ciel.wav

	mid, track = new_midi_file()

	# -------- start --------
	device = th.device('cpu')

	config = Config(False, "results", "simple")
	
	network = FullNetwork().to(device)
	config.load_model(0, network)

	db = loading_utils.get_filtered_db()
	row = db.iloc[2]
	audio_filename = row["audio_filename"]
	midi_filename = row["midi_filename"]

	# audio_filename = "gamme.wav"
	# audio_filename = "chanson.wav"
	audio_filename = "chateau_dans_le_ciel.wav"
	ps, frame_len = loading_utils.generate_spectro(audio_filename, 0, 10, custom_ds_path="audio_perso/")
	print("frame_len : {}".format(frame_len))
	
	plt.imshow(ps.T, origin="lower")
	plt.show()

	ps_th = th.Tensor(ps.reshape((1,1)+ps.shape)).to(device)
	print(ps_th.shape)

	out = network(ps_th).cpu().detach().numpy()
	np.save("out.npy", out)


	out = np.load("out.npy")
	frame_len = 0.022675736961451247
	# out = np.greater(out, 0.5).astype(np.int16)
	onsets = out[0,0]
	frames = out[0,1]


	events = []
	high_frames = np.zeros_like(frames)
	res = np.where(np.greater(frames, 0.9))
	for time, note in zip(*res):
		if high_frames[time, note] == 0:
			cur_time = time
			while cur_time >= 0 and frames[cur_time, note] > 0.3:
				high_frames[cur_time, note] = 1
				cur_time -= 1
			events.append(("onset", note+21, cur_time))
			cur_time = time
			while cur_time < frames.shape[0] and frames[cur_time, note] > 0.3:
				high_frames[cur_time, note] = 1
				cur_time += 1
			events.append(("lift", note+21, cur_time))
	# print(res)
	plt.imshow(high_frames.T, origin="lower")
	plt.show()
	# exit()



	# all_peaks = [scipy.signal.find_peaks(onset_row, height=0.13, distance=4)[0] for onset_row in onsets.T]
	# found_onsets = []
	# for i, peaks in enumerate(all_peaks):
	# 	note = i+21
	# 	found_onsets = found_onsets + [(note, peak_time) for peak_time in peaks]
	# found_onsets = sorted(found_onsets, key=lambda x: x[1])

	# found_lifts = []
	# viz = np.zeros_like(onsets)
	# for note, onset_time in found_onsets:
	# 	cur_time = onset_time+1
	# 	viz[onset_time, note-21] = 1
	# 	while cur_time+1 < frames.shape[0] and frames[cur_time, note-21] > 0.1:
	# 		viz[cur_time, note-21] = 1
	# 		cur_time += 1
	# 	found_lifts.append((note, cur_time))
	
	# events = [("onset", note, frame) for note, frame in found_onsets] + [("lift", note, frame) for note, frame in found_lifts]
	events = sorted(events, key=lambda x: x[2])


	tempo = 500000
	ticks_per_bits = mid.ticks_per_beat
	sec_per_tick = tempo / 1e6 / ticks_per_bits

	last_frame = 0
	for event_type, note, frame in events:
		delay = frame - last_frame
		velocity = 0 if event_type == "lift" else 64
		time = int(delay*frame_len/sec_per_tick)
		track.append(Message('note_on', channel=0, note=note, velocity=velocity, time=time))
		last_frame = frame
	mid.save('new_song.mid')

	print(mid)

	# onset_peaks = np.stack([scipy.signal.find_peaks(onset_row, height=0.2, distance=4) for onset_row in onsets])
	# print(onset_peaks.shape)

	plt.imshow(onsets.T, origin="lower")
	plt.show()

	# plt.imshow(viz.T, origin="lower")
	# plt.show()

	plt.imshow(frames.T, origin="lower")
	plt.show()


if __name__ == "__main__":
	main()

