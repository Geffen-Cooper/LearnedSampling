'''
This file defines a standardized dataset class that is shared across
all the datasets.
'''

import pandas as pd
import os
import numpy as np
import re
from scipy.signal import resample
import argparse
from pathlib import Path
import torch
import pickle
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import matplotlib
import torch.nn.functional as F

class HARClassifierDataset(Dataset):
	""" PyTorch dataset class for HAR data. This is used to train the classifiers

	Parameters
	----------

	dataset_dir: str
		global path to the preprocessed data

	subjects: list (int)
		list of subjects to load data for

	sensors: list (str)
		list of sensors to get channel subset from

	body_parts: list (str)
		list of body parts to get sensor channels from

	activities: list (int)
		list of activities to load

	train: bool
		whether to get the training data

	val: bool
		whether to get the validation data

	val_frac: float
		fraction of training data to segment for validation

	window_size: int
		number of samples per window

	overlap_frac: float
		sliding window overlap fraction for training data

	normalize: bool
		whether to normalize the data
	
	**kwargs:
		makes it easier to pass in args without needing to filter
	"""

	def __init__(self, 
			  dataset_dir: str, 
			  subjects: list, 
			  sensors: list, 
			  body_parts:list , 
			  activities: list,
			  train: bool, 
			  val: bool, 
			  val_frac: float, 
			  window_size: int, 
			  overlap_frac: float = 0.5, 
			  normalize: bool = True,
			  **kwargs):

		dataset_dir = os.path.expanduser(dataset_dir)

		self.dataset_dir = dataset_dir
		self.classifier_training = False

		self.train = train
		self.val = val
		if train:
			print("========= Building Training Dataset =========")
		elif val:
			print("========= Building Val Dataset =========")
		else:
			print("========= Building Test Dataset =========")
		# load the metadata
		with open(os.path.join(dataset_dir,'metadata.pickle'), 'rb') as handle:
			self.dataset_info = pickle.load(handle)
		self.sensor_channel_map = self.dataset_info['sensor_channel_map']
		label_map = self.dataset_info['label_map']

		# determine which channels to use (keep relative channel order of original data)
		self.active_channels = []
		for sensor in sensors:
			for bp in body_parts:
				self.active_channels.append(self.sensor_channel_map[bp][sensor])
		self.active_channels = np.sort(np.concatenate(self.active_channels))
		print(f"Body Parts: {body_parts}")
		print(f"Sensors:  {sensors}")
		print(f"Active Channels: {self.active_channels}")


		# load the raw data (keep relative order of original labels)
		prefix = f"{dataset_dir}/"
		self.subjects = subjects
		print(f"Subjects: {subjects}")

		self.raw_data = {subject: [] for subject in subjects}
		self.raw_labels = {subject: [] for subject in subjects}
		self.window_idxs = {subject: [] for subject in subjects}

		for subject in self.subjects:
			self.raw_data[subject] = np.load(f"{prefix}data_{subject}.npy")[:,self.active_channels] # (n,ch)
			self.raw_labels[subject] = np.load(f"{prefix}labels_{subject}.npy") # (n)
			self.window_size = window_size
			if window_size == 0:
				self.window_idxs[subject] = np.load(f"{prefix}windows_{subject}.npy") # (n,2)


		# filter out the selected activities and do train-val split
		activities = np.sort(np.array(activities))
		print(f"Activities: {activities}")
		self.selected_activity_label_map = { 
			class_idx : label_map[activity_idx] for class_idx, activity_idx in enumerate(activities)
		}
		print(f"Label Map: {self.selected_activity_label_map}")

		label_swap = {activity_idx : class_idx for class_idx, activity_idx in enumerate(activities)}
		idx_filters = {subject: [] for subject in subjects}

		# for data we will window using uniform window
		if window_size > 0:
			for subject in subjects:
				# remove data and labels we don't want 
				idxs_to_keep = []
				for activity_idx in activities:
					# keep idxs from selected activity
					idxs = (self.raw_labels[subject] == activity_idx).nonzero()[0]
					# if train or val, then segment, otherwise if test keep all
					train_len = int(len(idxs)*(1-val_frac))
					if train == True:
						idxs = idxs[:train_len]
					elif val == True:
						idxs = idxs[train_len:]
					idxs_to_keep.append(idxs)
				# merge across activities
				idxs_to_keep = np.concatenate(idxs_to_keep)
				idx_filters[subject] = idxs_to_keep
				
				self.raw_data[subject] = self.raw_data[subject][np.sort(idxs_to_keep),:]
				self.raw_labels[subject] = self.raw_labels[subject][np.sort(idxs_to_keep)]
				
				# realign labels to class idxs
				for activity_idx in activities:
					idxs_to_swap = (self.raw_labels[subject] == activity_idx).nonzero()[0]
					self.raw_labels[subject][idxs_to_swap] = label_swap[activity_idx]
		# for variable sized predefined windows
		else:
			for subject in subjects:
				# remove data and labels we don't want 
				idxs_to_keep = []
				windows_to_keep = []
				for activity_idx in activities:
					# we need to get the windows for this activity

					# keep idxs from selected activity
					idxs = (self.raw_labels[subject] == activity_idx).nonzero()[0]

					# get the idx of the first and last window for this activity
					first_win = (self.window_idxs[subject][:,0] == min(idxs)).nonzero()[0][0] # idx into windows
					last_win = (self.window_idxs[subject][:,1] == (max(idxs)+1)).nonzero()[0][0]

					# if train or val, then segment, otherwise if test keep all
					train_len = int((last_win-first_win+1)*(1-val_frac)) # number of windows
					if train == True:
						# gets idxs of train windows, then get idx of start of first window to idx of end of last window
						train_windows = np.arange(first_win,last_win+1)[:train_len]
						idxs = np.arange(self.window_idxs[subject][train_windows[0]][0], self.window_idxs[subject][train_windows[-1]][1])

						# only keep these window_idxs
						windows_to_keep.append(train_windows)
					elif val == True:
						val_windows = np.arange(first_win,last_win+1)[train_len:]
						idxs = np.arange(self.window_idxs[subject][val_windows[0]][0], self.window_idxs[subject][val_windows[-1]][1])

						# only keep these window_idxs
						windows_to_keep.append(val_windows)
					else: # test
						windows_to_keep.append(np.arange(first_win,last_win+1))
					idxs_to_keep.append(idxs)
				# merge across activities
				idxs_to_keep = np.concatenate(idxs_to_keep)
				windows_to_keep = np.concatenate(windows_to_keep)
				
				self.raw_data[subject] = self.raw_data[subject][np.sort(idxs_to_keep),:]
				self.raw_labels[subject] = self.raw_labels[subject][np.sort(idxs_to_keep)]
				self.window_idxs[subject] = self.window_idxs[subject][np.sort(windows_to_keep)]
				
				# realign labels to class idxs
				for activity_idx in activities:
					idxs_to_swap = (self.raw_labels[subject] == activity_idx).nonzero()[0]
					self.raw_labels[subject][idxs_to_swap] = label_swap[activity_idx]

  
		# normalize
		all_data = np.concatenate(list(self.raw_data.values()))
		if train == True:
			self.mean = np.mean(all_data, axis=0)
			self.std = np.std(all_data, axis=0)
			np.save(os.path.join(dataset_dir,"training_data_mean"),self.mean)
			np.save(os.path.join(dataset_dir,"training_data_std"),self.std)
		else:
			self.mean = np.load(os.path.join(dataset_dir,"training_data_mean.npy"))
			self.std = np.load(os.path.join(dataset_dir,"training_data_std.npy"))
		# apply training mean/std to train/val/test data
		if normalize:
			for subject in self.subjects:
				self.raw_data[subject] = (self.raw_data[subject]-self.mean)/(self.std + 1e-5)

		# create windows, for test data we do dense prediction on every sample
		if train or val:
			stride = int(window_size*(1-overlap_frac))
		else:
			stride = 1

		self.window_labels = {subject: [] for subject in subjects}

		# preloaded windows (gesture recognition)
		if window_size == 0:
			for subject in self.subjects:
				# need to reindex the windows
				window_sizes = np.diff(self.window_idxs[subject],axis=1).flatten()
				new_windows = np.zeros_like(self.window_idxs[subject])
				window_labels = []

				new_windows[0,0] = 0
				new_windows[0,1] = window_sizes[0]
				window_labels.append(self.raw_labels[subject][new_windows[0,0]])

				for i in range(1,len(window_sizes)):
					new_windows[i,0] = new_windows[i-1,1] # window start is end of last window
					new_windows[i,1] = new_windows[i,0] + window_sizes[i] # window end just increments by size
					window_labels.append(self.raw_labels[subject][new_windows[i,0]])

				self.window_idxs[subject] = new_windows
				self.window_labels[subject] = np.array(window_labels)
		else:
			self.window_idxs = {subject: [] for subject in subjects}
			for subject in subjects:
				idxs, labels = self.create_windows(self.raw_data[subject],self.raw_labels[subject],window_size,stride)
				self.window_idxs[subject] = idxs
				self.window_labels[subject] = labels

		self.budget = kwargs['budget']
		self.policy = kwargs['sampling_policy']

	@staticmethod
	def create_windows(data: np.ndarray, labels: np.ndarray, window_size: int, stride: int):
		""" Partitions the raw data into windows.

		Parameters
		----------

		data: np.ndarray
			data array of dimension (L x C) where L is the time dimension

		labels: np.ndarray
			label array of dimension L where L is the time dimension

		window_size: int
			number of samples per window

		stride: int
			how much overlap per window in units of samples

		Returns
		-------

		window_idxs: np.ndarray
			an (N x 2) array of the starting and ending idx for each window
			where N is the number of windows
		
		window_labels: np.ndarray
			an array of length N of the labels for each window where N is
			the number of windows
		"""

		# form windows
		start_idxs = np.arange(0,data.shape[0]-window_size,stride)
		end_idxs = start_idxs + window_size

		window_labels = labels[end_idxs-1] # last label in window

		return np.stack([start_idxs,end_idxs]).T, window_labels

	def __getitem__(self, idx):
		# index into subject then window
		# e.g. [[0,1000], [1000,2000]], if idx = 1200 then count is 1000 so idx becomes 200

		# need to double check how subejct data aligns with windows...
		count = 0
		for subject, subject_windows in self.window_idxs.items():
			count += subject_windows.shape[0]
			if count > idx:
				count -= subject_windows.shape[0]
				break
		
		idx = idx - count

		# get the window idxs
		start,end = self.window_idxs[subject][idx]
		
		# get the data window
		X = self.raw_data[subject][start:end,:]

		if self.policy == "learned":
			budget = self.budget
			Y = self.window_labels[subject][idx]

			# if training the policy, need to find the label to map h_i to skip
			if self.classifier_training == False:
				h,y,time_step = find_best_skip(self.model,X,Y,self.skip_i,budget)
				rem_budget = budget - self.skip_i
				policy_input = torch.cat([h,torch.tensor([time_step]),torch.tensor([rem_budget])])
			
			if self.train == True:
				# print(f"============={self.mean},{self.std}")
				# exit()
				if self.classifier_training == True:
					return torch.tensor(X).float(), torch.tensor(Y).long()
				else:
					return policy_input.float(), y.long()
			else: # for validation, we also want to see classification result (we will forward pass with X using current policy)
				if self.classifier_training == True:
					return torch.tensor(X).float(), torch.tensor(Y).long()
				else:
					return policy_input.float(), torch.tensor(X).float(), y.long(), torch.tensor(Y).long()


		# ------------ random
		if self.policy == "random":
			budget = self.budget
			seq_len = X.shape[0]
			rand_idxs = torch.sort(torch.randperm(seq_len)[:budget])[0]
			X = X[rand_idxs,:]
			X = np.column_stack([rand_idxs,X])
		# ------------ random

		# ------------ subsample
		if "uniform_subsampling" in self.policy:
			budget = self.budget
			seq_len = X.shape[0]
			seq_len = 21 # avg seq length since don't know ahead of time
			if "impair" in self.dataset_dir:
				seq_len = 21 # avg sequence length after downsampling
			delay = seq_len // budget
			if delay == 0:
				delay = 1

			idxs = torch.arange(seq_len)[::delay][:budget]

			# if the sequence happens to be shorter than average
			# make sure to not exceed
			if X.shape[0] < seq_len:
				valid_idxs = (idxs < X.shape[0]).nonzero().flatten()
				if len(valid_idxs) == 1:
					idxs = torch.tensor([idxs[0],X.shape[0]-1])
				else:
					idxs = idxs[valid_idxs]
			X = X[idxs,:]
			X = np.column_stack([idxs,X])
		# ------------ subsample

		if self.policy == "dense":
			seq_len = X.shape[0]
			idxs = torch.arange(seq_len)
			X = np.column_stack([idxs,X])

		# idxs = torch.sort(torch.randperm(X.shape[0])[::4])[0]
		# X = X[idxs,:]
		# X = np.column_stack([indices[idxs],X])

		# get the label
		try:
			Y = self.window_labels[subject][idx]
		except:
			print(idx+count,count,idx)
			print(self.window_labels[subject],self.raw_labels[subject].shape)
			exit()

		# return the sample and the class
		return torch.tensor(X).float(), torch.tensor(Y).long()

	def __len__(self):
		if self.window_size != 0:
			return sum([len(wl) for wl in self.window_labels.values()])
		else:
			# if windows predetermined
			return sum([len(wl) for wl in self.window_idxs.values()])

	def visualize_batch(self,body_part,sensor):
		matplotlib.rcParams.update({'font.size': 6})
		idxs = torch.randperm(len(self))[:16]
		fig,ax = plt.subplots(4,4,figsize=(9,5))
		fig.subplots_adjust(wspace=0.6,hspace=1)
		body_part_idxs = self.sensor_channel_map[body_part][sensor]
		for i,idx in enumerate(idxs):
			bp_0 = np.where(self.active_channels == body_part_idxs[0])[0][0]
			bp_1 = np.where(self.active_channels == body_part_idxs[1])[0][0]
			bp_2 = np.where(self.active_channels == body_part_idxs[2])[0][0]
			sensor_data,l = self.__getitem__(idx)
			x = sensor_data[:,bp_0]
			y = sensor_data[:,bp_1]
			z = sensor_data[:,bp_2]

			i_x = i % 4
			i_y = i // 4
			x_ = np.arange(x.shape[0])
			ax[i_y,i_x].plot(x_,x,label='X')
			ax[i_y,i_x].plot(x_,y,label='Y')
			ax[i_y,i_x].plot(x_,z,label='Z')
			ax[i_y,i_x].set_xlabel("Sample #")
			ax[i_y,i_x].set_ylabel("Value")
			ax[i_y,i_x].set_title(self.selected_activity_label_map[int(l)])
			ax[i_y,i_x].grid()
			ax[i_y,i_x].set_ylim([-2.5,2.5])
		plt.savefig("viz.png")


# this function will be called by __getitem__ with a given X
# the function will first execute the trained policy until skip_i (retrieves prepended X and h_i)
# then it will roll out several Xs based on the different skip options
# for each [X_pre, X_roll], it will get Y_hat and determine the one with highest confidence on Y
# it will then return h_i, skip_label
def find_best_skip(model, X, Y, skip_i, budget):
	X = torch.tensor(X)
	with torch.no_grad():
		model.eval()
		# execute policy up to skip_i
		h_i = torch.zeros((1,1,model.hidden_size)) # batch size 1, one layer
		c_i = torch.zeros((1,1,model.hidden_size)) # batch size 1, one layer
		idx = 0
		# for skip_i=0 there is no pre because we don't observe anything yet
		X_pre = torch.zeros_like(X)[:skip_i,:] # X_pre is number of observations observed before next skip
		time_idxs = torch.zeros(budget)
		seq_len = X.shape[0]
		skips = []
		pre_budget = budget
		# does not get executed for skip_0 since we need to learn the first skip
		for i in range(skip_i):
			# policy forward pass, map this to a skip amount
			p_in = torch.cat([h_i[0][0],torch.tensor([idx]),torch.tensor([pre_budget])])
			skip_pred = model.other_forward(policy_input=p_in.unsqueeze(0),mode="PolicyForward")
			skip = model.skip_map[torch.argmax(skip_pred).item()]
			skips.append(skip)
			# observe input at this skipped value
			idx += skip
			pre_budget -= 1
			# if the predicted skip goes beyond, then we stop here
			# don't apply this skip, and use the current point as our sample
			if idx >= seq_len:
				idx -= skip
				skip_i = i # if skip_i=3 and we go over on i=1, then skip_i=1
				X_pre = X_pre[:i,:] # truncate the pre
				i -= 1 # may be wrong
				pre_budget += 1
				break
			time_idxs[i] = idx
			X_pre[i,:] = X[idx,:]

			# update the hidden state with next observation
			X_new = torch.cat([torch.tensor([idx]),X_pre[i,:]])
			h_i, c_i = model.other_forward(x=X_new.unsqueeze(0).unsqueeze(0).float(), h=h_i, c=c_i, mode="IterateForward")

		# with the remaining budget, observe a skip forward and then rollout the rest of the policy
		trajectories = []
		rem_budget = budget - skip_i - 1 # subtract how much we observed (skip_i) plus the current observation (skip_amt)
		preds = torch.zeros(model.skip_options,model.output_size)
		for tr_i,skip_amt in enumerate(model.skip_map.values()):
			'''add checks here
			-if rem_len is <=0, then we have no rollout and we went over the edge of sequence
				-in this case we did not fully utilize the budget so we need to truncate time_idxs and X_traj
			-if rem_len after observations is less than budget, just observe 1 by 1 until end and then truncate
			'''
			rem_len = seq_len - idx - skip_amt
			next_idx = idx + skip_amt
			curr_budget = rem_budget
			# if we try to skip past the end, we do not acquire any new sample
			if rem_len <= 0:
				# print(skip_amt,"-")
				# if we haven't observed anything yet, observe the start
				if skip_i == 0:
					time_idxs = torch.tensor([idx])
					X_traj = X[idx,:]
					trajectories.append(torch.cat([time_idxs,X_traj]))
				else:# otherwise, we don't observe anything else and use the pre
					time_idxs = time_idxs[:i+1]
					X_traj = X_pre
					if X_traj.shape[0] == 1: # one time step
						# print(time_idxs,X_traj)
						trajectories.append(torch.cat([time_idxs.unsqueeze(0),X_traj],dim=1))
					else:
						trajectories.append(torch.cat([time_idxs.unsqueeze(1),X_traj],dim=1))
				hid_state = h_i.clone() # copy it because h_i will be returned
				cid_state = c_i.clone() # copy it because need to do this for every skip_amt
			# else: # forward step with the skip and then rollout with existing policy
			# 	hid_state = h_i.clone() # copy it because h_i will be returned
			# 	cid_state = c_i.clone() # copy it because need to do this for every skip_amt

			# 	# update the hidden state with next observation from skip
			# 	X_new = torch.cat([torch.tensor([next_idx]),X[next_idx,:]])
			# 	hid_state, cid_state = model.other_forward(x=X_new.unsqueeze(0).unsqueeze(0).float(), h=hid_state, c=cid_state, mode="IterateForward")

			# 	# now rollout the rest of the policy with the remaining budget
			# 	while curr_budget > 0:
			# 		# policy forward pass, map this to a skip amount
			# 		p_in = torch.cat([hid_state[0][0],torch.tensor([next_idx]),torch.tensor([curr_budget])])
			# 		skip_pred = model.other_forward(policy_input=p_in.unsqueeze(0),mode="PolicyForward")
			# 		skip = model.skip_map[torch.argmax(skip_pred).item()]
			# 		skips.append(skip)
			# 		# observe input at this skipped value
			# 		next_idx += skip
			# 		# if the predicted skip goes beyond, then we stop here
			# 		if next_idx >= seq_len:
			# 			break

			# 		# update the hidden state with next observation
			# 		X_new = torch.cat([torch.tensor([next_idx]),X[next_idx,:]])
			# 		hid_state, cid_state = model.other_forward(x=X_new.unsqueeze(0).unsqueeze(0).float(), h=hid_state, c=cid_state, mode="IterateForward")

			# 		curr_budget -= 1
			
			# preds[tr_i,:] = F.softmax(model.fc(hid_state)[0],dim=1)[0]

			# =============== Uniform Rollout ================
			# if the amount left to observe from is less than the budget, just observe it all
			elif (rem_len-1) < rem_budget:
				# print(skip_amt,"--")
				rollout_idxs = torch.arange(idx+skip_amt+1,seq_len)
				if skip_i == 0:
					# print(skip_amt,torch.tensor([idx+skip_amt]),rollout_idxs)
					time_idxs = torch.cat([torch.tensor([idx+skip_amt]),rollout_idxs])
				else:
					# print(skip_i, i,time_idxs,rollout_idxs,skip_amt,idx,seq_len)
					time_idxs = torch.cat([time_idxs[:i+1],torch.tensor([idx+skip_amt]),rollout_idxs])
				
				X_traj = torch.vstack([X_pre,X[idx+skip_amt,:],X[rollout_idxs,:]])
				trajectories.append(torch.cat([time_idxs.unsqueeze(1),X_traj],dim=1))
			else:
				# print(skip_amt,"---")
				if rem_budget > 0:
					delay = (rem_len-1) // rem_budget # remaining length after observations
					rollout_idxs = torch.arange(idx+skip_amt,seq_len)[delay::delay][:rem_budget]
				else:
					rollout_idxs = torch.tensor([]) # no rollout if no budget left
				
				if skip_i == 0:
					# print(skip_amt,torch.tensor([idx+skip_amt]),rollout_idxs)
					time_idxs = torch.cat([torch.tensor([idx+skip_amt]),rollout_idxs])
				else:
					time_idxs = torch.cat([time_idxs[:i+1],torch.tensor([idx+skip_amt]),rollout_idxs])
				
				if rem_budget > 0:
					X_traj = torch.vstack([X_pre,X[idx+skip_amt,:],X[rollout_idxs,:]])
				else:
					X_traj = torch.vstack([X_pre,X[idx+skip_amt,:]])
				try:
					trajectories.append(torch.cat([time_idxs.unsqueeze(1),X_traj],dim=1))
				except:
					print(time_idxs.shape,X_traj.shape,time_idxs,X_traj,i,skip_i,rollout_idxs,idx,skip_amt,skips)
					exit()

		# print(trajectories)
		preds = torch.zeros(model.skip_options,model.output_size)
		for traj_i,traj in enumerate(trajectories):
			# print(traj.float().unsqueeze(0).shape)
			# print(f"time_idxs:{time_idxs}\nrollout_idxs: {rollout_idxs}\nseq_len: {seq_len}\nidx:{idx}\nskip_amt:{skip_amt}\nrem_len:{rem_len}\nrem_budget:{rem_budget}\ndelay:{delay}")
			if len(traj.shape) == 1:
				traj = traj.unsqueeze(0)
			out = model.other_forward(x=traj.float().unsqueeze(0),mode='SimpleForward')
			preds[traj_i,:] = F.softmax(out[0],dim=1)[0]
		
		best_skip = torch.argmax(preds[:,int(Y)])
		# print(h_i[0][0], best_skip)
		# exit()

		return h_i[0][0], best_skip, idx



from torch.nn.utils.rnn import pad_sequence

def rnn_collate_fn(batch):
	# batch is a list of (sequence_tensor, label)
	
	sequences, labels = zip(*batch)
	# for i,seq in enumerate(sequences):
	# 	print(i,seq.shape,labels[i])
	# exit()

	# Pad sequences
	padded = pad_sequence(sequences, batch_first=True)  # shape: (batch, max_len, 3)

	# Get original lengths
	lengths = torch.tensor([seq.shape[0] for seq in sequences])

	labels = torch.tensor(labels)

	return (padded, lengths), labels

def val_collate_fn(batch):
	# batch is a list of (sequence_tensor, label)
	
	p_in, X, y_skip, Y = zip(*batch)

	# Pad sequences
	padded = pad_sequence(X, batch_first=True)  # shape: (batch, max_len, 3)

	# Get original lengths
	lengths = torch.tensor([seq.shape[0] for seq in X])

	p_in = torch.stack(p_in,dim=0)
	y_skip = torch.stack(y_skip)
	Y = torch.stack(Y)

	return (p_in, padded, lengths), (y_skip, Y)

def load_har_classifier_dataloaders(train_subjects, test_subjects, **kwargs):
	""" Creates train, val, and test dataloaders for HAR classifiers.

		Parameters
		----------

		train_subjects: list (int)
			list of subjects for training
		
		test_subjects: list (int)
			list of subjects for testing
		
		**kwargs:
			parameters used for the dataset (e.g. batch_size, body parts, subjects, etc.)

		Returns
		-------

		train_loader: Dataloader
			PyTorch dataloader for training
		val_loader: Dataloader
			PyTorch dataloader for validation
		test_loader: Dataloader
			PyTorch dataloader for testing
		"""

	batch_size = kwargs['batch_size']
	kwargs['subjects'] = train_subjects
	train_ds = HARClassifierDataset(**kwargs,train=True,val=False)
	val_ds = HARClassifierDataset(**kwargs,train=False,val=True)
	kwargs['subjects'] = test_subjects
	test_ds = HARClassifierDataset(**kwargs,train=False,val=False)

	if 'rnn' in kwargs['architecture']:
		train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=False,drop_last=True,num_workers=4,collate_fn=rnn_collate_fn)
		val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=False,drop_last=True,num_workers=4,collate_fn=rnn_collate_fn)
		test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=False,drop_last=True,num_workers=4,collate_fn=rnn_collate_fn)
	else:
		train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=False,drop_last=True,num_workers=4)
		val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=False,drop_last=True,num_workers=4)
		test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=False,drop_last=True,num_workers=4)

	return train_loader, val_loader, test_loader


def generate_activity_sequence(data,labels,min_duration,max_duration,sampling_rate):#, seed):
	# np.random.seed(seed)
	# first make contiguous segments
	contig_labels = np.zeros_like(labels)
	contig_data = np.zeros_like(data)
	counter = 0
	for act in np.unique(labels):
		idxs = (labels == act).nonzero()[0]
		contig_labels[counter:counter+len(idxs)] = labels[idxs]
		contig_data[counter:counter+len(idxs),:] = data[idxs,:]
		counter += len(idxs)

	activity_idxs = {i : (contig_labels == i).nonzero()[0] for i in np.unique(contig_labels)}
	duration = np.arange(min_duration,max_duration+1)

	X = np.zeros_like(contig_data)
	y = np.zeros_like(contig_labels)

	activity_counters = {act: 0 for act in np.unique(contig_labels)}
	remaining_activities = list(np.unique(contig_labels))
	sample_counter = 0

	while len(remaining_activities) > 0:
		# randomly sample an activity
		act = int(np.random.choice(np.array(remaining_activities), 1)[0])

		# randomly sample a duration
		dur = np.random.choice(duration, 1)[0]

		# access this chunk of data and add to sequence
		start = int(activity_counters[act])
		end = int(start + dur*sampling_rate)

		activity_counters[act] += (end-start)

		# check if hit end
		if end >= activity_idxs[act].shape[0]:
			end = int(activity_idxs[act].shape[0])-1
			remaining_activities.remove(act)

		start = activity_idxs[act][start]
		end = activity_idxs[act][end]

		X[sample_counter:sample_counter+end-start,:] = contig_data[start:end,:]
		y[sample_counter:sample_counter+end-start] = contig_labels[start:end]
		sample_counter += (end-start)

	return X,y