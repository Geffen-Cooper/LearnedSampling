'''
This file has dataset specific functions to load and preprocess the data from
each dataset into a standard format (per subject numpy arrays). Preprocessing
consists of: loading the raw data, dealing with nan, and resampling to 25Hz
'''

import pandas as pd
import os
import numpy as np
import re
from scipy.signal import resample
import pickle
import xml.etree.ElementTree as ET



def find_closest_index(original_array: np.ndarray , new_array: np.ndarray) -> np.ndarray:
	""" When resampling data to a lower sampling rate, the sample level labels also need to be adjusted.
		During resampling, we will get fewer samples so the corresponding time stamps get adjusted.
		To adjust the sample level labels, this function gets the index at the closest time stamp
		in the original array.

	Parameters
	----------

	original_array: np.ndarray
		an array of time stamps for each sample in the orignal data

	new_array: np.ndarray
		an array of time stamps for each sample in the new data (will be shorter if lower sampling rate)


	Returns
	-------

	closest_indices: np.ndarray
		the indices in the orignal label array to access to set the labels for the new sampling rate
	"""

	# find indices of orignal where samples of new array should be inserted to preserve order
	indices = np.searchsorted(original_array, new_array)
	indices = np.clip(indices, 1, len(original_array)-1)
	
	# consider indices 0->n-1 and 1->n
	left_values = original_array[indices - 1]
	right_values = original_array[indices]
	
	# get index with closest timestamp (new array timestamps fall between old array timestamps so need to choose closest)
	closest_indices = np.where(np.abs(new_array - left_values) < np.abs(new_array - right_values),
							   indices - 1,
							   indices)
	
	return closest_indices

def preprocess_DSADS(dataset_dir: str) -> dict:
	""" Loads the DSADS raw data and saves it in a standard format.

	https://archive.ics.uci.edu/dataset/256/daily+and+sports+activities

	Each subject's data and labels will be saved as data_[subject].npy and labels_[subject].npy
	in dataset_dir/preprocessed/. The data will have shape (N x C) where N is the number
	of raw samples per subject and C is the number of sensor channels.

	Parameters
	----------

	dataset_dir: str
		global path of where the dataset has been installed.


	Returns
	-------

	dataset_info: dict
		metadata about the raw data, specifically the 
		sensor channel map, list of subjects, and label map
	"""

	dataset_dir = os.path.expanduser(dataset_dir)
	
	# DSADS directory structure is a01/p1/s01.txt (activity, subject, segment)
	activity_folders = os.listdir(dataset_dir)

	# Filter folder names that match the structure 'a' followed by exactly two digits
	activity_folders = [folder for folder in activity_folders if re.match(r'^a\d{2}$', folder)]
	activity_folders.sort(key=lambda f: int(re.sub('\D', '', f)))

	subject_folders = os.listdir(os.path.join(dataset_dir,activity_folders[0]))
	subject_folders.sort(key=lambda f: int(re.sub('\D', '', f)))
	NUM_SUBJECTS = len(subject_folders)

	segment_files = os.listdir(os.path.join(dataset_dir,activity_folders[0],subject_folders[0]))
	segment_files.sort(key=lambda f: int(re.sub('\D', '', f)))
	NUM_SEGMENTS = len(segment_files)
	SEGMENT_LEN,NUM_CHANNELS = pd.read_csv(os.path.join(dataset_dir,activity_folders[0],subject_folders[0],segment_files[0]),header=None).values.shape
	num_samples_per_activity = NUM_SEGMENTS*SEGMENT_LEN

	# we separate the data by subject
	training_data = {subject: [] for subject in range(NUM_SUBJECTS)} # raw data
	training_labels = {subject: [] for subject in range(NUM_SUBJECTS)} # raw labels

	# merge data for each subject into a numpy array
	for subject_i, subject_folder in enumerate(subject_folders):
		for activity_i, activity_folder in enumerate(activity_folders):
			# create the data array which contains samples across all segment files
			data_array = np.zeros((num_samples_per_activity,NUM_CHANNELS))
			label_array = np.zeros(num_samples_per_activity)
			for segment_i, segment_file in enumerate(segment_files):
				data_file_path = os.path.join(dataset_dir,activity_folder,subject_folder,segment_file)
				data_segment = pd.read_csv(data_file_path,header=None).values
				start = segment_i*SEGMENT_LEN
				end = start + SEGMENT_LEN
				data_array[start:end,:] = data_segment[:,:]
				label_array[start:end] = activity_i

			# put into list
			training_data[subject_i].append(data_array)
			training_labels[subject_i].append(label_array)
		
	# now concatenate and save data
	output_folder = os.path.join(dataset_dir,"preprocessed_data")
	os.makedirs(output_folder,exist_ok=True)
	for subject_i in range(NUM_SUBJECTS):
		training_data[subject_i] = np.concatenate(training_data[subject_i])
		training_labels[subject_i] = np.concatenate(training_labels[subject_i])

		np.save(os.path.join(output_folder,f"data_{subject_i+1}"),training_data[subject_i])
		np.save(os.path.join(output_folder,f"labels_{subject_i+1}"),training_labels[subject_i])


	# ------------- dataset metadata -------------
	body_parts = ['torso','right_arm','left_arm','right_leg','left_leg']
	sensors = ['acc','gyro','mag']
	sensor_dims = 3 # XYZ
	channels_per_sensor = len(sensors)*sensor_dims

	# dict to get index of sensor channel by bp and sensor
	sensor_channel_map = {
		bp: 
		{
			sensor: np.arange(bp_i*channels_per_sensor+sensor_i*sensor_dims,
							bp_i*channels_per_sensor+sensor_i*sensor_dims+sensor_dims)
					for sensor_i,sensor in enumerate(sensors)
		} for bp_i,bp in enumerate(body_parts)
	}

	label_map = {
			0:'sitting',
			1:'standing',
			2:'lying on back',
			3:'lying on right side',
			4:'ascending stairs',
			5:'descending stairs',
			6:'standing in elevator',
			7:'moving in elevator',
			8:'walking in parking lot',
			9:'walking on flat treadmill',
			10:'walking on inclined treadmill',
			11:'running on treadmill,',
			12:'exercising on stepper',
			13:'exercising on cross trainer',
			14:'cycling on exercise bike horizontal',
			15:'cycling on exercise bike vertical',
			16:'rowing',
			17:'jumping',
			18:'playing basketball'
			}
	
	dataset_info = {
		'sensor_channel_map': sensor_channel_map,
		'list_of_subjects': [subject_i+1 for subject_i in range(NUM_SUBJECTS)],
		'label_map': label_map
	}
	
	with open(os.path.join(output_folder,"metadata.pickle"), 'wb') as file:
		pickle.dump(dataset_info, file)


def preprocess_RWHAR(dataset_dir: str) -> dict:
	""" Loads the RWHAR raw data and saves it in a standard format.

	http://wifo5-14.informatik.uni-mannheim.de/sensor/dataset/realworld2016/realworld2016_dataset

	Each subject's data and labels will be saved as data_[subject].npy and labels_[subject].npy
	in dataset_dir/preprocessed/. The data will have shape (N x C) where N is the number
	of raw samples per subject and C is the number of sensor channels.

	Parameters
	----------

	dataset_dir: str
		global path of where the dataset has been installed.


	Returns
	-------

	dataset_info: dict
		metadata about the raw data, specifically the 
		sensor channel map, list of subjects, and label map
	"""

	dataset_dir = os.path.expanduser(dataset_dir)

	label_map = {
			0:'climbingdown',
			1:'climbingup',
			2:'jumping',
			3:'lying',
			4:'running',
			5:'sitting',
			6:'standing',
			7:'walking'
			}

	body_parts = ['chest',
			  'forearm',
			  'head',
			  'shin',
			  'thigh',
			  'upperarm',
			  'waist']

	og_sampling_rate = 50
	new_sampling_rate = 25

	activities = label_map.values()

	# RWHAR directory structure is proband1/data/acc_jumping_head.csv
	subject_folders = os.listdir(dataset_dir)

	# Filter folder names that match the structure 'proband' followed by exactly two digits
	subject_folders = [folder for folder in subject_folders if 'proband' in folder]
	subject_folders.sort(key=lambda f: int(re.sub('\D', '', f)))
	NUM_SUBJECTS = len(subject_folders)

	file_count = {sf:0 for sf in subject_folders}
	active_subjects = []

	# first filter out subjects which have missing data
	for subject_i,subject_folder in enumerate(subject_folders):
		# for a given subject, get all activity csvs (8 activities, 7 body parts)
		activity_csvs = os.listdir(os.path.join(dataset_dir,subject_folder,'data'))
		activity_csvs.sort()
		print(subject_folder)

		# keep a dict of activity files present
		act_bp_dict = {act:{} for act in activities}
		for k in act_bp_dict.keys():
			for bp in body_parts:
				act_bp_dict[k][bp] = False

		# iterate over activity files
		for activity_csv in activity_csvs:
			if activity_csv.endswith(".csv"):
				# determine the label and body part
				activity_str = activity_csv.split("_")[1]
				body_part = activity_csv.split("_")[2]
				if body_part.isdigit(): # some files are split into parts
					body_part = activity_csv.split("_")[3]
				file_count[subject_folder] += 1
				body_part = body_part[:-4]
				act_bp_dict[activity_str][body_part] = True

		# if have less than 8*7 True values, then data is missing
		count = 0
		for act in act_bp_dict.keys():
			for bp in act_bp_dict[act]:
				if act_bp_dict[act][bp] == True:
					count += 1
				else:
					print(f"subject {subject_folder} is missing {act}-{bp}")
		if count == len(body_parts)*len(activities):
			active_subjects.append(subject_i)
	print(f"active_subjects (idxs): {active_subjects}")

	# we separate the data by subject
	training_data = {subject: [] for subject in range(NUM_SUBJECTS)} # raw data
	training_labels = {subject: [] for subject in range(NUM_SUBJECTS)} # raw labels


	# then load all the data
	for subject_i,subject_folder in enumerate(subject_folders):
		if subject_i not in active_subjects:
			print(f"Skipping {subject_folder}")
			continue

		# keep a dict of activity data present
		act_bp_dict = {act:{} for act in activities}
		for k in act_bp_dict.keys():
			for bp in body_parts:
				act_bp_dict[k][bp] = []

		# for a given subject, get all activity csvs (8 activities, 7 body parts)
		activity_csvs = os.listdir(os.path.join(dataset_dir,subject_folder,'data'))
		activity_csvs.sort()
		print(subject_folder)

		# for each activity, get all body part csvs
		for activity_i,activity in enumerate(activities):
			prefix = f"acc_{activity}_"
			for activity_csv in activity_csvs:
				if activity_csv.startswith(prefix) and activity_csv.endswith(".csv"):
					activity_str = activity_csv.split("_")[1]
					body_part = activity_csv.split("_")[2]
					if body_part.isdigit(): # some files are split into parts
						body_part = activity_csv.split("_")[3]
					# load the data for every body part
					file_path = os.path.join(dataset_dir,subject_folder,'data',activity_csv)
					print(f"{activity_str}-{body_part[:-4]}")
					# filter start and end segments with no activity, don't need id from csv
					if activity_str == 'jumping':
						act_bp_dict[activity_str][body_part[:-4]].append(pd.read_csv(file_path).values[100:,1:])
					else:
						act_bp_dict[activity_str][body_part[:-4]].append(pd.read_csv(file_path).values[3*100:-3*100,1:])

		# first merge csvs for activities that got split into segments
		for act in act_bp_dict.keys():
			for bp in act_bp_dict[act]:
				if len(act_bp_dict[act][bp]) > 1:
					data_arrays = act_bp_dict[act][bp]
					act_bp_dict[act][bp] = np.concatenate(data_arrays,axis=0)
				else:
					# no more list
					act_bp_dict[act][bp] = act_bp_dict[act][bp][0]

		# now try to temporally align data across body parts as best as possible
		for act_i,act in enumerate(act_bp_dict.keys()):
			start_times = []
			for bp in act_bp_dict[act]:
				start_times.append(act_bp_dict[act][bp][0,0])
			print(f"{subject_folder}-{act}: {start_times}")
			latest_start = max(start_times)
			# remove initial rows if can get closer to the latest start time
			for bp in act_bp_dict[act]:
				times = act_bp_dict[act][bp][:,0]
				closest_idx = np.argmin(abs(times - latest_start))
				act_bp_dict[act][bp] = act_bp_dict[act][bp][closest_idx:,1:]
			# get min length so duration is the same
			lengths = []
			for bp in act_bp_dict[act]:
				lengths.append(act_bp_dict[act][bp].shape[0])
			print(f"{subject_folder}-{act}: {lengths}")
			shortest = min(lengths)
			for bp in act_bp_dict[act]:
				act_bp_dict[act][bp] = act_bp_dict[act][bp][:shortest,:]

			# now merge body parts into one array
			trunc_len = shortest
			data_array = np.zeros((trunc_len,3*len(body_parts)))
			label_array = np.zeros(trunc_len)

			for bp_i,bp in enumerate(act_bp_dict[act]):
				data_array[:,bp_i*3:(bp_i+1)*3] = act_bp_dict[act][bp][:trunc_len,:]
			label_array[:] = act_i
			print(data_array.shape)

			# resample data
			resampling_factor = new_sampling_rate / og_sampling_rate
			old_length = len(data_array[:,0])
			new_length = int(old_length * resampling_factor)
			data_array = resample(data_array, new_length,axis=0)

			# resample labels
			t_e = old_length/og_sampling_rate
			t_old = np.linspace(0,t_e,old_length)
			t_e = new_length/new_sampling_rate
			t_new = np.linspace(0,t_e,new_length)
			closest_idxs = find_closest_index(t_old,t_new)
			label_array = label_array[closest_idxs]

			# put into list
			training_data[subject_i].append(data_array)
			training_labels[subject_i].append(label_array)

	output_folder = os.path.join(dataset_dir,"preprocessed_data")
	os.makedirs(output_folder,exist_ok=True)

	# now concatenate and save data
	for subject_i in range(NUM_SUBJECTS):
		if subject_i not in active_subjects:
			print(f"Skipping {subject_folder}")
			continue
		print(len(training_data[subject_i]))
		training_data[subject_i] = np.concatenate(training_data[subject_i])
		training_labels[subject_i] = np.concatenate(training_labels[subject_i])
	  
		print(f"==== {subject_i} ====")
		print(training_data[subject_i].shape)
		print(training_labels[subject_i].shape)

		np.save(os.path.join(output_folder,f"data_{subject_i+1}"),training_data[subject_i])
		np.save(os.path.join(output_folder,f"labels_{subject_i+1}"),training_labels[subject_i])
	
	# ------------- dataset metadata -------------
	sensors = ['acc']
	sensor_dims = 3 # XYZ
	channels_per_sensor = len(sensors)*sensor_dims

	# dict to get index of sensor channel by bp and sensor
	sensor_channel_map = {
		bp: 
		{
			sensor: np.arange(bp_i*channels_per_sensor+sensor_i*sensor_dims,
							bp_i*channels_per_sensor+sensor_i*sensor_dims+sensor_dims)
					for sensor_i,sensor in enumerate(sensors)
		} for bp_i,bp in enumerate(body_parts)
	}
	
	dataset_info = {
		'sensor_channel_map': sensor_channel_map,
		'list_of_subjects': np.array(active_subjects)+1,
		'label_map': label_map
	}
	
	with open(os.path.join(output_folder,"metadata.pickle"), 'wb') as file:
		pickle.dump(dataset_info, file)

def preprocess_PAMAP2(dataset_dir: str) -> dict:
	""" Loads the PAMAP raw data and saves it in a standard format.

	https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring

	Each subject's data and labels will be saved as data_[subject].npy and labels_[subject].npy
	in dataset_dir/preprocessed/. The data will have shape (N x C) where N is the number
	of raw samples per subject and C is the number of sensor channels.

	Parameters
	----------

	dataset_dir: str
		global path of where the dataset has been installed.


	Returns
	-------

	dataset_info: dict
		metadata about the raw data, specifically the 
		sensor channel map, list of subjects, and label map
	"""

	dataset_dir = os.path.expanduser(dataset_dir)

	label_map = {
			1:'lying',
			2:'sitting',
			3:'standing',
			4:'walking',
			5:'running',
			6:'cycling',
			7:'nordic walking',
			12:'ascending stairs',
			13:'descending stairs',
			16:'vacuuming',
			17:'ironing',
			24:'rope jumping'
			}

	body_parts = ['hand',
			  'chest',
			  'ankle']
	
	active_columns = np.array([1, # label
							   4,5,6, # hand acc
							   21,22,23, # chest acc
							   38,39,40]) # ankle acc

	og_sampling_rate = 100
	new_sampling_rate = 25

	activities = label_map.values()

	# PAMAP2 directory structure is Protocol/subject101.dat
	subject_files = os.listdir(dataset_dir)

	# Filter folder names that match the structure 'proband' followed by exactly two digits
	subject_files = [file for file in subject_files if 'subject' in file]
	subject_files.sort(key=lambda f: int(re.sub('\D', '', f)))
	NUM_SUBJECTS = len(subject_files)

	# we separate the data by subject
	training_data = {subject: [] for subject in range(NUM_SUBJECTS)} # raw data
	training_labels = {subject: [] for subject in range(NUM_SUBJECTS)} # raw labels


	# then load all the data
	for subject_i,subject_file in enumerate(subject_files):

		data = pd.read_table(os.path.join(dataset_dir,subject_file), header=None, sep='\s+')
		data = data.interpolate(method='linear', limit_direction='both')
		data_array = data.values[:,active_columns[1:]]
		label_array = data.values[:,active_columns[0]]

		# make labels contiguous
		activities = label_map.keys()
		activity_label_map = { 
			class_idx : label_map[activity_idx] for class_idx, activity_idx in enumerate(activities)
		}
		# print(f"Label Map: {activity_label_map}")

		label_swap = {activity_idx : class_idx for class_idx, activity_idx in enumerate(activities)}
		
		# realign labels to class idxs
		for activity_idx in activities:
			idxs_to_swap = (label_array == activity_idx).nonzero()[0]
			label_array[idxs_to_swap] = label_swap[activity_idx]
  

		# resample data
		resampling_factor = new_sampling_rate / og_sampling_rate
		old_length = len(data_array[:,0])
		new_length = int(old_length * resampling_factor)
		data_array = resample(data_array, new_length,axis=0)

		# resample labels
		t_e = old_length/og_sampling_rate
		t_old = np.linspace(0,t_e,old_length)
		t_e = new_length/new_sampling_rate
		t_new = np.linspace(0,t_e,new_length)
		closest_idxs = find_closest_index(t_old,t_new)
		label_array = label_array[closest_idxs]

		# put into list
		training_data[subject_i] = data_array
		training_labels[subject_i] = label_array

	output_folder = os.path.join(dataset_dir,"preprocessed_data")
	os.makedirs(output_folder,exist_ok=True)

	# now save data
	for subject_i in range(NUM_SUBJECTS):
	  
		print(f"==== {subject_i} ====")
		print(training_data[subject_i].shape)
		print(training_labels[subject_i].shape)

		np.save(os.path.join(output_folder,f"data_{subject_i+1}"),training_data[subject_i])
		np.save(os.path.join(output_folder,f"labels_{subject_i+1}"),training_labels[subject_i])
	
	# ------------- dataset metadata -------------
	sensors = ['acc']
	sensor_dims = 3 # XYZ
	channels_per_sensor = len(sensors)*sensor_dims

	# dict to get index of sensor channel by bp and sensor
	sensor_channel_map = {
		bp: 
		{
			sensor: np.arange(bp_i*channels_per_sensor+sensor_i*sensor_dims,
							bp_i*channels_per_sensor+sensor_i*sensor_dims+sensor_dims)
					for sensor_i,sensor in enumerate(sensors)
		} for bp_i,bp in enumerate(body_parts)
	}
	
	dataset_info = {
		'sensor_channel_map': sensor_channel_map,
		'list_of_subjects': np.arange(NUM_SUBJECTS)+1,
		'label_map': activity_label_map
	}
	
	with open(os.path.join(output_folder,"metadata.pickle"), 'wb') as file:
		pickle.dump(dataset_info, file)
	

def preprocess_Opportunity(dataset_dir: str) -> dict:
	""" Loads the Opportunity raw data and saves it in a standard format.

	https://archive.ics.uci.edu/dataset/226/opportunity+activity+recognition

	Each subject's data and labels will be saved as data_[subject].npy and labels_[subject].npy
	in dataset_dir/preprocessed/. The data will have shape (N x C) where N is the number
	of raw samples per subject and C is the number of sensor channels.

	Parameters
	----------

	dataset_dir: str
		global path of where the dataset has been installed.


	Returns
	-------

	dataset_info: dict
		metadata about the raw data, specifically the 
		sensor channel map, list of subjects, and label map
	"""

	dataset_dir = os.path.expanduser(dataset_dir)

	body_parts = ["BACK",
			  "RUA",
			  "RLA",
			  "LUA",
			  "LLA",
			  "L-SHOE",
			  "R-SHOE"]

	label_map = {0:"Null", 
				1:"Stand", 
				2:"Walk", 
				3:"Sit", 
				4:"Lie"
				}

	label_col = "Locomotion"
	sensor = 'InertialMeasurementUnit'

	upper_channel_list = ["accX",
						"accY",
						"accZ"]

	foot_channel_list = ["Body_Ax",
						"Body_Ay",
						"Body_Az"]
	
	og_sampling_rate = 30
	new_sampling_rate = 25

	# there are 4 users
	NUM_SUBJECTS = 4
	subjects = [1,2,3,4]
	# there are 5 normal runs and one drill run
	NUM_RUNS = 6

	# each user has 6 runs
	runs = ["ADL1",
			"ADL2",
			"ADL3",
			"ADL4",
			"ADL5",
			"Drill"]

	def get_column_mapping():
		'''
		This function returns a nested dictionary that returns all the columns
		and their corresponding sub items, e.g. InertialMeasurementUnit --> L-SHOE --> Body_Ax
		'''
		with open(os.path.join(dataset_dir,'column_names.txt'), 'r') as file:
			text = file.read()

		# Extract the strings between "Column:" and newline character or semicolon
		pattern = r'Column:\s+(\S.*?)(?=\n|;|$)'
		columns = re.findall(pattern, text)

		# Split the extracted strings into lists of individual words
		columns_list = [column.split() for column in columns]

		ms_idx = 0
		channel_idx_start = 1
		channel_idx_end = 243
		label_idx_start = 243

		# The "MILLISEC" column
		col_mapping_dict = {columns_list[ms_idx][1] : 0}

		# The sensor channel columns
		for col in columns_list[channel_idx_start:channel_idx_end]:
			col_idx = int(col[0]) - 1 # e.g. 0
			sensor = col[1] # e.g. "Accelerometer"
			position = col[2] # e.g. "RKN^"
			channel_type = col[3] # e.g. "accX" --> REED has an additional subchannel but we don't use REED so ignore

			# check if created sensor sub_dict
			if sensor not in col_mapping_dict.keys():
				col_mapping_dict[sensor] = {position : {channel_type : col_idx} }
			# check if created position sub_dict
			if position not in col_mapping_dict[sensor].keys():
				col_mapping_dict[sensor][position] = {channel_type : col_idx}
			else:
				col_mapping_dict[sensor][position][channel_type] = col_idx

		# The label columns
		for col in columns_list[label_idx_start:]:
			col_idx = int(col[0]) - 1 # e.g. 0
			label_level = col[1] # e.g. "Locomotion"
			col_mapping_dict[label_level] = col_idx

		return col_mapping_dict

	# get the col idxs of desired channels
	col_map_dict = get_column_mapping()
	active_col_idxs = []
	for bp in body_parts:
		if bp == 'L-SHOE' or bp == 'R-SHOE':
			channel_list = foot_channel_list
		else:
			channel_list = upper_channel_list
		for ch in channel_list:
			active_col_idxs.append(col_map_dict[sensor][bp][ch])
	active_col_idxs.append(col_map_dict[label_col])
	active_col_idxs = np.array(active_col_idxs)

	# we separate the data by subject
	training_data = {subject: [] for subject in range(NUM_SUBJECTS)} # raw data
	training_labels = {subject: [] for subject in range(NUM_SUBJECTS)} # raw labels


	# then load all the data
	for subject_i,subject in enumerate(subjects):
		for file_name in (runs):
			subject_file = f"S{subject}-{file_name}.dat"
			print(subject_file)
			# load the data
			data_file_path = os.path.join(dataset_dir,subject_file)
			data_array = pd.read_csv(data_file_path, sep=' ', header=None).values

			# only keep the accelerometer columns for the desired body parts
			data_array = data_array[:,active_col_idxs]
			
			# remove rows with Nans
			non_nan_rows = (np.isnan(data_array).sum(axis=1) == 0).nonzero()[0]
			data_array = data_array[non_nan_rows]

			# split data and labels
			label_array = data_array[:,-1] # last columns
			data_array = data_array[:,:-1] # drop last column

			# map labels to be contiguous
			if label_col != "Locomotion": 
				label_array[label_array == 406516] = 1
				label_array[label_array == 406517] = 2
				label_array[label_array == 404516] = 3
				label_array[label_array == 404517] = 4
				label_array[label_array == 406520] = 5
				label_array[label_array == 404520] = 6
				label_array[label_array == 406505] = 7
				label_array[label_array == 404505] = 8
				label_array[label_array == 406519] = 9
				label_array[label_array == 404519] = 10
				label_array[label_array == 406511] = 11
				label_array[label_array == 404511] = 12
				label_array[label_array == 406508] = 13
				label_array[label_array == 404508] = 14
				label_array[label_array == 408512] = 15
				label_array[label_array == 407521] = 16
				label_array[label_array == 405506] = 17
			else:
				label_array[label_array == 4] = 3
				label_array[label_array == 5] = 4


			# convert data to m/s^2 from milli-g
			data_array = data_array/1000*9.8

			# resample data
			resampling_factor = new_sampling_rate / og_sampling_rate
			old_length = len(data_array[:,0])
			new_length = int(old_length * resampling_factor)
			data_array = resample(data_array, new_length,axis=0)

			# resample labels
			t_e = old_length/og_sampling_rate
			t_old = np.linspace(0,t_e,old_length)
			t_e = new_length/new_sampling_rate
			t_new = np.linspace(0,t_e,new_length)
			closest_idxs = find_closest_index(t_old,t_new)
			label_array = label_array[closest_idxs]

			# put into list
			training_data[subject_i].append(data_array)
			training_labels[subject_i].append(label_array)

		training_data[subject_i] = np.concatenate(training_data[subject_i])
		training_labels[subject_i] = np.concatenate(training_labels[subject_i])

	output_folder = os.path.join(dataset_dir,"preprocessed_data")
	os.makedirs(output_folder,exist_ok=True)

	# now save data
	for subject_i in range(NUM_SUBJECTS):
	  
		print(f"==== {subject_i} ====")
		print(training_data[subject_i].shape)
		print(training_labels[subject_i].shape)

		np.save(os.path.join(output_folder,f"data_{subject_i+1}"),training_data[subject_i])
		np.save(os.path.join(output_folder,f"labels_{subject_i+1}"),training_labels[subject_i])
	
	# ------------- dataset metadata -------------
	sensors = ['acc']
	sensor_dims = 3 # XYZ
	channels_per_sensor = len(sensors)*sensor_dims

	# dict to get index of sensor channel by bp and sensor
	sensor_channel_map = {
		bp: 
		{
			sensor: np.arange(bp_i*channels_per_sensor+sensor_i*sensor_dims,
							bp_i*channels_per_sensor+sensor_i*sensor_dims+sensor_dims)
					for sensor_i,sensor in enumerate(sensors)
		} for bp_i,bp in enumerate(body_parts)
	}
	
	dataset_info = {
		'sensor_channel_map': sensor_channel_map,
		'list_of_subjects': np.arange(NUM_SUBJECTS)+1,
		'label_map': label_map
	}
	
	with open(os.path.join(output_folder,"metadata.pickle"), 'wb') as file:
		pickle.dump(dataset_info, file)

def preprocess_gesture(dataset_dir: str) -> dict:
	""" Loads the gesture raw data and saves it in a standard format.

	Each subject's data and labels will be saved as data_[subject].npy and labels_[subject].npy
	in dataset_dir/preprocessed/. The data will have shape (N x C) where N is the number
	of raw samples per subject and C is the number of sensor channels.

	Parameters
	----------

	dataset_dir: str
		global path of where the dataset has been installed.


	Returns
	-------

	dataset_info: dict
		metadata about the raw data, specifically the 
		sensor channel map, list of subjects, and label map
	"""

	dataset_dir = os.path.expanduser(dataset_dir)

	label_map = {
			0: '1-right',
			1: '2-left',
			2: '3-up',
			3: '4-down',
			4: '5-circle-clockwise',
			5: '6-circle-counter-clockwise',
			6: '7-square-clockwise',
			7: '8-square-counter-clockwise',
			8: '9-up-right',
			9: '10-up-left',
			10: '11-right-down',
			11: '12-left-down',
			12: '13-v-top-left',
			13: '14-v-top-right',
			14: '15-^-bottom-left',
			15: '16-^-bottom-right',
			16: '17-s-top-right',
			17: '18-s-top-left',
			18: '19-s-bottom-left',
			19: '20-s-bottom-right',
			}

	body_parts = ['right_wrist']
	

	og_sampling_rate = 9
	new_sampling_rate = 9

	activities = label_map.values()

	# gesture directory structure is subject/activity/data.txt
	subject_folders = os.listdir(dataset_dir)
	subject_folders = [sf for sf in subject_folders if len(sf) == 3]
	subject_folders.sort(key=lambda f: int(re.sub('\D', '', f)))
	NUM_SUBJECTS = len(subject_folders)

	activity_folders = os.listdir(os.path.join(dataset_dir,subject_folders[0]))
	activity_folders.sort(key=lambda f: int(re.sub('\D', '', f)))

	# we separate the data by subject
	training_data = {subject: [] for subject in range(NUM_SUBJECTS)} # raw data
	training_labels = {subject: [] for subject in range(NUM_SUBJECTS)} # raw labels
	training_windows = {subject: [] for subject in range(NUM_SUBJECTS)} # raw window idxs


	# merge data for each subject into a numpy array
	for subject_i, subject_folder in enumerate(subject_folders):
		window_idx = 0
		for activity_i, activity_folder in enumerate(activity_folders):
			data_files = os.listdir(os.path.join(dataset_dir,subject_folder,activity_folder))
			data_files.sort(key=lambda f: int(re.sub('\D', '', f)))
			data = []
			labels = []#np.zeros(len(data_files))
			window_labels = np.zeros((len(data_files),2)).astype(int)
			# we do not want to do normal windowing, our samples are already segmented
			# store the data in a single large numpy array as before but store the window indices in another array
			# then instead of calling create windows, we simply load these windows
			# we want to train the initial RNN, and then with random sampling
			for data_i,(data_file) in enumerate(data_files):
				df = pd.read_csv(os.path.join(dataset_dir,subject_folder,activity_folder,data_file),sep='\s+', header=None)
				# print(os.path.join(dataset_dir,subject_folder,activity_folder,data_file))
				data.append(df[[3,4,5]].values)
				if window_idx == 0:
					window_labels[data_i,:] = np.array([0,len(df)])
				else:
					window_labels[data_i,0] = window_idx
					window_labels[data_i,1] = window_idx + len(df) # en = st + len
				window_idx += len(df)
				labels.append(np.zeros(len(df))+int(activity_i))
			data = np.concatenate(data)
			labels = np.concatenate(labels)

			# put into list
			training_data[subject_i].append(data)
			training_labels[subject_i].append(labels)
			training_windows[subject_i].append(window_labels)

	output_folder = os.path.join(dataset_dir,"preprocessed_data")
	os.makedirs(output_folder,exist_ok=True)

	# now save data
	for subject_i in range(NUM_SUBJECTS):
		training_data[subject_i] = np.concatenate(training_data[subject_i])
		training_labels[subject_i] = np.concatenate(training_labels[subject_i])
		training_windows[subject_i] = np.concatenate(training_windows[subject_i])

		print(f"==== {subject_i} ====")
		print(training_data[subject_i].shape)
		print(training_labels[subject_i].shape)
		print(training_windows[subject_i].shape)

		np.save(os.path.join(output_folder,f"data_{subject_i+1}"),training_data[subject_i])
		np.save(os.path.join(output_folder,f"labels_{subject_i+1}"),training_labels[subject_i])
		np.save(os.path.join(output_folder,f"windows_{subject_i+1}"),training_windows[subject_i])
	
	# ------------- dataset metadata -------------
	sensors = ['acc']
	sensor_dims = 3 # XYZ
	channels_per_sensor = len(sensors)*sensor_dims

	# dict to get index of sensor channel by bp and sensor
	sensor_channel_map = {
		bp: 
		{
			sensor: np.arange(bp_i*channels_per_sensor+sensor_i*sensor_dims,
							bp_i*channels_per_sensor+sensor_i*sensor_dims+sensor_dims)
					for sensor_i,sensor in enumerate(sensors)
		} for bp_i,bp in enumerate(body_parts)
	}
	
	dataset_info = {
		'sensor_channel_map': sensor_channel_map,
		'list_of_subjects': np.arange(NUM_SUBJECTS)+1,
		'label_map': label_map
	}
	
	with open(os.path.join(output_folder,"metadata.pickle"), 'wb') as file:
		pickle.dump(dataset_info, file)

def read_xml(path):
	# Parse the XML file
	tree = ET.parse(path)  # Replace with your actual file path
	root = tree.getroot()

	# store all windows in a list
	windows = []

	# Extract data from XML
	for gesture in root.findall('Gesture'):
		x_vals, y_vals, z_vals, t_vals = [], [], [], []
		for stroke in gesture.findall('Stroke'):
			for point in stroke.findall('Point'):
				x_vals.append(float(point.get('X')))
				y_vals.append(float(point.get('Y')))
				z_vals.append(float(point.get('Z')))
				t_vals.append(float(point.get('T')))
		windows.append(np.column_stack([t_vals,x_vals,y_vals,z_vals]))

	return windows


def preprocess_gesture_impair(dataset_dir: str) -> dict:
	""" Loads the gesture raw data and saves it in a standard format.

	Each subject's data and labels will be saved as data_[subject].npy and labels_[subject].npy
	in dataset_dir/preprocessed/. The data will have shape (N x C) where N is the number
	of raw samples per subject and C is the number of sensor channels.

	Parameters
	----------

	dataset_dir: str
		global path of where the dataset has been installed.


	Returns
	-------

	dataset_info: dict
		metadata about the raw data, specifically the 
		sensor channel map, list of subjects, and label map
	"""

	dataset_dir = os.path.expanduser(dataset_dir)

	label_map = {
			0: 'circle',
			1: 'double',
			2: 'rotate_fast_slow',
			3: 'rotate_slow_fast',
			4: 'shake',
			5: 'tap'
			}

	body_parts = ['right_wrist']
	

	og_sampling_rate = 25
	new_sampling_rate = 12.5

	activities = label_map.values()

	# gesture directory structure is subject/activity/data.txt
	subject_folders = os.listdir(dataset_dir)
	subject_folders = [sf for sf in subject_folders if sf.isdigit()]
	subject_folders.sort(key=lambda f: int(re.sub('\D', '', f)))
	NUM_SUBJECTS = len(subject_folders)

	activity_files = os.listdir(os.path.join(dataset_dir,subject_folders[0]))

	# we separate the data by subject
	training_data = {subject: [] for subject in range(NUM_SUBJECTS)} # raw data
	training_labels = {subject: [] for subject in range(NUM_SUBJECTS)} # raw labels
	training_windows = {subject: [] for subject in range(NUM_SUBJECTS)} # raw window idxs


	# merge data for each subject into a numpy array
	for subject_i, subject_folder in enumerate(subject_folders):
		window_idx = 0
		for activity_i, activity_file in enumerate(activity_files):
			data = []
			labels = []#np.zeros(len(data_files))
			gesture_windows = read_xml(os.path.join(dataset_dir,subject_folder,activity_file))
			window_labels = np.zeros((len(gesture_windows),2)).astype(int)
			# we do not want to do normal windowing, our samples are already segmented
			# store the data in a single large numpy array as before but store the window indices in another array
			# then instead of calling create windows, we simply load these windows
			# we want to train the initial RNN, and then with random sampling
			for data_i,(gesture_window) in enumerate(gesture_windows):
				# print(os.path.join(dataset_dir,subject_folder,activity_folder,data_file))
				# resample data
				resampling_factor = new_sampling_rate / og_sampling_rate
				old_length = len(gesture_window[:,0])
				new_length = int(old_length * resampling_factor)
				gesture_window = resample(gesture_window, new_length,axis=0)


				data.append(gesture_window[:,np.array([1,2,3])])
				if window_idx == 0:
					window_labels[data_i,:] = np.array([0,len(gesture_window)])
				else:
					window_labels[data_i,0] = window_idx
					window_labels[data_i,1] = window_idx + len(gesture_window) # en = st + len
				window_idx += len(gesture_window)
				labels.append(np.zeros(len(gesture_window))+int(activity_i))
			data = np.concatenate(data)
			labels = np.concatenate(labels)

			# put into list
			training_data[subject_i].append(data)
			training_labels[subject_i].append(labels)
			training_windows[subject_i].append(window_labels)

	output_folder = os.path.join(dataset_dir,"preprocessed_data")
	os.makedirs(output_folder,exist_ok=True)

	# now save data
	for subject_i in range(NUM_SUBJECTS):
		training_data[subject_i] = np.concatenate(training_data[subject_i])
		training_labels[subject_i] = np.concatenate(training_labels[subject_i])
		training_windows[subject_i] = np.concatenate(training_windows[subject_i])

		print(f"==== {subject_i} ====")
		print(training_data[subject_i].shape)
		print(training_labels[subject_i].shape)
		print(training_windows[subject_i].shape)

		np.save(os.path.join(output_folder,f"data_{subject_i+1}"),training_data[subject_i])
		np.save(os.path.join(output_folder,f"labels_{subject_i+1}"),training_labels[subject_i])
		np.save(os.path.join(output_folder,f"windows_{subject_i+1}"),training_windows[subject_i])
	
	# ------------- dataset metadata -------------
	sensors = ['acc']
	sensor_dims = 3 # XYZ
	channels_per_sensor = len(sensors)*sensor_dims

	# dict to get index of sensor channel by bp and sensor
	sensor_channel_map = {
		bp: 
		{
			sensor: np.arange(bp_i*channels_per_sensor+sensor_i*sensor_dims,
							bp_i*channels_per_sensor+sensor_i*sensor_dims+sensor_dims)
					for sensor_i,sensor in enumerate(sensors)
		} for bp_i,bp in enumerate(body_parts)
	}
	
	dataset_info = {
		'sensor_channel_map': sensor_channel_map,
		'list_of_subjects': np.arange(NUM_SUBJECTS)+1,
		'label_map': label_map
	}
	
	with open(os.path.join(output_folder,"metadata.pickle"), 'wb') as file:
		pickle.dump(dataset_info, file)


if __name__ == '__main__':
	# preprocess_DSADS(os.path.expanduser("~/Projects/data/dsads"))
	# preprocess_RWHAR(os.path.expanduser("~/Projects/data/rwhar"))
	# preprocess_PAMAP2(os.path.expanduser("~/Projects/data/pamap2/"))
	# preprocess_Opportunity(os.path.expanduser("~/Projects/data/opportunity"))
	# preprocess_gesture(os.path.expanduser("~/Projects/data/gestures-dataset/gestures-dataset"))
	preprocess_gesture_impair(os.path.expanduser("~/Projects/data/gesture_impair"))