import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pickle
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import f1_score
import multiprocessing
import json
import torch.nn.functional as F

from models.model_builder import model_builder
from datasets.dataset import HARClassifierDataset, val_collate_fn, rnn_collate_fn
from experiments.train import train, validate
from utils.setup_funcs import PROJECT_ROOT, MODEL_ROOT, DATA_ROOT, init_logger, init_seeds
from datasets.preprocess_raw_data import preprocess_DSADS, preprocess_RWHAR, preprocess_PAMAP2, preprocess_gesture
from utils.parse_results import get_results


def get_args():
	parser = argparse.ArgumentParser(
			description="Dataset and model arguments for training HAR policies",
			formatter_class=argparse.ArgumentDefaultsHelpFormatter
		)
	
	parser.add_argument(
			"--architecture",
			default="attend",
			type=str,
			choices=["attend", "tinyhar", "convlstm","vrnn"],
			help="HAR architecture",
		)
	parser.add_argument(
			"--dataset",
			default="dsads",
			type=str,
			choices=["dsads", "rwhar", "pamap2", "opportunity","geffen","gesture","gesture_impair"],
			help="HAR dataset",
		)

	parser.add_argument("--eval", action="store_true", help="Get results of pretrained policies (optional)")
	parser.add_argument("--single_sensor_checkpoint_prefix", default="single_sensor_logfile", type=str, help="name for single sensor classifier training session (optioanl)")
	parser.add_argument("--logging_prefix", default="logfile", type=str, help="name for training session (optional)")
	parser.add_argument("--seed", default=0, type=int, help="seed for experiment, this must match the seeds used when training the classifier", required=True)
	parser.add_argument('--subjects', default=[1,2,3,4,5,6,7], nargs='+', type=int, help='List of subjects', required = True)
	parser.add_argument('--sensors', default=["acc"], nargs='+', type=str, help='List of sensors', required=True)
	parser.add_argument('--body_parts',default=["torso","right_arm","left_arm","right_leg","left_leg"], nargs='+', type=str, help='List of body parts', required=True)
	parser.add_argument('--activities', default=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18], nargs='+', type=int, help='List of activities',required=True)
	parser.add_argument("--val_frac", default=0.1, type=float, help="fraction of training data for validation",required=True)
	parser.add_argument("--window_size", default=8, type=int, help="sliding window size for pretrained model",required=True)
	parser.add_argument("--budget", default=7, type=int, help="number of observations [2,7]",required=True)
	parser.add_argument("--input_dim", default=3, type=int, help="RNN Input Dimension")
	parser.add_argument("--sampling_policy", default="learned", type=str, help="Sampling Policy Type")

	parser.add_argument("--policy_batch_size", default=128, type=int, help="training batch size")
	parser.add_argument("--policy_lr", default=1e-4, type=float, help="learning rate")
	parser.add_argument("--policy_epochs", default=50, type=int, help="training epochs")
	parser.add_argument("--ese", default=10, type=int, help="early stopping epochs")
	parser.add_argument("--log_freq", default=200, type=int, help="after how many batches to log")

	# finetuning arguments
	parser.add_argument("--classifier_batch_size", default=32, type=int, help="finetuning batch size")
	parser.add_argument("--classifier_lr", default=1e-4, type=float, help="learning rate for finetuning model")
	parser.add_argument("--classifier_epochs", default=5, type=int, help="finetuning epochs")
	args = parser.parse_args()

	return args


def train_LOOCV(**kwargs):
	""" Trains N policies for Leave One Subject Out Cross Validation

		Parameters
		----------
		
		**kwargs:
			parameters used for the dataset (e.g. batch_size, body parts, subjects, etc.)

		Returns
		-------
	"""

	subjects = kwargs['subjects']
	logging_prefix = kwargs['logging_prefix']
	seed = kwargs['seed']

	# setup the session
	logger = init_logger(f"{logging_prefix}/train_log_seed{seed}")
	init_seeds(seed)

	logger.info(json.dumps(kwargs,indent=4))

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	kwargs['device'] = device

	results_table = {subject: None for subject in subjects}

	for subject_i, subject in enumerate(subjects):
		train_subjects = subjects[:subject_i] + subjects[subject_i+1:]
		test_subjects = [subjects[subject_i]]

		logger.info(f"Train Group: {train_subjects} --> Test Group: {test_subjects}")

		# generic name of this log
		kwargs['train_logname'] = f"{logging_prefix}/{test_subjects}_seed{seed}"

		# create the dataset
		preprocessed_path = os.path.join(DATA_ROOT[kwargs['dataset']], "preprocessed_data")
		if not os.path.isdir(preprocessed_path):
			if kwargs['dataset'] == 'dsads':
				preprocess_DSADS(DATA_ROOT[kwargs['dataset']])
			elif kwargs['dataset'] == 'rwhar':
				preprocess_RWHAR(DATA_ROOT[kwargs['dataset']])
			elif kwargs['dataset'] == 'pamap2':
				preprocess_PAMAP2(DATA_ROOT[kwargs['dataset']])
			elif kwargs['dataset'] == 'gesture':
				preprocess_gesture(DATA_ROOT[kwargs['dataset']])
			elif kwargs['dataset'] == 'gesture_impair':
				preprocess_gesture(DATA_ROOT[kwargs['dataset']])
		kwargs['dataset_dir'] = preprocessed_path

		kwargs['subjects'] = train_subjects
		kwargs['normalize'] = True
		train_ds = HARClassifierDataset(**kwargs,train=True,val=False)
		val_ds = HARClassifierDataset(**kwargs,train=False,val=True)
		kwargs['subjects'] = test_subjects
		test_ds = HARClassifierDataset(**kwargs,train=False,val=False)
		
		# ====================== Prepare for Policy Training ======================

		# load the pretrained model
		model_ckpt_path = os.path.join(MODEL_ROOT,"saved_data/checkpoints",f"{kwargs['single_sensor_checkpoint_prefix']}/{test_subjects}_seed{seed}.pth")
		model = model_builder(**kwargs)
		model.load_state_dict(torch.load(model_ckpt_path)['model_state_dict'],strict=False)
		model = model.to('cpu')
		kwargs['best_f1'] = 0

		
		# training loop for the policy
		# we train at each skip iteration to learn to map h_i to some skip
		for skip_i in range(kwargs['budget']):

			ckpt_path = os.path.join(MODEL_ROOT,f"saved_data/checkpoints/",f"{kwargs['train_logname']}.pth")
			if os.path.exists(ckpt_path):
				logger.info(f"Loading Pretrained Model: {ckpt_path}")
				model.load_state_dict(torch.load(ckpt_path)['model_state_dict'])
				best_f1 = torch.load(ckpt_path)['val_f1']
				kwargs['best_f1'] = best_f1

			# we need this step because the __getitem__ function relies on
			# the model and the skip_i to create policy samples
			train_ds.model = model
			train_ds.skip_i = skip_i

			val_ds.model = model
			val_ds.skip_i = skip_i

			test_ds.model = model
			test_ds.skip_i = skip_i

			kwargs['classifier_training'] = False
			kwargs['policy_logging'] = True
			train_ds.classifier_training = False
			val_ds.classifier_training = False
			test_ds.classifier_training = False

			train_loader = torch.utils.data.DataLoader(train_ds, batch_size=kwargs['policy_batch_size'], shuffle=True, pin_memory=False,drop_last=True,num_workers=1)
			val_loader = torch.utils.data.DataLoader(val_ds, batch_size=kwargs['policy_batch_size'], shuffle=False, pin_memory=False,drop_last=True,num_workers=1,collate_fn=val_collate_fn)
			test_loader = torch.utils.data.DataLoader(test_ds, batch_size=kwargs['policy_batch_size'], shuffle=False, pin_memory=False,drop_last=True,num_workers=1,collate_fn=val_collate_fn)

			# try toload the best validation set from last skip_i
			ckpt_path = os.path.join(MODEL_ROOT,f"saved_data/checkpoints/",f"{kwargs['train_logname']}.pth")
			if os.path.exists(ckpt_path):
				logger.info(f"Loading Pretrained Model: {ckpt_path}")
				model.load_state_dict(torch.load(ckpt_path)['model_state_dict'])
				best_f1 = torch.load(ckpt_path)['val_f1']
				kwargs['best_f1'] = best_f1

			kwargs['lr'] = kwargs['policy_lr'] #/ (2*skip_i+1)
			kwargs['epochs'] = kwargs['policy_epochs']
			
			kwargs['model'] = model
			kwargs['loss_fn'] = nn.CrossEntropyLoss()
			kwargs['optimizer'] = torch.optim.Adam(model.parameters(),lr=kwargs['lr'])
			kwargs['train_logname'] = f"{logging_prefix}/{test_subjects}_seed{seed}"
			kwargs['device'] = 'cpu'
			kwargs['train_loader'] = train_loader
			kwargs['val_loader'] = val_loader
			kwargs['logger'] = logger
			kwargs['lr_scheduler'] = torch.optim.lr_scheduler.CosineAnnealingLR(kwargs['optimizer'],kwargs['epochs'])
			 

			# train(**kwargs)
			

			# try toload the best validation set from last skip_i
			ckpt_path = os.path.join(MODEL_ROOT,f"saved_data/checkpoints/",f"{kwargs['train_logname']}.pth")
			if os.path.exists(ckpt_path):
				logger.info(f"Loading Pretrained Model: {ckpt_path}")
				model.load_state_dict(torch.load(ckpt_path)['model_state_dict'])
				best_f1 = torch.load(ckpt_path)['val_f1']
				kwargs['best_f1'] = best_f1


			kwargs['lr'] = kwargs['classifier_lr'] #/ (2*skip_i+1)
			kwargs['epochs'] = kwargs['classifier_epochs']

			kwargs['model'] = model
			kwargs['loss_fn'] = nn.CrossEntropyLoss()
			kwargs['optimizer'] = torch.optim.Adam(model.parameters(),lr=kwargs['lr'])
			kwargs['train_logname'] = f"{logging_prefix}/{test_subjects}_seed{seed}"
			kwargs['device'] = 'cpu'
			kwargs['logger'] = logger
			kwargs['lr_scheduler'] = torch.optim.lr_scheduler.CosineAnnealingLR(kwargs['optimizer'],kwargs['epochs'])


			# then here we enable classifier training and disable policy logging
			kwargs['classifier_training'] = True
			kwargs['policy_logging'] = False
			train_ds.classifier_training = True
			val_ds.classifier_training = True
			test_ds.classifier_training = True
			kwargs['train_loader'] = torch.utils.data.DataLoader(train_ds, batch_size=kwargs['classifier_batch_size'], shuffle=True, pin_memory=False,drop_last=True,num_workers=1,collate_fn=rnn_collate_fn)
			kwargs['val_loader'] = torch.utils.data.DataLoader(val_ds, batch_size=kwargs['classifier_batch_size'], shuffle=False, pin_memory=False,drop_last=True,num_workers=1,collate_fn=rnn_collate_fn)
			test_loader = torch.utils.data.DataLoader(test_ds, batch_size=kwargs['classifier_batch_size'], shuffle=False, pin_memory=False,drop_last=True,num_workers=1,collate_fn=rnn_collate_fn) 

			# train(**kwargs)


		# then here we enable classifier training and disable policy logging
		kwargs['classifier_training'] = True
		kwargs['policy_logging'] = False
		train_ds.classifier_training = True
		val_ds.classifier_training = True
		test_ds.classifier_training = True
		kwargs['train_loader'] = torch.utils.data.DataLoader(train_ds, batch_size=kwargs['classifier_batch_size'], shuffle=True, pin_memory=False,drop_last=True,num_workers=1,collate_fn=rnn_collate_fn)
		kwargs['val_loader'] = torch.utils.data.DataLoader(val_ds, batch_size=kwargs['classifier_batch_size'], shuffle=False, pin_memory=False,drop_last=True,num_workers=1,collate_fn=rnn_collate_fn)
		test_loader = torch.utils.data.DataLoader(test_ds, batch_size=kwargs['classifier_batch_size'], shuffle=False, pin_memory=False,drop_last=True,num_workers=1,collate_fn=rnn_collate_fn) 

		logger.info("============== Testing ==============")
		
		
		# load the one with the best validation accuracy and evaluate on test set
		ckpt_path = os.path.join(MODEL_ROOT,f"saved_data/checkpoints/",f"{kwargs['train_logname']}.pth")
		logger.info(f"{ckpt_path}")
		model.load_state_dict(torch.load(ckpt_path)['model_state_dict'])

		kwargs['model'] = model
		kwargs['val_loader'] = test_loader
		test_acc,test_f1,test_loss = validate(**kwargs)
		logger.info(f"FINAL Test F1: {test_f1}, Test Acc: {test_acc}")
		logger.info("==========================================\n\n")

		results_table[subject] = (test_f1, test_acc)


	logger.info(f"Results: {results_table}")

	# create parent directories if needed
	kwargs['train_logname'] = os.path.join(logging_prefix,f"results_seed{seed}")
	path_items = kwargs['train_logname'].split("/")
	if  len(path_items) > 1:
		Path(os.path.join(MODEL_ROOT,"saved_data/results",*path_items[:-1])).mkdir(parents=True, exist_ok=True)

	with open(os.path.join(MODEL_ROOT,"saved_data/results",kwargs['train_logname']+".pickle"), 'wb') as file:
		pickle.dump(results_table, file)
		


if __name__ == '__main__':

	args = get_args()
	eval_only = args.eval
	args = vars(args)

	# organize logs by dataset and architecture
	if 'single_sensor_checkpoint_prefix' in args.keys():
		args['single_sensor_checkpoint_prefix'] = os.path.join(args['dataset'],args['architecture'],args['single_sensor_checkpoint_prefix'])

	args['logging_prefix'] = os.path.join(args['dataset'],args['architecture'],args['logging_prefix'])

	if eval_only:
		base_path = os.path.join(MODEL_ROOT,"saved_data/results",args['logging_prefix'])
		result_logs = [os.path.join(base_path, filename) for filename in os.listdir(base_path)]
		mean, std, results_table = get_results(result_logs)
		print(f"Mean: {round(mean*100,3)}, std: {round(std*100,3)}")
	else:
		train_LOOCV(**args)
	
	