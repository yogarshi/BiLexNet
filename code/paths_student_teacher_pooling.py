__author__ = 'yogarshi'
import sys, os
import argparse
import math
import torch
import torch.nn.functional as F
import torch.optim as O
from torch import nn
import time
import numpy as np


from evaluation_common import *

# Path LSTM parameters -- TODO : Maybe move some/all of these in the model constructor?
NUM_LAYERS = 2
LSTM_HIDDEN_DIM = 60
PROJECTION_DIM = 50
LEMMA_DIM = 50
POS_DIM = 4
DEP_DIM = 5
DIR_DIM = 1
MINIBATCH_SIZE = 4
LOSS_EPSILON = 0.0 # 0.01
MAX_PATH_BATCH = 500 # Reduce this number if you get an out of memory CUDA errora


class PathLSTMClassifier(nn.Module):

	def __init__(self, model_type, num_lemmas_en, num_lemmas_hi, num_pos, num_dep, num_directions=5, n_epochs=10, num_relations=2,
                 alpha=0.01, lemma_embeddings_en=None, lemma_embeddings_hi=None, en_vocab = None, hi_vocab= None, label_dict = None, relations = None,
				 dropout=0.0, use_xy_embeddings=False, num_hidden_layers=0, use_gpu=False,  temperature=20, kd_alpha=0.5, project_embeds=False):
		super(PathLSTMClassifier, self).__init__()

		self.n_epochs = n_epochs
		self.model_type = model_type
		self.num_lemmas = num_lemmas_en
		self.num_pos = num_pos
		self.num_dep = num_dep
		self.num_directions = num_directions
		self.num_relations = num_relations
		self.alpha = alpha
		self.dropout = dropout
		self.use_xy_embeddings = use_xy_embeddings
		self.num_hidden_layers = num_hidden_layers
		self.update = True
		self.relations = relations
		self.use_gpu = use_gpu
		self.project_embeds = project_embeds
		self.temperature = temperature
		self.kd_alpha = kd_alpha
		self.en_vocab = en_vocab
		self.hi_vocab = hi_vocab
		self.label_dict = label_dict
		self.ff_dropout = dropout

		print ("Using the fixed attention model")

		# self.pad_idx = lemma_embeddings_en.shape[0]


		if lemma_embeddings_en is not None:
			print ("Initializing English embeddings with pre-trained embeddings")
			weights = torch.from_numpy(lemma_embeddings_en).float()
			#print weights.type()
			self.lemma_embeddings_en = nn.Embedding(weights.shape[0], weights.shape[1], padding_idx=0)
			# self.lemma_embeddings = nn.Embedding.from_pretrained(weights)
			self.lemma_embeddings_en.weight = nn.Parameter(weights)
			self.lemma_embeddings_en_dim = weights.shape[1]
		else:
			self.lemma_embeddings_en = nn.Embedding(num_lemmas_en + 1, LEMMA_DIM, padding_idx=num_lemmas_en)
			self.lemma_embeddings_en_dim = LEMMA_DIM

		if lemma_embeddings_hi is not None:
			print ("Initializing Hindi embeddings with pre-trained embeddings")
			weights = torch.from_numpy(lemma_embeddings_hi).float()
			# print weights.type()
			self.lemma_embeddings_hi = nn.Embedding(weights.shape[0], weights.shape[1], padding_idx=0)
			self.lemma_embeddings_hi.weight = nn.Parameter(weights)
			self.lemma_embeddings_hi_dim = weights.shape[1]
		else:
			self.lemma_embeddings_hi = nn.Embedding(num_lemmas_hi + 1,
													LEMMA_DIM,
													padding_idx=num_lemmas_hi)
			self.lemma_embeddings_hi_dim = LEMMA_DIM

		self.lemma_embeddings_en.weight.requires_grad = False
		self.lemma_embeddings_hi.weight.requires_grad = False


		print ('Creating the network')

		# Word/lemma embedding

		self.pos_embeddings = nn.Embedding(num_pos,POS_DIM,padding_idx=0)
		self.dep_embeddings = nn.Embedding(num_dep,DEP_DIM,padding_idx=0)
		self.dir_embeddings = nn.Embedding(num_directions,DIR_DIM,padding_idx=0)



		# if self.model_type == 'embed_only':
		# 	self.fc1 = nn.Linear(2*self.lemma_embeddings_en_dim, 50)
		# else:
		# 	self.fc1 = nn.Linear(LSTM_HIDDEN_DIM + 2 * self.lemma_embeddings_en_dim, 50)
		# self.fc2 = nn.Linear(50,self.num_relations)

		self.fc_mono = self.fc_cross = nn.Sequential(
			nn.Linear(LSTM_HIDDEN_DIM + 2*self.lemma_embeddings_en_dim, 50),
			#nn.Linear(LSTM_HIDDEN_DIM + 2*PROJECTION_DIM, 50),
			nn.Dropout(p=self.ff_dropout),
			nn.ReLU(),
			# nn.Linear(50,50),
			# nn.Dropout(p=self.dropout),
			# nn.ReLU(),
			nn.Linear(50, num_relations)
		)

		# self.fc_mono = self.fc_cross = nn.Sequential(
		# 	nn.Linear(2 * self.lemma_embeddings_en_dim, 50),
		# 	# nn.Linear(LSTM_HIDDEN_DIM + 2*PROJECTION_DIM, 50),
		# 	nn.Dropout(p=self.ff_dropout),
		# 	nn.ReLU(),
		# 	# nn.Linear(50,50),
		# 	# nn.Dropout(p=self.dropout),
		# 	# nn.ReLU(),
		# 	nn.Linear(50, num_relations)
		# )

		# self.fc_mono = self.fc_cross = nn.Sequential(
		# 	nn.Linear(LSTM_HIDDEN_DIM, 50),
		# 	# nn.Linear(LSTM_HIDDEN_DIM + 2*PROJECTION_DIM, 50),
		# 	nn.Dropout(p=self.ff_dropout),
		# 	nn.ReLU(),
		# 	# nn.Linear(50,50),
		# 	# nn.Dropout(p=self.dropout),
		# 	# nn.ReLU(),
		# 	nn.Linear(50, num_relations)
		# )

		if self.project_embeds:
			pass
			# This keeps the embeddings fixed and instead uses a non-linear transformation that is tune
			# self.en_projection = nn.Sequential(
			# 	nn.Linear(self.lemma_embeddings_en_dim, PROJECTION_DIM, )
			# )
			#
			# self.hi_projection = nn.Sequential(
			# 	nn.Linear(self.lemma_embeddings_hi_dim, PROJECTION_DIM, )
			# )
			#
			# self.lemma_embeddings_en.weight.requires_grad = False
			# self.lemma_embeddings_hi.weight.requires_grad = False
			#
			# self.fc1 = nn.Linear(LSTM_HIDDEN_DIM + 2 * PROJECTION_DIM, 50)
			# self.path_lstm = nn.LSTM(
			# 	PROJECTION_DIM + POS_DIM + DEP_DIM + DIR_DIM,
			# 	LSTM_HIDDEN_DIM, NUM_LAYERS, batch_first=True, dropout=self.dropout)
			#
			# self.path_lstm_st = nn.LSTM(
			# 	PROJECTION_DIM + POS_DIM + DEP_DIM + DIR_DIM,
			# 	LSTM_HIDDEN_DIM, NUM_LAYERS, batch_first=True, dropout=self.dropout)
		else:
			self.path_lstm = nn.LSTM(
				self.lemma_embeddings_en_dim + POS_DIM + DEP_DIM + DIR_DIM,
				LSTM_HIDDEN_DIM, NUM_LAYERS, batch_first=True, dropout=self.dropout)
			self.path_lstm_st = nn.LSTM(
				self.lemma_embeddings_en_dim + POS_DIM + DEP_DIM + DIR_DIM,
				LSTM_HIDDEN_DIM, NUM_LAYERS, batch_first=True, dropout=self.dropout)

		# A feature transformation for the cross-lingual paths
		self.fc_cross = nn.Sequential(
			nn.Linear(LSTM_HIDDEN_DIM + 2*self.lemma_embeddings_en_dim, 50),
			#nn.Linear(LSTM_HIDDEN_DIM + 2*PROJECTION_DIM, 50),
			nn.Dropout(p=self.ff_dropout),
			nn.ReLU(),
			# nn.Linear(50,50),
			# nn.Dropout(p=self.ff_dropout),
			# nn.ReLU(),
			nn.Linear(50, num_relations)
		)

		# self.fc_cross = nn.Sequential(
		# 	nn.Linear(2 * self.lemma_embeddings_en_dim, 50),
		# 	# nn.Linear(LSTM_HIDDEN_DIM + 2*PROJECTION_DIM, 50),
		# 	nn.Dropout(p=self.ff_dropout),
		# 	nn.ReLU(),
		# 	# nn.Linear(50,50),
		# 	# nn.Dropout(p=self.ff_dropout),
		# 	# nn.ReLU(),
		# 	nn.Linear(50, num_relations)
		# )

		# self.fc_cross = nn.Sequential(
		# 	nn.Linear(LSTM_HIDDEN_DIM, 50),
		# 	# nn.Linear(LSTM_HIDDEN_DIM + 2*PROJECTION_DIM, 50),
		# 	nn.Dropout(p=self.ff_dropout),
		# 	nn.ReLU(),
		# 	# nn.Linear(50,50),
		# 	# nn.Dropout(p=self.ff_dropout),
		# 	# nn.ReLU(),
		# 	nn.Linear(50, num_relations)
		# )

		self.en_word_att = nn.Linear(self.lemma_embeddings_en_dim,25)
		self.hi_word_att = nn.Linear(self.lemma_embeddings_hi_dim, 25)
		self.label_att = nn.Embedding(self.num_relations,25)

		self.att_scores = nn.Sequential(nn.Linear(2*self.lemma_embeddings_en_dim + 25,50),
										nn.ReLU(),
										nn.Dropout(p=self.ff_dropout),
										nn.Linear(50,1))


		total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
		print ("Total parameters = {0}".format(total_params))

		print (self.pos_embeddings.weight)

	def distillation(self, y, teacher_scores, labels):
		p = F.log_softmax(y / self.temperature, )
		q = F.softmax(teacher_scores / self.temperature)
		l_kl = F.kl_div(p, q, size_average=False) * (self.temperature ** 2) / \
			   y.shape[0]
		l_ce = F.cross_entropy(y, labels)

		return l_kl * self.kd_alpha + l_ce * (1. - self.kd_alpha)

	def train_batch(self,criterion,minibatch_size,batch_indices,x_y_vectors,
		X_train_xlingual_en,X_train_xlingual_hi,X_train_mono_en, trans_lemmas_vectors, y_train):
		main_loss = None

		batch_preds = []
		batch_golds = []
		for i in range(minibatch_size):
			idx = batch_indices[i]
			gold = y_train[idx,].view(1, )
			input = (x_y_vectors[idx], X_train_xlingual_en[idx], X_train_xlingual_hi[idx], X_train_mono_en[idx], trans_lemmas_vectors[idx], gold)
			preds, preds_cross, _ = self.forward(input)

			# sys.stderr.write("---DEBUG : PREDS = {0}, PREDS CROSS = {1}\n".format(' '.join(map(str, F.softmax(preds.reshape(self.num_relations)).tolist())) ,
			# 																					   ' '.join(map(str,F.softmax(preds_cross.reshape(self.num_relations)).tolist()))))

			max_arg = torch.argmax(preds).tolist()
			batch_preds.append(max_arg)
			batch_golds.append(gold.tolist()[0])
			if not main_loss:
				main_loss = criterion(preds, gold) 
				aux_loss = self.distillation(preds_cross, preds, gold)
				aux_loss_val = aux_loss.item()
				main_loss += aux_loss
			else:
				main_loss += criterion(preds, gold)
				aux_loss = self.distillation(preds_cross, preds, gold)
				aux_loss_val += aux_loss.item()
				main_loss += aux_loss

		return main_loss, aux_loss_val, batch_golds, batch_preds

		
		



	def fit(self, X_train_xlingual_en, X_train_xlingual_hi, X_train_mono_en, X_train_mono_hi, y_train, x_y_vectors,
			 trans_lemmas_vectors, weight, X_val_xlingual_en, X_val_xlingual_hi, X_val_mono_en, X_val_mono_hi, x_y_vectors_val, trans_lemmas_vectors_val, y_val):

		weight = torch.FloatTensor(weight)
		if self.use_gpu:
			weight = weight.cuda()

		# criterion = nn.CrossEntropyLoss()
		criterion = nn.CrossEntropyLoss(weight=weight)
		# st_criterion = nn.Cros
		# opt = O.SGD(self.parameters(),lr=self.alpha)
		opt = O.Adam(self.parameters(),lr=self.alpha)
		# for name, param in self.named_parameters():
		# 	if param.requires_grad:
		# 		print name, param.data
		num_examples = len(y_train)
		if self.use_gpu:
			y_train = torch.cuda.LongTensor(y_train)
		else:
			y_train = torch.LongTensor(y_train)

		minibatch_size = min(MINIBATCH_SIZE, num_examples)
		nminibatches = int(math.ceil(num_examples / minibatch_size))
		previous_loss = 100000

		for epoch in range(self.n_epochs):

			# Freeze certain parameters after a few epochs
			# if epoch == 5:
			# 	sys.stdout.write("Freezing params\n")
			# 	for param in self.att_scores.parameters():
			# 		param.requires_grad = False
			# 	self.label_att.weight.requires_grad = False
			#
			# 	for param in self.path_lstm.parameters():
			# 		param.requires_grad = False
			# 	for param in self.fc_mono.parameters():
			# 		param.requires_grad = False


			print ('Starting epoch {0}'.format(epoch + 1))
			print ('Total mini-batches {0}'.format(nminibatches))
			epoch_loss = 0.0
			epoch_main_loss = 0.0
			epoch_aux_loss = 0.0
			epoch_indices = torch.LongTensor(np.random.permutation(num_examples))
			start_time = time.time()
			all_preds = []
			all_golds = []

			# with torch.autograd.set_detect_anomaly(True):
			for minibatch in range(nminibatches):

				loss = None
				batch_indices = torch.LongTensor(epoch_indices[minibatch * minibatch_size:(minibatch + 1) * minibatch_size])
				opt.zero_grad()
				self.train()

				main_loss, aux_loss_val, gold_batch, preds_batch = self.train_batch(criterion, minibatch_size,batch_indices,
					x_y_vectors,X_train_xlingual_en,X_train_xlingual_hi,X_train_mono_en, trans_lemmas_vectors, y_train)
				all_preds += preds_batch
				all_golds += gold_batch
				main_loss /= minibatch_size
				main_loss.backward()
				opt.step()
				epoch_main_loss += main_loss.item()
				epoch_aux_loss += aux_loss_val/minibatch_size

			end_time = time.time()
			# epoch_loss /= num_examples
			epoch_main_loss /= nminibatches
			epoch_aux_loss /= nminibatches
			epoch_loss /= nminibatches
			# print ('Epoch {0}, main loss = {1}, Time taken = {2}'.format(epoch + 1, epoch_loss,end_time - start_time))
			print ('Epoch {0}, main loss = {1}, aux loss = {2}, Time taken = {3}'.format(epoch + 1,epoch_main_loss, epoch_aux_loss, end_time - start_time))
			# evaluate(all_golds,all_preds,self.relations, do_full_reoprt=True)


			# On the validation set
			# pred, vecs, probs = self.predict(X_val_xlingual_en, X_val_xlingual_hi, X_val_mono_en,X_val_mono_hi,x_y_vectors_val,trans_lemmas_vectors_val)
			# evaluate(y_val,pred, self.relations, do_full_reoprt=True)

			if math.fabs(previous_loss - epoch_loss) < LOSS_EPSILON:
				break
			previous_loss = epoch_loss


	def predict(self, X_test_xlingual_en, X_test_xlingual_hi, X_test_mono_en, X_test_mono_hi, x_y_vectors, trans_lemmas_vectors=None, test_time=False):

		self.eval()
		num_examples = len(X_test_xlingual_en)
		all_preds = []
		all_probs = []
		all_vecs = []
		for i in range(num_examples):

			if test_time:
				input = (x_y_vectors[i], X_test_xlingual_en[i], X_test_xlingual_hi[i], X_test_mono_en[i], trans_lemmas_vectors[i])
				preds , vecs = self.forward(input, test_time)
			else:
				input = (x_y_vectors[i], X_test_xlingual_en[i], X_test_xlingual_hi[i], X_test_mono_en[i], trans_lemmas_vectors[i])
				preds, vecs = self.forward(input, test_time, val_time=True)
			final_preds = F.softmax(preds)[0]
			# all_vecs.append(vecs)
			#print (preds)
			max, max_arg = torch.max(final_preds,0)
			all_preds.append(max_arg[0].item())
			#all_probs.append(float(max))
			all_probs.append(preds.tolist()[0])
		#print (all_preds)
		return all_preds, all_vecs, all_probs

	def generate_path_vec(self, p, lang, st=False):
		paths_batch, lengths, counts = p
		num_paths = paths_batch.shape[0]
		# print ("Num paths = {0}".format(num_paths))
		sys.stdout.flush()

		# Each of these should be shaped (num_paths x longest path x embeddings)
		if lang == "en":
			lemma_embeds = self.lemma_embeddings_en(paths_batch[:, :, 0])
			if self.project_embeds:
				lemma_embeds = self.en_projection(lemma_embeds)
		else:
			lemma_embeds = self.lemma_embeddings_hi(paths_batch[:, :, 0])
			if self.project_embeds:
				lemma_embeds = self.hi_projection(lemma_embeds)
		dep_embeds = self.dep_embeddings(paths_batch[:, :, 1])
		pos_embeds = self.pos_embeddings(paths_batch[:, :, 2])
		dir_embeds = self.dir_embeddings(paths_batch[:, :, 3])

		path_embeds = torch.cat([lemma_embeds, dep_embeds, pos_embeds, dir_embeds], dim=2)

		# Pack -> run through lstm -> unpack
		# We do this in batches of MAX_PATH_BATCH to avoid out-of-memory errors
		iters = int(math.ceil(num_paths * 1.0 / MAX_PATH_BATCH))

		# print (iters)
		all_path_vecs = []
		for i in range(iters):
			packed_paths = nn.utils.rnn.pack_padded_sequence(path_embeds[i * MAX_PATH_BATCH:min((i + 1) * MAX_PATH_BATCH, num_paths)],
															 lengths[i * MAX_PATH_BATCH:min((i + 1) * MAX_PATH_BATCH, num_paths)],
															 batch_first=True)
			# if st:
			# 	lstm_out, (ht, ct) = self.path_lstm_st(packed_paths)
			# else:
			lstm_out, (ht, ct) = self.path_lstm(packed_paths)
			# Grab the last time step on the last layer
			path_vecs = ht[-1]
			all_path_vecs.append(path_vecs)

		path_vecs = torch.cat(all_path_vecs)

		# Mutiply by counts
		path_vecs = path_vecs * counts.unsqueeze(1)
		# Sum all paths
		path_vec = path_vecs.sum(dim=0)
		# path_vec /= counts.sum()

		# if self.use_gpu:
		# 	path_vec = path_vec.cuda()

		return path_vec


	def forward(self, input, test_time=False,val_time=False):
		"""
		:param input: (w1,w2,xlingual en paths, xlingual hi paths)
		:return:
		"""

		vec_list = []

		if not test_time and not val_time:
			x_y_batch, p_xling_en_all, p_xling_hi_all, p_mono_en_all, trans_lemmas_vectors, gold_label = input
		else:
			x_y_batch, p_xling_en_all, p_xling_hi_all, p_mono_en_all, trans_lemmas_vectors = input
		x_embed = self.lemma_embeddings_en(x_y_batch[0])
		if self.project_embeds:
			x_embed = self.en_projection(x_embed)
		if not test_time and not val_time:
			y_embed = self.lemma_embeddings_en(x_y_batch[1])
			if self.project_embeds:
				y_embed = self.en_projection(y_embed)
		else:
			y_embed = self.lemma_embeddings_hi(x_y_batch[1])
			if self.project_embeds:
				y_embed = self.hi_projection(y_embed)


		x_y_embed = torch.cat((x_embed, y_embed))

		sys.stdout.flush()
		if self.model_type == 'xling_only':

			if not test_time and not val_time:

				# Get features from monolingual data



				# if self.use_gpu:
				# 	path_vec_mono_en = path_vec_mono_en.cuda()

				## PATH BASED PART STARTS HERE
				p_mono_en = p_mono_en_all[0]
				# for p_mono_en in p_mono_en_all:
				if p_mono_en is not None:
					path_vec_mono_en = self.generate_path_vec(p_mono_en,"en") / p_mono_en[2].sum()
				else:
					path_vec_mono_en = torch.zeros((LSTM_HIDDEN_DIM))

				if self.use_gpu:
					path_vec_mono_en = path_vec_mono_en.cuda()
				# PATH BASED PART ENDS HERE

				path_vec_mono_en /= len(p_mono_en_all)
				# path_vec = path_vec_mono_en
				fc_input = torch.cat((x_y_embed, path_vec_mono_en))
				#fc_input = x_y_embed
				# fc_input = path_vec_mono_en
				# fc_output = self.fc2(F.relu(self.fc1(torch.cat((x_y_embed, path_vec_mono_en))))).view(1,-1)
				fc_output = self.fc_mono(fc_input).view(1,-1)


				# Get features from cross-lingual data
				num_trans = trans_lemmas_vectors.shape[0]
				## PATH BASED PART STARTS HERE
				null_vector = torch.zeros((LSTM_HIDDEN_DIM))
				if self.use_gpu:
					null_vector = null_vector.cuda()
				# print p_xling_en_all, p_xling_hi_all
				# print trans_lemmas_vectors
				assert len(p_xling_en_all) == len(p_xling_hi_all)

				path_vecs_xling_all_l = []
				count = 0
				for i in range(num_trans):


					p_xling_en = p_xling_en_all[i]
					if p_xling_en is not None:
						en_vec = self.generate_path_vec(p_xling_en, "en",st=True)
						count += p_xling_en[2].sum().item()
					else:
						en_vec = null_vector

					p_xling_hi = p_xling_hi_all[i]
					if p_xling_hi is not None:
						hi_vec = self.generate_path_vec(p_xling_hi, "hi", st=True)
						count += p_xling_hi[2].sum().item()
					else:
						hi_vec = null_vector

					path_vec_xling_curr = en_vec + hi_vec


					if self.use_gpu:
						path_vec_xling_curr = path_vec_xling_curr.cuda()
					if count > 0:
						path_vec_xling_final = path_vec_xling_curr/ count
					else:
						path_vec_xling_final = path_vec_xling_curr
					path_vecs_xling_all_l.append(path_vec_xling_final)
				path_vecs_xling_all = torch.stack(path_vecs_xling_all_l)
				## PATH BASED PART ENDS HERE



				# path_vecs_xling_all = torch.cat(path_vecs_xling_all_l, dim=0)
				# path_vecs_xling_all = torch.cat(())


				# path_vecs_xling_all = torch.randn((num_trans,LSTM_HIDDEN_DIM))

				if self.use_gpu:
					path_vecs_xling_all = path_vecs_xling_all.cuda()

				trans_lemmas_embeds = self.lemma_embeddings_hi(trans_lemmas_vectors)

				# print path_vecs_xling_all.shape, trans_lemmas_embeds.shape
				# print x_embed.unsqueeze(0).expand(path_vecs_xling_all.shape[0],x_embed.shape[0]).shape

				# avg_weights =  F.softmax(torch.bmm(x_embed.unsqueeze(0).expand(num_trans,x_embed.shape[0]).view(num_trans, 1, x_embed.shape[0]), trans_lemmas_embeds.view(num_trans, trans_lemmas_embeds.shape[1], 1)).squeeze(2), dim=0)

				prepool_fc_input = torch.cat((x_embed.unsqueeze(0).expand(num_trans, x_embed.shape[0]), trans_lemmas_embeds,path_vecs_xling_all,), dim=1)
				#  prepool_fc_input = torch.cat((x_embed.unsqueeze(0).expand(num_trans, x_embed.shape[0]), trans_lemmas_embeds), dim=1)
				# prepool_fc_input = path_vecs_xling_all
				#prepool_fc_input = torch.cat((path_vecs_xling_all,), dim=1)
				# fc_input = avg_weights.expand(num_trans, prepool_fc_input.shape[1]) * prepool_fc_input
				# fc_input = fc_input.sum(dim=0)


				# Embed the label
				# label_embed = self.label_att(gold_label)
				# en_word_transform = self.en_word_att(x_embed).expand(num_trans, 25) # Shape =  x25
				# hi_words_transform = self.hi_word_att(trans_lemmas_embeds) # Shape = num_trans x 25
				#
				# att_input1 = label_embed.expand(num_trans,label_embed.shape[1]).view(num_trans, 1, label_embed.shape[1])
				# att_input2 = F.tanh(en_word_transform + hi_words_transform).view(num_trans,25,1)
				# att_vec= F.softmax(torch.bmm(att_input1, att_input2).squeeze(2), dim=0)
				# # att_vec = F.softmax(F.tanh(en_word_transform + hi_words_transform)*label_embed)
				# fc_input = (att_vec.expand(num_trans,prepool_fc_input.shape[1]) * prepool_fc_input).sum(dim=0)
				# # fc_input = fc_input
				# # self.en_word_att() + self.hi
				#

				label_embed = self.label_att(gold_label)
				# label_embed = torch.zeros((self.num_relations))
				# if self.use_gpu:
				# 	label_embed = label_embed.cuda()
				# label_embed[gold_label] = 1
				en_word_transform = x_embed.expand(num_trans,self.lemma_embeddings_en_dim)
				# label_transform = label_embed.expand(num_trans,self.num_relations)
				label_transform = label_embed.expand(num_trans,25)
				input = torch.cat((trans_lemmas_embeds,en_word_transform,label_transform), dim=1)
				output = F.softmax(self.att_scores(input)/0.1, dim=0)
				fc_input = (output.expand(num_trans, prepool_fc_input.shape[1]) * prepool_fc_input).sum(dim=0)

				# sys.stderr.write("---DEBUG : SOFTMAX SCORES = {0}, count = {1}\n".format(' '.join(map(str,output.reshape(num_trans).tolist())),count))
				# sys.stderr.write("---DEBUG : EN WORD = {0}, EN WORD 2 = {1}. HI WORDS = {2}, LABEL = {3}\n".format(self.en_vocab[x_y_batch[0].tolist()],
				# 																					  self.en_vocab[x_y_batch[1].tolist()],
				# 																					  ' '.join([self.hi_vocab[x] for x in trans_lemmas_vectors.tolist()]).encode('utf-8'),
				# 																								   self.label_dict[gold_label.tolist()[0]]))



				# print prepool_fc_input.shape
				# prepool_fc_input = prepool_fc_input.transpose(0,1).view(1,-1,num_trans)
				# # print prepool_fc_input.shape
				# fc_input = F.avg_pool1d(prepool_fc_input,kernel_size=num_trans).squeeze(-1)
				# print fc_input.shape

				# y_embed_trans = self.lemma_embeddings_hi(trans_lemmas_vectors[0])
				# if self.project_embeds:
				# 	y_embed_trans = self.hi_projection(y_embed_trans)
				# x_y_embed_trans = torch.cat((x_embed,y_embed_trans))
				#
				# fc_input = torch.cat((x_y_embed_trans, path_vec_xling))
				# fc_input = torch.cat((x_y_embed, path_vec_xling))
				fc_output_cross = self.fc_cross(fc_input).view(1,-1)
				# fc_output_cross = self.fc2(F.relu(self.fc1(fc_input))).view(-1,1)

				# assert fc_output.shape == fc_output_cross.shape

				return fc_output, fc_output_cross, vec_list



			else:


				# If there is a cross-lingual path then get predictions from there
				# if True:
				if True :

					## PATH BASED PART STARTS HERE
					null_vector = torch.zeros((LSTM_HIDDEN_DIM))
					if self.use_gpu:
						null_vector = null_vector.cuda()
					# print p_xling_en_all, p_xling_hi_all
					# print trans_lemmas_vectors
					count = 0

					p_xling_en = p_xling_en_all[0]
					if p_xling_en is not None:
						en_vec = self.generate_path_vec(p_xling_en, "en",st=True)
						count += p_xling_en[2].sum().item()
					else:
						en_vec = null_vector

					p_xling_hi = p_xling_hi_all[0]
					if p_xling_hi is not None:
						hi_vec = self.generate_path_vec(p_xling_hi, "hi",st=True)
						count += p_xling_hi[2].sum().item()
					else:
						hi_vec = null_vector

					path_vec_xling = en_vec + hi_vec
					if count > 0:
						path_vec_xling /= count

					if self.use_gpu:
						path_vec_xling = path_vec_xling.cuda()
					## PATH BASED PART ENDS HERE
					fc_input = torch.cat((x_y_embed, path_vec_xling))
					# fc_input = x_y_embed
					# fc_input = path_vec_xling

					# fc_output = self.fc2(F.relu(self.fc1(fc_input))).view(1, -1)
					# return fc_output, vec_list

					fc_output_cross = self.fc_cross(fc_input).view(1, -1)
					return fc_output_cross, vec_list

				else:
					path_vec_mono_en = torch.zeros((LSTM_HIDDEN_DIM))
					if self.use_gpu:
						path_vec_mono_en = path_vec_mono_en.cuda()
					for p_mono_en in p_mono_en_all:
						if p_mono_en is not None:
							path_vec_mono_en += self.generate_path_vec(p_mono_en, "en", st=True)
					path_vec_mono_en /= len(p_mono_en_all)
					path_vec = path_vec_mono_en

					y_embed_trans = self.lemma_embeddings_en(trans_lemmas_vectors[0])

					x_y_embed_trans = torch.cat((x_embed, y_embed_trans))
					fc_input = torch.cat((x_y_embed_trans, path_vec))
					fc_output = self.fc2(F.relu(self.fc1(fc_input))).view(1, -1)

					return fc_output, vec_list
