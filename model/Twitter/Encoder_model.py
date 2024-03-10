import argparse
import time
import torch
from tqdm import tqdm
import copy as cp
import torch.nn.functional as F

from torch.utils.data import random_split
from torch_geometric.data import DataLoader, DataListLoader
from torch_geometric.nn import DataParallel
from torch.nn import Linear, ReLU, Tanh, LeakyReLU
from torch_geometric.nn import global_mean_pool, GATConv, SAGEConv, GCNConv, \
	global_max_pool, global_add_pool, GINConv, TopKPooling, global_sort_pool, GlobalAttention
from torch_geometric.utils import to_undirected, add_self_loops
import GCL.losses as L
import GCL.augmentors as A
from GCL.models import DualBranchContrast
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pack_sequence, pad_packed_sequence
from utils.data_loader import *
from utils.eval_helper import *
from seq_augs import Seq_Augmentor
from graph_augs import Graph_Augmentor
from DisenGCN import DisenGCN, Cor_loss
from layers import TransformerEncoder, DisenTransformerEncoder
import numpy as np

class G_Encoder(torch.nn.Module):
	def __init__(self, args):
		super(G_Encoder, self).__init__()
		self.num_features = dataset.num_features
		self.num_classes = args.num_classes
		self.nhid = args.nhid
		self.model = args.Gmodel
		self.K = args.K
		self.routit = args.routit
		self.cor_weight = args.cor_weight
		self.tau = args.tau

		if self.model == 'gcn':
			self.conv1 = GCNConv(self.num_features, self.nhid)
			self.conv2 = GCNConv(self.num_features, self.nhid)
		elif self.model == 'sage':
			self.conv1 = SAGEConv(self.num_features, self.nhid)
			self.conv2 = SAGEConv(self.num_features, self.nhid)
		elif self.model == 'gat':
			self.conv1 = GATConv(self.num_features, self.nhid)
			self.conv2 = GATConv(self.num_features, self.nhid)
		elif self.model == 'gin':
			nn = torch.nn.Sequential(Linear(self.num_features, self.nhid), ReLU(), Linear(self.nhid, self.nhid))
			self.conv1 = GINConv(nn)
			self.conv2 = GINConv(nn)
		elif self.model == 'disen':
			self.conv1 = DisenGCN(self.num_features, self.nhid, 1, self.K, self.routit, self.cor_weight, self.tau)
			self.conv2 = DisenGCN(self.num_features, self.nhid, 1, self.K, self.routit, self.cor_weight, self.tau)
		self.lin1 = torch.nn.Linear(self.nhid * 2, self.nhid)
		self.readout = args.readout
		self.bn0 = torch.nn.BatchNorm1d(self.num_features)
		self.gate_nn = torch.nn.Sequential(Linear(self.nhid, 1), LeakyReLU())
		self.nn = torch.nn.Sequential(Linear(self.nhid, self.nhid), LeakyReLU(), Linear(self.nhid, self.nhid))
		self.AttPool = GlobalAttention(gate_nn=self.gate_nn, nn=self.nn)

	def forward(self, x, edge_index, batch, edge_weight):
		x = self.bn0(x)
		if self.model == 'disen':
			x, k_x = self.conv1(x, edge_index)
			x = F.relu(x)
		else:
			x = F.relu(self.conv1(x, edge_index))
			k_x = x

		if self.readout == 'mean':
			g = global_mean_pool(x, batch)
			k_g = global_mean_pool(k_x, batch)
		elif self.readout == 'max':
			g = global_max_pool(x, batch)
			k_g = global_max_pool(k_x, batch)
		elif self.readout == 'sum':
			g = global_add_pool(x, batch)
			k_g = global_add_pool(k_x, batch)
		elif self.readout == 'sort':
			g = global_sort_pool(x, batch, k=50)
			k_g = global_sort_pool(k_x, batch)
		elif self.readout == 'att':
			g = self.AttPool(x, batch)
			k_g = self.AttPool(k_x, batch)
		return g, k_g

class Seq_Encoder(torch.nn.Module):
	def __init__(self, args):
		super(Seq_Encoder, self).__init__()
		self.num_features = dataset.num_features
		self.num_classes = args.num_classes
		self.nhid = args.nhid
		self.k = args.K1
		self.lin1 = torch.nn.Linear(self.nhid, self.nhid)
		self.bn0 = torch.nn.BatchNorm1d(self.num_features)
		self.bn1 = torch.nn.BatchNorm1d(self.nhid)
		self.Smodel = args.Smodel
		self.max_len = args.max_len
		self.position_embedding = torch.nn.Embedding(self.max_len, self.nhid)
		self.LayerNorm = torch.nn.LayerNorm(self.nhid)
		self.dropout = torch.nn.Dropout(0.0)
		if self.Smodel == 'LSTM':
			self.seq_encoder = torch.nn.LSTM(input_size=self.num_features, hidden_size=self.nhid,
								  bias=True, batch_first=False, dropout=0.0, bidirectional=False)
		elif self.Smodel == 'GRU':
			self.seq_encoder = torch.nn.GRU(input_size=self.num_features, hidden_size=self.nhid, num_layers=1,
									bias=True, batch_first=False, dropout=0.0, bidirectional=False)
		elif self.Smodel == 'Transformer':
			self.seq_encoder = TransformerEncoder(n_layers=2, hidden_size=self.nhid, inner_size=self.nhid, hidden_dropout_prob=0.5,  attn_dropout_prob=0.5)
		elif self.Smodel == 'DisenTransformer':
			self.seq_encoder = DisenTransformerEncoder(n_layers=2, hidden_size=self.nhid, k_interests=self.k,
													   inner_size=self.nhid, hidden_dropout_prob=0.5, n_heads=2,
													   attn_dropout_prob=0.5)
		self.lin2 = torch.nn.Linear(self.num_features, self.nhid)
		self.AdaptiveMaxPool1d = torch.nn.AdaptiveMaxPool1d(1)

	def RNN_Max_pooling(self, x):
		x = x.unsqueeze(0).permute(0, 2, 1)
		x = self.AdaptiveMaxPool1d(x).view(x.shape[1])
		return x
	def Transformer_Max_pooling(self, x):
		x = x.permute(0, 2, 1)
		x = self.AdaptiveMaxPool1d(x).squeeze(2)
		return x

	def get_attention_mask(self, item_seq):
		"""Generate left-to-right uni-directional attention mask for multi-head attention."""
		attention_mask = (item_seq > 0).long()
		extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
		# mask for left-to-right unidirectional
		max_len = attention_mask.size(-1)
		attn_shape = (1, max_len, max_len)
		subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
		subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
		subsequent_mask = subsequent_mask.long().to(item_seq.device)

		extended_attention_mask = extended_attention_mask * subsequent_mask
		extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
		extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
		return extended_attention_mask

	def gather_indexes(self, output, gather_index):
		"""Gathers the vectors at the specific positions over a minibatch"""
		gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
		output_tensor = output.gather(dim=1, index=gather_index)
		return output_tensor.squeeze(1)

	def forward(self, x, edge_index, batch, data, Aug):
		x = self.bn0(x)
		''' sequential view '''

		seqs_index = []
		news_seqs = torch.zeros([data.num_graphs, self.max_len], dtype=int)
		for s_id in range(data.num_graphs):
			seq = (batch == s_id).nonzero().squeeze().tolist()
			if s_id == 0: seq = seq[1:]
			if Aug == None:
				pass
			else:
				seq = Aug(seq)
			seqs_index.append(seq)
			news_seqs[s_id][:len(seq)] = torch.LongTensor(seq)
		seq_len = [len(s) for s in seqs_index]
		# seq_len = torch.Tensor(
		seq_feature = [x[seqs_index[i]] for i in range(data.num_graphs)]
		if self.Smodel == 'GRU' or self.Smodel == 'LSTM':
			seq_feature = pad_sequence(seq_feature)
			seq_feature = pack_padded_sequence(seq_feature, seq_len, enforce_sorted=False)
			lstm, _ = self.seq_encoder(seq_feature)  # shape=(seq_length,batch_size,input_size)
			x, _ = pad_packed_sequence(lstm)
			x = x.permute(1, 0, 2)  # shape=(batch_size,seq_length,input_size)
			s = torch.stack([torch.mean(x[i][:seq_len[i]], dim=0) for i in range(data.num_graphs)], dim=0) #mean
			# s = torch.stack([self.RNN_Max_pooling(x[i][:seq_len[i]]) for i in range(data.num_graphs)], dim=0)  # max
			s = self.bn1(F.relu(self.lin1(s)))

		elif self.Smodel == 'Transformer':
			seq_len = torch.tensor(seq_len).cuda()
			position_ids = torch.arange(news_seqs.size(1), dtype=torch.long).cuda()
			position_ids = position_ids.unsqueeze(0).expand_as(news_seqs)
			position_embedding = self.position_embedding(position_ids)

			seq_feature = self.lin2(x[news_seqs])
			input_emb = seq_feature + position_embedding
			input_emb = self.LayerNorm(input_emb)
			input_emb = self.dropout(input_emb)

			extended_attention_mask = self.get_attention_mask(news_seqs).cuda()
			trm_output = self.seq_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
			output = trm_output[-1]
			s = torch.mean(output, dim=1)#self.gather_indexes(output, seq_len - 1)
			# s = self.Transformer_Max_pooling(output)
			s = self.bn1(F.relu(self.lin1(s)))
		elif self.Smodel == 'DisenTransformer':
			position_ids = torch.arange(news_seqs.size(1), dtype=torch.long).cuda()
			position_embedding = self.position_embedding(position_ids)

			seq_feature = self.lin2(x[news_seqs])
			input_emb = seq_feature
			input_emb = self.LayerNorm(input_emb)
			input_emb = self.dropout(input_emb)

			trm_output = self.seq_encoder(input_emb, position_embedding, output_all_encoded_layers=True)
			output = trm_output[-1]
			s = torch.mean(output, dim=1)  # self.gather_indexes(output, seq_len - 1)
			# s = self.Transformer_Max_pooling(output)
			s = self.bn1(F.relu(self.lin1(s)))
		return s

class G_Constrative(torch.nn.Module):
	def __init__(self, args, G_encoder, Seq_encoder):
		super(G_Constrative, self).__init__()
		self.G_encoder = G_encoder
		self.nhid = args.nhid
		self.K = args.K
		self.num_classes = args.num_classes
		self.Seq_encoder = Seq_encoder
		self.aug_ratio = args.aug_ratio
		self.Seq_augment1 = Seq_Augmentor(aug_ratio=self.aug_ratio)
		self.G_aug1 = Graph_Augmentor(aug_ratio=self.aug_ratio)
		# self.G_aug1 = A.Compose([A.NodeDropping(pn=self.aug_ratio)])# , A.EdgeAdding(pe=self.aug_ratio), A.EdgeRemoving(pe=self.aug_ratio)
		# aug2 = A.Compose([A.NodeDropping(pn=0.2)])
		self.pre_agg = args.pre_agg
		self.s_fc = torch.nn.Linear(self.nhid, self.num_classes)
		self.g_fc = torch.nn.Linear(self.nhid, self.num_classes)
		self.cat_fc = torch.nn.Linear(self.nhid * 2, self.num_classes)
		self.project = torch.nn.Sequential(Linear(self.nhid, self.nhid), LeakyReLU(), Linear(self.nhid, self.nhid))
		self.lin = torch.nn.Linear(self.nhid // self.K, self.nhid // self.K).cuda()

	def forward(self, data):
		x, edge_index, batch = data.x, data.edge_index, data.batch

		# Node_id add 1, x extend 1 dim
		x = torch.cat([torch.zeros([1, x.shape[1]]).cuda(), x], dim=0)
		add_edge = torch.ones_like(edge_index)
		edge_index = edge_index + add_edge
		batch = torch.cat([torch.LongTensor([0]).cuda(), batch])
		data.x, data.edge_index, data.batch = x, edge_index, batch

		# graph augment
		data1 = self.G_aug1(data)
		x1, edge_index1, batch1 = data1.x, data1.edge_index, data1.batch

		# graph encoder
		g1, k_g1 = self.G_encoder(x1, edge_index1, batch, None)
		g, k_g = self.G_encoder(data.x, data.edge_index, data.batch, None)

		# sequence encoder
		s = self.Seq_encoder(x, edge_index, batch, data, None)
		s1 = self.Seq_encoder(x, edge_index, batch, data, self.Seq_augment1)

		# 预测
		if self.pre_agg == 'sum':
			log = F.log_softmax(self.g_fc(g+s), dim=-1)
		elif self.pre_agg == 'cat':
			log = F.log_softmax(self.cat_fc(torch.cat([g, s], dim=1)), dim=-1)
		elif self.pre_agg == 'graph':
			log = F.log_softmax(self.g_fc(g), dim=-1)
		elif self.pre_agg == 'seq':
			log = F.log_softmax(self.s_fc(s), dim=-1)
		return log, g, g1, s, s1, 0
