import sys,os
sys.path.append(os.getcwd())
# from Process.process import *
import torch as th

# from torch_scatter import scatter_mean
import torch.nn.functional as F
# import numpy as np
# from tools.earlystopping2class import EarlyStopping
# from torch_geometric.data import DataLoader
# from tqdm import tqdm
# from Process.rand5fold import *
# from tools.evaluate import *
# from torch_geometric.nn import GCNConv
# from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
# import copy
import random


cuda_id = 1
# os.environ
device = th.device(f'cuda:{cuda_id}' if cuda_id >= 0 else 'cpu')
class SCL(th.nn.Module):
    def __init__(self, temperature=0.1):
        super(SCL, self).__init__()
        self.temperature = temperature

    def forward(self, inrep_1, inrep_2, label_1, label_2=None):
        inrep_1.to(device)
        inrep_2.to(device)
        bs_1 = int(inrep_1.shape[0])
        bs_2 = int(inrep_2.shape[0])

        if label_2 == None:
            normalize_inrep_1 = F.normalize(inrep_1, p=2, dim=1)
            normalize_inrep_2 = F.normalize(inrep_2, p=2, dim=1)
            cosine_similarity = th.matmul(normalize_inrep_1, normalize_inrep_2.t())  # bs_1, bs_2

            diag = th.diag(cosine_similarity)
            cos_diag = th.diag_embed(diag)  # bs,bs

            label = th.unsqueeze(label_1, -1)
            if label.shape[0] == 1:
                cos_loss = th.zeros(1)
            else:
                for i in range(label.shape[0] - 1):
                    if i == 0:
                        label_mat = th.cat((label, label), -1)
                    else:
                        label_mat = th.cat((label_mat, label), -1)  # bs, bs
                # print(label_mat.size())
                # print(label.size())
                # exit(0)

                mid_mat_ = (label_mat.eq(label_mat.t()))
                mid_mat = mid_mat_.float()

                cosine_similarity = (cosine_similarity - cos_diag) / self.temperature  # the diag is 0
                mid_diag = th.diag_embed(th.diag(mid_mat))
                mid_mat = mid_mat - mid_diag

                cosine_similarity = cosine_similarity.masked_fill_(mid_diag.byte(), -float('inf'))  # mask the diag

                cos_loss = th.log(
                    th.clamp(F.softmax(cosine_similarity, dim=1) + mid_diag, 1e-10, 1e10))  # the sum of each row is 1

                cos_loss = cos_loss * mid_mat

                cos_loss = th.sum(cos_loss, dim=1) / (th.sum(mid_mat, dim=1) + 1e-10)  # bs

        else:
            if bs_1 != bs_2:
                while bs_1 < bs_2:
                    inrep_2 = inrep_2[:bs_1]
                    label_2 = label_2[:bs_1]
                    break
                while bs_2 < bs_1:
                    inrep_2_ = inrep_2
                    ra = random.randint(0, int(inrep_2_.shape[0]) - 1)
                    pad = inrep_2_[ra].unsqueeze(0)
                    lbl_pad = label_2[ra].unsqueeze(0)
                    inrep_2 = th.cat((inrep_2, pad), 0)
                    label_2 = th.cat((label_2, lbl_pad), 0)
                    bs_2 = int(inrep_2.shape[0])

            normalize_inrep_1 = F.normalize(inrep_1, p=2, dim=1)
            normalize_inrep_2 = F.normalize(inrep_2, p=2, dim=1)
            cosine_similarity = th.matmul(normalize_inrep_1, normalize_inrep_2.t())  # bs_1, bs_2

            label_1 = th.unsqueeze(label_1, -1)
            label_1_mat = th.cat((label_1, label_1), -1)
            for i in range(label_1.shape[0] - 1):
                if i == 0:
                    label_1_mat = label_1_mat
                else:
                    label_1_mat = th.cat((label_1_mat, label_1), -1)  # bs, bs

            label_2 = th.unsqueeze(label_2, -1)
            label_2_mat = th.cat((label_2, label_2), -1)
            for i in range(label_2.shape[0] - 1):
                if i == 0:
                    label_2_mat = label_2_mat
                else:
                    label_2_mat = th.cat((label_2_mat, label_2), -1)  # bs, bs

            mid_mat_ = (label_1_mat.t().eq(label_2_mat))
            mid_mat = mid_mat_.float()

            cosine_similarity = cosine_similarity / self.temperature
            cos_loss = th.log(th.clamp(F.softmax(cosine_similarity, dim=1), 1e-10, 1e10))
            cos_loss = cos_loss * mid_mat  # find the sample with the same label
            cos_loss = th.sum(cos_loss, dim=1) / th.sum(mid_mat + 1e-10, dim=1)  # bs

        cos_loss = -th.mean(cos_loss, dim=0)
        return cos_loss