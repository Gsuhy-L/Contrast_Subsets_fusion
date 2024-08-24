import sys,os
sys.path.append('/home/ubuntu/PyProjects_gsuhyl/PyProjects/BiGCN-source')
from Process.process import *
import torch as th
from torch_scatter import scatter_mean
import torch.nn.functional as F
import numpy as np
from tools.earlystopping import EarlyStopping
from torch_geometric.data import DataLoader
from tqdm import tqdm
from Process.rand5fold import *
from tools.evaluate import *
import torch.nn as nn
from torch_geometric.nn import GCNConv,GATConv
import copy

from model.Twitter.utils import get_data_loaders, get_model, MLP
from pathlib import Path
import yaml
import torch
from sampler import Sampler
#        self.extractor = ExtractorMLP(64, False, 0, 1)
class ExtractorMLP(nn.Module):

    def __init__(self, hidden_size, learn_edge_att = False, extractor_dropout_p = 0, Gnum_m = 1):
        super().__init__()
        self.learn_edge_att = learn_edge_att
        dropout_p = extractor_dropout_p

        if self.learn_edge_att:
            self.feature_extractor = MLP([hidden_size * 3, hidden_size * 4, hidden_size, 1], dropout=dropout_p)
        else:
            self.feature_extractor = MLP([hidden_size * 3, hidden_size * 2, hidden_size, Gnum_m], dropout=dropout_p)
        self.init_emb()

    def init_emb(self):
        # initrange = -1.5 / self.embedding_dim
        for m in self.modules():
            if isinstance(m, th.nn.Linear):
                th.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
    def forward(self, emb, edge_index, batch):
        # if self.learn_edge_att:
        #     col, row = edge_index
        #     f1, f2 = emb[col], emb[row]
        #     f12 = torch.cat([f1, f2], dim=-1)
        #     att_log_logits = self.feature_extractor(f12, batch[col])
        # else:
        # print(emb.shape)
        # print(e)
        print(emb.shape)
        # print(e)
        att_log_logits = self.feature_extractor(emb, batch)

        return att_log_logits

class Net(th.nn.Module):
    # def __init__(self,in_feats,hid_feats,out_feats,model_config):
    def __init__(self, in_feats, hid_feats, out_feats, model_config, device):
        super(Net, self).__init__()
        # self.TDrumorGCN = TDrumorGCN(in_feats, hid_feats, out_feats)
        # self.BUrumorGCN = BUrumorGCN(in_feats, hid_feats, out_feats)
        # self.UDrumorGCN = UDrumorGCN(in_feats, hid_feats, out_feats)

#edge_attr_dim,num_class,aux_info是原来论文中读取数据集的时候顺便读取出来的，
        #由于数据集合不同，这里我们使用人为定义的方式
        # self.model = get_model(x_dim, edge_attr_dim, num_class, aux_info['multi_label'], model_config, device)
        self.attn_model = get_model(in_feats, 1, 4, 1, model_config, device)

        self.extractor=ExtractorMLP(model_config['hidden_size'], model_config, model_config["Gnum_m"])
        self.device = device
        #模型训练中的参数只有get_model以及extractor两个模型参数，
        #这个采样模型中没有可学习的参数
        self.sampler_model = Sampler(model_config, device)

        # self.batch_size = batch_size

        self.epochs = model_config["epochs"]
        self.Gnum_m = int(model_config["Gnum_m"])

        # self.num_class = num_class
        self.multi_label = model_config['multi_label']
        self.learn_edge_att = model_config['learn_edge_att']
        # self.criterion = Criterion(self.num_class, self.multi_label)

        self.lr_decay_factor = float(model_config.get('lr_decay_factor', 0.5))
        self.lr_decay_step = int(model_config.get('lr_decay_step', 30))
        self.split_way = model_config['edge_split']
        self.pred_coef = float(model_config["pred_coef"])
        self.sampler_coef = float(model_config["sampler_coef"])
        self.counter_coef = float(model_config["counter_coef"])
        # self.optimizer_attn = optimizer
        #self.fc=th.nn.Linear((out_feats+hid_feats),4)


    # def forward_pass(self, data, X_features, atts):
    #     datalen = len(data[0])
    #
    #         # for hj in range(self.Gnum_m):
    #         #     clf_logits_f = torch.cat(
    #                 # (clf_logits_f, torch.mul(att_scores[hj].reshape(-1, 1), clf_logits[hj + 1].unsqueeze(0))), 0)
    #
    #     clf_logits = torch.zeros(datalen, self.num_class).unsqueeze(0).to(self.device)
    #     for index, subgraph in enumerate(data):
    #         if atts != None:
    #             att = atts[index]
    #         edge_att = None
    #         if len(subgraph.edge_index) != 0:
    #             if self.learn_edge_att:
    #                 edge_att = att
    #             else:
    #                 edge_att = self.lift_node_att_to_edge_att(att, subgraph.edge_index)
    #
    #         clf_logit = self.attn_model(subgraph.x.float(), subgraph.edge_index, subgraph.batch,
    #                                     edge_attr=subgraph.edge_attr, edge_atten=edge_att)
    #         clf_logits = torch.cat((clf_logits, clf_logit.unsqueeze(0)), 0)
    #         clf_logits_f = torch.zeros(datalen, self.num_class).unsqueeze(0).to(self.device)
    #
    #         # for hj in range(self.Gnum_m):
    #         #     clf_logits_f = torch.cat(
    #         #         (clf_logits_f, torch.mul(att_scores[hj].reshape(-1, 1), clf_logits[hj + 1].unsqueeze(0))), 0)
    #     clf_logits_f = clf_logits_f.sum(0)
    #     pred_loss = self.criterion(clf_logits_f, data[0].y)
    #     return pred_loss, clf_logits_f

#这里全图特征和子图特征应该是共用的
    #传进来的数据可以不用，但是一定不能没有
    def forward(self, data, X_features, state, atts):
        # print(len(data))
        x_out, x_clf_logit, x_feature = self.get_ori(data,states = 'one')
        #x_out, x_clf_logit = self.get_ori(data)

        #这里对卷积后的特征 change the feature's dim
        att_log_logit = self.extractor(x_feature,data.edge_index,data.batch)
        att = (att_log_logit).sigmoid()
        # if self.learn_edge_att:
        #    edge_att = att
        # else:
        #    edge_att = self.lift_node_att_to_edge_att(att, data.edge_index).to(self.device)
            #我们现在有数据，每个节点对应的图是什么，现在我们要按照图来去数据，
        #现在我们获取到了数据采样的节点
        #这里好像不一定会把根节点进行采样,这里加入要随机有走的话，需要改变BATCH的之。
        #或者先不挑选SUBGRAPH的batch，先进行随机游走。
        if state == "train":
            #sub_datas, sub_atts = self.sampler_model(data, x_feature, att, data.x)
            sub_datas, sub_atts = self.sampler_model(data, att, data.x)
            #self.sampler_model(data, att, data.x)

            x_clf_logit_two, x_feature = self.attn_model(sub_datas[0], sub_datas[0].x, sub_datas[0].edge_index, sub_datas[0].batch,states="two")
            #x_out_two = F.log_softmax(x_clf_logit_two, dim=1)
            # print(sub_atts)
            # print(low_out)
            # pred_loss, clf_logits_f = self.forward_pass(data, X_features, atts)

            #return pred_loss, clf_logits_f
            return x_out, x_clf_logit, x_clf_logit_two
        if state == "eval":
            #sub_datas, sub_atts = self.sampler_model(data, x_feature, att, data.x)
            #x_clf_logit_two, x_feature = self.attn_model(sub_datas[0], sub_datas[0].x, sub_datas[0].edge_index,
            #                                             sub_datas[0].batch)
            # x_out_two = F.log_softmax(x_clf_logit_two, dim=1)
            # print(sub_atts)
            # print(low_out)
            # pred_loss, clf_logits_f = self.forward_pass(data, X_features, atts)

            # return pred_loss, clf_logits_f
            return x_out, x_clf_logit, #x_clf_logit_two

    # def batch_split(self, embeddings, i):
    #     subGraph_embs = []
    #
    #     for sub in embeddings:
    #         subGraph_embs.append(sub.index_select(0, torch.tensor(np.arange(self.sub_split[i], self.sub_split[i + 1]))))
    #     return subGraph_embs

    @staticmethod
    def lift_node_att_to_edge_att(node_att, edge_index):
        src_lifted_att = node_att[edge_index[0]]
        dst_lifted_att = node_att[edge_index[1]]
        edge_att = src_lifted_att * dst_lifted_att
        return edge_att.sum(1, keepdim=True)



#该方法训练阶段和测试阶段唯一不同之处在于是否启用了eval
    def get_ori(self, data, states):
        #获取原始特征的GAT结果，以及复制后的节点特征
        #在大框架FORWARD中又掉用了注意力模型
        # ori_embs, X_feature = self.attn_model.get_emb(data.x.float(), data.edge_index, batch=data.batch)
        #对GAT结果进行降维
        # att_log_logit = self.extractor(ori_embs, data.edge_index, data.batch)
        # att = (att_log_logit).sigmoid()
        # if self.learn_edge_att:
        #     edge_att = att
        # else:
        #     edge_att = self.lift_node_att_to_edge_att(att, data.edge_index).to(self.device)

        #two feature are the same feature
        #这里是FORWARD函数的最终输出，
        x_clf_logit, x_feature = self.attn_model(data, data.x, data.edge_index, data.batch, states=states)


        # pred_loss2 = self.criterion(clf_logit, data.y)
        # pred_loss_ori = pred_loss2

        # return ori_embs, att, pred_loss_ori, X_feature
        x = F.log_softmax(x_clf_logit, dim=1)
        #classif ,out
        return x, x_clf_logit, x_feature

