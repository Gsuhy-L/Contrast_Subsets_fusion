from platform import node
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch as th
from torch_geometric.nn import GATv2Conv, global_add_pool,GINConv,GCNConv,GATConv
from model.Twitter.Atten import AttentionalAggregation
import torch_geometric.utils as utils
from torch_scatter import scatter_mean
import copy

device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')

class GAT(nn.Module):

    def __init__(self, config, num_class, x_dim, device):
        super(GAT, self).__init__()

        self.dropout_rate = config['dropout_p']
        self.hidden_size = config['hidden_size']
        self.n_layers = config["n_layers"]
        self.num_class = num_class
        self.device = device
        alpha = 0.3

        # self.gnn1 = GATv2Conv(x_dim, out_channels=self.hidden_size//8, dropout=self.dropout_rate, heads=8, negative_slope=alpha)
        self.gnn1 = GCNConv(x_dim,self.hidden_size)
        # self.gnn1 = GATConv(x_dim,self.hidden_size)

        #这里GNN2输入原来是self.hidden_size//8
        # self.gnn2 = GATv2Conv(self.hidden_size+x_dim, self.hidden_size, dropout=self.dropout_rate, concat=False, negative_slope=alpha)
        # self.gnn2 = GATConv(self.hidden_size+x_dim, self.hidden_size)

        self.gnn2 = GCNConv(x_dim+self.hidden_size,self.hidden_size)

        
        # self.BN1 = nn.BatchNorm1d(self.hidden_size+x_dim)
        # self.BN2 = nn.BatchNorm1d(self.hidden_size*2)

        self.relu = nn.ReLU()
        self.pool = global_add_pool
        self.fc = th.nn.Linear((self.hidden_size*2),4)
        # gate_nn=nn.Sequential(nn.Linear(self.hidden_size,self.hidden_size),
        #                           nn.Tanh(),
        #                           nn.Linear(self.hidden_size,1,bias=False))

        self.gate_nn = AttentionalAggregation(self.hidden_size)
        self.fc_out = nn.Sequential(
            nn.Linear(self.hidden_size*2, 100),
            nn.Dropout(self.dropout_rate),
            nn.ReLU(),
            nn.Linear(100, 1 if num_class==2 else num_class)
        )

        # self.reset_parameters()
    
    @staticmethod
    def MLP(in_channels: int, out_channels: int):
        return nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels),
        )
    
    def reset_parameters(self):
        for name, param in self.fc_out.named_parameters():
            if name.__contains__("weight"):
                init.xavier_normal_(param)

    # def node_encoder(self, x, sentence_tokens=None):
    #     if sentence_tokens!=None:
    #         X_feature = torch.tensor([]).to(self.device)
    #         for x0 in sentence_tokens:
    #             for x1 in x0:
    #                 x_emb = self.embedding(torch.tensor(x1).to(self.device))
    #                 x_encode = self.context_LSTM(x_emb)
    #                 X_feature = torch.cat((X_feature, torch.mean(x_encode[0], dim=0).unsqueeze(0)), 0)
    #
    #     else:
    #         X_feature = x
    #     return X_feature

    def global_graph_encoding(self, data, x, edge_index,edge_atten,states):
        #        node_rep1 = self.gnn1(x=x, edge_index=edge_index, edge_attr=edge_atten)
        #TODO 原式如上，好像假如加上edge_atten，权重无法回传？
        x1 = copy.copy(x)
        # print()
        print('--------------------'+states)
        print(edge_index)
        print(x)
        node_rep1 = self.gnn1(x=x, edge_index=edge_index)

        # x1 = copy.copy()
        x2 = copy.copy(node_rep1)
        rootindex = data.rootindex
        # root_extend = torch.FloatTensor(len(data.batch), x1.size(1)).to(device)
        root_extend = th.zeros(len(data.batch), x1.size(1)).to(device)
        # print(root_extend.dtype)
        # aa = torch.FloatTensor(3,2).to(device)
        # print(aa.dtype)
        batch_size = max(data.batch) + 1
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            temp = rootindex[num_batch]
            root_extend[index] = x1[temp]

        node_rep1 = th.cat((node_rep1,root_extend),1)
        #TODO 这里经过修改
        # node_rep1 =   self.relu(self.BN1(node_rep1))
        node_rep1 =   self.relu(node_rep1)
        node_rep1 = F.dropout(node_rep1,training=self.training)

        graph_output = self.gnn2(node_rep1, edge_index)
        graph_output = self.relu(graph_output)
        root_extend = th.zeros(len(data.batch), x2.size(1)).to(device)
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x2[rootindex[num_batch]]
        graph_output = th.cat((graph_output, root_extend), 1)



        # graph_output  = self.relu(self.BN2(graph_output))



        return graph_output

    def forward(self, data, x, edge_index, batch, edge_atten=None, states=None):

        # x = self.node_encoder(x, sentence_tokens)

        X_global = self.global_graph_encoding(data, x, edge_index, edge_atten, states=states)
        # X_feat = F.dropout(X_global, p = self.dropout_rate, training=self.training)
        # out = self.gate_nn(X_feat, batch)
        X_feat = X_global
        out = scatter_mean(X_feat,batch,dim=0)
        out = self.fc(out)
        #out = F.log_softmax(out)
        #
        # out = self.fc_out(out)
        return out ,X_feat
    
    # def get_emb(self, x, edge_index, batch, edge_atten=None):
    #     # x = self.node_encoder(x, sentence_tokens)
    #     X_feature = copy.deepcopy(x.detach())
    #
    #     X_global = self.global_graph_encoding(x, edge_index,  edge_atten=edge_atten)
    #     X_feat = F.dropout(X_global, p = self.dropout_rate, training=self.training)
    #
    #     return X_feat, X_feature

    # def get_pred_from_emb(self, emb, batch):
    #     return self.fc_out(self.gate_nn(emb, batch))
