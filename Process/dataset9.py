import os
import numpy as np
import torch
import random
from torch.utils.data import Dataset
from torch_geometric.data import Data
from Process.getTwittergraph import Node_tweet

def sparse_mx_to_torch(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    # print(type(sparse_mx))
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    # print('----------------------------------------------------')
    if len(sparse_mx.row) == 0 or len(sparse_mx.col) == 0:
        print(sparse_mx.row, sparse_mx.col)
        print('data bug')
        print('sparse_mx.data',sparse_mx.data)
        print('sparse_mx.shape',sparse_mx.shape)
    # print('--------row col-------:',type(sparse_mx.row),type(sparse_mx.col)) dp.ndarray
    if np.NAN in sparse_mx.data:
        print('有NaN数据')
    # with open('test_matraix_data.txt','a',encoding='utf-8')as f:
    #     v_list = []
    #     for v in sparse_mx.data:
    #         v_list.append(str(v))
    #     f.writelines(v_list)
    # assert sparse_mx.data.sum() == np.float32(len(sparse_mx.row))
    # print('data sparse_mx.data.sum',sparse_mx.data.sum(),type(sparse_mx.data.sum()))
    # print('data len(sparse_mx.row)',len(sparse_mx.row),type(len(sparse_mx.row)))
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data.astype(np.float32))
    shape = torch.Size(sparse_mx.shape)
    return indices,values,shape


class GraphDataset(Dataset):
    def __init__(self, fold_x, treeDic,lower=2, upper=100000, droprate=0,
                 data_path=os.path.join('..','..', 'data', 'Weibograph')):
        self.fold_x = list(filter(lambda id: id in treeDic and len(treeDic[id]) >= lower and len(treeDic[id]) <= upper, fold_x))
        self.treeDic = treeDic
        self.data_path = data_path
        self.droprate = droprate

    def __len__(self):
        return len(self.fold_x)

    def __getitem__(self, index):
        id =self.fold_x[index]
        data=np.load(os.path.join(self.data_path, id + ".npz"), allow_pickle=True)
        edgeindex = data['edgeindex']
        if self.droprate > 0:
            row = list(edgeindex[0])
            col = list(edgeindex[1])
            length = len(row)
            poslist = random.sample(range(length), int(length * (1 - self.droprate)))
            poslist = sorted(poslist)
            row = list(np.array(row)[poslist])
            col = list(np.array(col)[poslist])
            new_edgeindex = [row, col]
        else:
            new_edgeindex = edgeindex
        return Data(x=torch.tensor(data['x'],dtype=torch.float32),
                    edge_index=torch.LongTensor(new_edgeindex),
             y=torch.LongTensor([int(data['y'])]), root=torch.LongTensor(data['root']),
             rootindex=torch.LongTensor([int(data['rootindex'])]))

def collate_fn(data):
    return data

class BiGraphDataset(Dataset):
    def __init__(self, fold_x, treeDic,lower=2, upper=100000, tddroprate=0,budroprate=0,
                 data_path=os.path.join('..','..', 'data', 'Weibograph')):
        print(len(fold_x))
        self.fold_x = list(filter(lambda id: id in treeDic and len(treeDic[id]) >= lower and len(treeDic[id]) <= upper, fold_x))
        print(len(self.fold_x))
        # print()
        self.treeDic = treeDic
        self.data_path = data_path
        self.tddroprate = tddroprate
        self.budroprate = budroprate

    def __len__(self):
        return len(self.fold_x)

    def __getitem__(self, index):
        id =self.fold_x[index]
        data=np.load(os.path.join(self.data_path, id + ".npz"), allow_pickle=True)
        edgeindex = data['edgeindex']
        if self.tddroprate > 0:
            row = list(edgeindex[0])
            col = list(edgeindex[1])
            length = len(row)
            poslist = random.sample(range(length), int(length * (1 - self.tddroprate)))
            poslist = sorted(poslist)
            row = list(np.array(row)[poslist])
            col = list(np.array(col)[poslist])
            new_edgeindex = [row, col]
        else:
            new_edgeindex = edgeindex

        burow = list(edgeindex[1])
        bucol = list(edgeindex[0])
        if self.budroprate > 0:
            length = len(burow)
            poslist = random.sample(range(length), int(length * (1 - self.budroprate)))
            poslist = sorted(poslist)
            row = list(np.array(burow)[poslist])
            col = list(np.array(bucol)[poslist])
            bunew_edgeindex = [row, col]
        else:
            bunew_edgeindex = [burow,bucol]
        return Data(x=torch.tensor(data['x'],dtype=torch.float32),
                    edge_index=torch.LongTensor(new_edgeindex),BU_edge_index=torch.LongTensor(bunew_edgeindex),
             y=torch.LongTensor([int(data['y'])]), root=torch.LongTensor(data['root']),
             rootindex=torch.LongTensor([int(data['rootindex'])]))

class BiGraphDataset1(Dataset):
    def __init__(self, fold_x, treeDic,lower=2, upper=100000, tddroprate=0,budroprate=0,dataname='Twitter16',
                 data_path=os.path.join('..','..', 'data', 'Twitter16graph')):
        self.fold_x = list(filter(lambda id: id in treeDic and len(treeDic[id]) >= lower and len(treeDic[id]) <= upper, fold_x))
        self.treeDic = treeDic
        self.data_path = data_path
        self.knowledge_data_path = './data/' + dataname + '_Knowledge'
        self.tddroprate = tddroprate  # rate of dropedge
        self.budroprate = budroprate

    def __len__(self):
        return len(self.fold_x)

    def __getitem__(self, index):
        id =self.fold_x[index]
        # self.data_path="/home/ubuntu/PyProjects_gsuhyl/PyProjects/RDEA-main-init/RDEA-main/Process/data/Twitter15graph/"
        data=np.load(os.path.join(self.data_path, id + ".npz"), allow_pickle=True)  # load the graph
        edgeindex = data['edgeindex'] # edgeindex->2 x num_nodes
        if self.tddroprate > 0: # utilize dropedge
            row = list(edgeindex[0])
            col = list(edgeindex[1])
            length = len(row)
            poslist = random.sample(range(length), int(length * (1 - self.tddroprate)))
            poslist = sorted(poslist)
            row = list(np.array(row)[poslist])
            col = list(np.array(col)[poslist])
            drop_edgeindex = [row, col]


        tree = self.treeDic[id]
        index2node = {}
        for i in tree:
            node = Node_tweet(idx=i)
            index2node[i] = node

        for j in tree:
            indexC = j
            indexP = tree[j]['parent']
            nodeC = index2node[indexC]
            ## not root node ##
            if not indexP == 'None':
                nodeP = index2node[int(indexP)]
                nodeC.parent = nodeP
                nodeP.children.append(nodeC)
            ## root node ##
            else:
                rootindex = indexC - 1
                root_index = nodeC.index
                root_word = nodeC.word

        mask = [0 for _ in range(len(index2node))]
        mask[rootindex] = 1
        root_node = index2node[int(rootindex + 1)]
        que = root_node.children.copy()
        while len(que) > 0:
            cur = que.pop()
            if random.random() >= 0.6:
                mask[int(cur.idx) - 1] = 1
                for child in cur.children:
                    que.append(child)
        knowledge_data=np.load(os.path.join(self.knowledge_data_path, id + ".npz"), allow_pickle=True)

        knowledge_edgeindex = knowledge_data['edgeindex'][0]
        # print(edgeindex.toarray())
        # print(knowledge_edgeindex)

        knowledge_edgeindex , knowledge_edgevalue, knowledge_edgeshape = sparse_mx_to_torch(knowledge_edgeindex)
        # if self.knowledge_droprate > 0:
        #     knowledge_row = list(knowledge_edgeindex[0])
        #     knowledge_col = list(knowledge_edgeindex[1])
        #     knowledge_length = len(knowledge_row)
        #     knowledge_poslist = random.sample(range(knowledge_length), int(knowledge_length * (1 - self.knowledge_droprate)))
        #     knowledge_poslist = sorted(knowledge_poslist)
        #     knowledge_row = list(np.array(knowledge_row)[knowledge_poslist])
        #     knowledge_col = list(np.array(knowledge_col)[knowledge_poslist])
        #     knowledge_new_edgeindex = [knowledge_row, knowledge_col]
        # else:
        knowledge_new_edgeindex = knowledge_edgeindex
        # print(id)
        knowledge_new_feature_ids = knowledge_data['feature_ids'].reshape(-1,1)

        if self.tddroprate > 0:
            return Data(x=torch.tensor(data['x'],dtype=torch.float32), # x->num_nodes x features ; edge_index->2 x num_nodes(top down)
                        x_pos=torch.tensor(data['x_pos'], dtype=torch.float32),
                        mask = torch.tensor(mask, dtype=torch.bool),
                        edge_index=torch.LongTensor(edgeindex),#BU_edge_index=torch.LongTensor(bunew_edgeindex), # BU_edge_index->2 x num_nodes(bottom up))
                        dropped_edge_index=torch.LongTensor(drop_edgeindex),
                 y=torch.LongTensor([int(data['y'])]), root=torch.LongTensor(data['root']), # y->label
                 rootindex=torch.LongTensor([int(data['rootindex'])])), \
                   Data(x=torch.tensor(knowledge_data['x'], dtype=torch.float32),
                        feature_ids=torch.tensor(knowledge_new_feature_ids),
                        doc_array=torch.tensor(knowledge_data['doc_array']),
                        # new_x = torch.tensor(data['x'], dtype=torch.float32),
                        # post_x = torch.tensor(data['post_x'],dtype=torch.float32),
                        edge_index=torch.LongTensor(knowledge_new_edgeindex),
                        edge_value=torch.FloatTensor(knowledge_edgevalue),
                        y=torch.LongTensor([int(knowledge_data['y'])]))


        else:
            return Data(x=torch.tensor(data['x'], dtype=torch.float32),
                        # x->num_nodes x features ; edge_index->2 x num_nodes(top down)
                        x_pos=torch.tensor(data['x_pos'], dtype=torch.float32),
                        mask=torch.tensor(mask, dtype=torch.bool),
                        edge_index=torch.LongTensor(edgeindex),
                        y=torch.LongTensor([int(data['y'])]), root=torch.LongTensor(data['root']),  # y->label
                        rootindex=torch.LongTensor([int(data['rootindex'])])), \
                   Data(x=torch.tensor(knowledge_data['x'], dtype=torch.float32),
                        feature_ids=torch.tensor(knowledge_new_feature_ids),
                        doc_array=torch.tensor(knowledge_data['doc_array']),
                        # new_x = torch.tensor(data['x'], dtype=torch.float32),
                        # post_x = torch.tensor(data['post_x'],dtype=torch.float32),
                        edge_index=torch.LongTensor(knowledge_new_edgeindex),
                        edge_value=torch.FloatTensor(knowledge_edgevalue),
                        y=torch.LongTensor([int(knowledge_data['y'])]))







class UdGraphDataset(Dataset):
    def __init__(self, fold_x, treeDic,lower=2, upper=100000, droprate=0,
                 data_path=os.path.join('..','..','data', 'Weibograph')):
        self.fold_x = list(filter(lambda id: id in treeDic and len(treeDic[id]) >= lower and len(treeDic[id]) <= upper, fold_x))
        self.treeDic = treeDic
        self.data_path = data_path
        self.droprate = droprate

    def __len__(self):
        return len(self.fold_x)

    def __getitem__(self, index):
        id =self.fold_x[index]
        data=np.load(os.path.join(self.data_path, id + ".npz"), allow_pickle=True)
        edgeindex = data['edgeindex']
        row = list(edgeindex[0])
        col = list(edgeindex[1])
        burow = list(edgeindex[1])
        bucol = list(edgeindex[0])
        row.extend(burow)
        col.extend(bucol)
        if self.droprate > 0:
            length = len(row)
            poslist = random.sample(range(length), int(length * (1 - self.droprate)))
            poslist = sorted(poslist)
            row = list(np.array(row)[poslist])
            col = list(np.array(col)[poslist])
        new_edgeindex = [row, col]

        return Data(x=torch.tensor(data['x'],dtype=torch.float32),
                    edge_index=torch.LongTensor(new_edgeindex),
             y=torch.LongTensor([int(data['y'])]), root=torch.LongTensor(data['root']),
             rootindex=torch.LongTensor([int(data['rootindex'])]))
