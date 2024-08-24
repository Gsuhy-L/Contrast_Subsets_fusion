import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch_scatter
from tqdm import tqdm
import copy
import networkx as nx
# from random import random
import random
from collections import defaultdict

EPSILON = torch.tensor(torch.finfo(torch.float32).tiny)



class Sampler(nn.Module):

    def __init__(self, device):
        super(Sampler, self).__init__()

        self.device = device
        # self.Nnum_k=float(sampler_config["Nnum_k"])
        # self.Gnum_m=int(sampler_config["Gnum_m"])
        # self.temperature=float(sampler_config["temperature"])
        # self.separate=sampler_config["separate"]
        self.Nnum_k = 0.6
        self.neg_Nnum_k = 0.2
        self.Gnum_m = 1
        self.temperature = 1
        self.separate = False
        # self.nn = nn.Sequential(
        #     nn.Linear(in_features=5000, out_features=64),
        #     nn.ReLU(),
        #     nn.Linear(in_features=64, out_features=4)
        # )
        # self.batch_size = batch_size

    def forward(self, data, att, X_feature):
        self.data = data
        self.X_feature = X_feature
        #calculate the nodes of every batch
        nodes_num = torch.bincount(self.data.batch)
        #calculate the start idx of each batch
        self.nodes_num = torch.cat([torch.tensor([0]).to(self.device), torch.cumsum(nodes_num, 0)])
        # print(self.nodes_num)

        self.subNnum_k = [int(np.ceil(d * self.Nnum_k)) for d in nodes_num.cpu().numpy()]
        self.neg_subNnum_k = [int(np.ceil(d * self.neg_Nnum_k)) for d in nodes_num.cpu().numpy()]
        # self.ori_embs = ori_embs
        self.att = att
        #这里开头为什么要设置成0
        #计算每个子节点开始的下表
        self.subNodes = torch.cat(
            [torch.tensor([0]).to(self.device), torch.cumsum(torch.tensor(self.subNnum_k).to(self.device), 0)])
        self.neg_subNodes = torch.cat(
            [torch.tensor([0]).to(self.device), torch.cumsum(torch.tensor(self.neg_subNnum_k).to(self.device), 0)])
        sub_datas, all_neg_top_ks = self.graph_sampling(self.att)

        sub_atts = []
        for idxs in self.all_top_ks:
            sub_atts.append(self.att.index_select(0, torch.tensor(idxs).to(self.device)))

        return sub_datas, sub_atts, all_neg_top_ks


    def gumbel_keys(self, att):
        # sample some gumbels
        uniform = torch.rand_like(att)
        z = -torch.log(-torch.log(uniform + EPSILON))
        att_g = att + z
        return att_g

    def batch_softmax(self, att_g):
        exp_logits = torch.exp(torch.tensor(att_g) / self.temperature)
        partition = torch_scatter.scatter_sum(exp_logits, self.edges_batch, 0)
        partition = partition.index_select(0, self.edges_batch.T.squeeze())
        softmax_logits = exp_logits / (partition + EPSILON)
        return softmax_logits

    def continuous_topk(self, att_g, i):

        khot_list = torch.tensor([]).to(self.device)
        onehot_approx = torch.zeros_like(att_g)

        for _ in range(self.subNnum_k[i]):
            khot_mask = torch.maximum(1.0 - onehot_approx, EPSILON)
            att_g = att_g + torch.log(khot_mask)
            onehot_approx = nn.functional.softmax(att_g / self.temperature, 0)
            khot_list = torch.cat((khot_list, onehot_approx.T), dim=0)

        if self.separate:
            return khot_list
        else:
            return torch.sum(khot_list, 0)

    # def getDeepwalkSeqs(self,  :

    def chain_graph(self, edge_index):
        G = nx.DiGraph()
        for idx, edge in enumerate(edge_index.T):
            src = edge[0].item()
            dst = edge[1].item()
            G.add_edge(src, dst)
        chain_edge_adjs = G.adj
        return chain_edge_adjs

    def chain_new_graph(self, edge_index):
        node_child_map = defaultdict(list)
        for i in range(len(edge_index[0])):
            parent_node = edge_index[0][i]
            child_node = edge_index[1][i]
            node_child_map[parent_node].append(child_node)

        return node_child_map

    def graph_radom_walk(self, top_k_idx):
        chain_edge_adjs = self.chain_new_graph(self.data.edge_index)
        # que = root_node.children.copy()
        # 原来方案中的链表中的节点ID都比模型中的下标打一
        graph_idx_lst = top_k_idx.copy()
        graph_idx_que = top_k_idx.copy()
        # que = chain_edge_adjs.copy()
        print(graph_idx_que)

        while len(graph_idx_que) > 0:
            # cur is num node
            # cur = graph_idx_que.delete()
            cur = graph_idx_que.item(0)
            graph_idx_que = graph_idx_que[1:]

            if random.random() >= 0.6:
                # mask[int(cur.idx) - 1] = 1
                current_idx_children = chain_edge_adjs[cur]
                for key in current_idx_children:
                    # for child in cur.children:
                    if key not in graph_idx_lst:
                        graph_idx_lst = np.append(graph_idx_lst, key)
                        graph_idx_que = np.append(graph_idx_que, key)
        return graph_idx_lst

    def graph_radom_walk1(self, top_k_idx):
        chain_edge_adjs = self.chain_new_graph(self.data.edge_index)
        # que = root_node.children.copy()
        # 原来方案中的链表中的节点ID都比模型中的下标打一
        graph_idx_lst = top_k_idx.copy()
        graph_idx_que = top_k_idx.copy()
        # que = chain_edge_adjs.copy()
        # print(graph_idx_que)
        que = graph_idx_que
        while len(que) > 0:
            cur = que.pop()
            if random.random() >= 0.6:
                # mask[int(cur.idx) - 1] = 1
                current_idx_children = chain_edge_adjs[cur]
                # graph_idx_que.append(key)
                for key in current_idx_children:
                    # for child in cur.children:
                    if key not in graph_idx_lst:
                        graph_idx_lst.extend(key)
                        que.extend(key)

                # for child in cur.children:
                #     que.append(child)
        # while len(graph_idx_que) > 0:
        # cur is num node
        # cur = graph_idx_que.delete()
        # cur = graph_idx_que.item(0)
        # graph_idx_que = graph_idx_que[1:]

        # if random.random() >= 0.6:
        #     # mask[int(cur.idx) - 1] = 1
        #     current_idx_children = chain_edge_adjs[cur]
        #     for key in current_idx_children:
        #     # for child in cur.children:
        #         if key not in graph_idx_lst:
        #             graph_idx_lst = np.append(graph_idx_lst,key)
        #             graph_idx_que = np.append(graph_idx_que,key)
        return graph_idx_lst

    def get_permute_edge(self, top_k_idxs):
        aug_ratio = 0.4
        #_, edge_num = data.edge_index.size()
        # 这里是统计所有的节点数量，
        node_num = len(top_k_idxs)
        permute_num = int(node_num * aug_ratio)
        unif = torch.ones(2, node_num)



        #edge_index = data.edge_index
        #这个

        # 这里是所有的数据，对所有的数据进行随机抽样,这里对节点进行采样，即：头部节点采样多次，尾部节点采样多次，然后返回
        # print(unif)
        # print(permute_num)
        add_edge_idx = unif.multinomial(7, replacement=True)
        #需要找到下表对应的之
        # permute_edge = []
        permute_head = []
        permute_tail = []
        for head_idx in add_edge_idx[0]:
            permute_head.append(top_k_idxs[head_idx])
        for tail_idx in add_edge_idx[1]:
            permute_tail.append(top_k_idxs[tail_idx])
        # # 随机抽样

        #unif = torch.ones(edge_num)
        # 边的数量减去绕动的数量，剩下的代表不进行变换的数量#
        # 设置保留边的比例
        #keep_edge_idx = unif.multinomial((edge_num - permute_num), replacement=False)

        # edge_index = edge_index[:, keep_edge_idx]
        # 按照第二个维度（边的下标）进行保留。
        #edge_index = torch.cat((edge_index[:, keep_edge_idx], add_edge_idx), dim=1)
        #data.edge_index = edge_index
        return permute_head, permute_tail

    def cos_sim(self, tensor_1, tensor_2):

        normalized_tensor_1 = tensor_1 / tensor_1.norm(dim=-1, keepdim=True)
        normalized_tensor_2 = tensor_2 / tensor_2.norm(dim=-1, keepdim=True)
        return (normalized_tensor_1 * normalized_tensor_2).sum(dim=-1)

    def all_reconstruct(self, random_top_k_idxs, att, copy_top_k_idxs, copy_sub_top_k_idxs, copy_sub_sig_data_batch_lst):
        # data_copy = []
        # print("top-k:")
        # print(random_top_k_idxs)

        data_copy = copy.deepcopy(self.data)

        edge_sel = []
        edge_tuple = []
        for idx, edge in enumerate(data_copy.edge_index.T):
            edge_tuple.append(edge)
            src = edge[0].item()
            dst = edge[1].item()
            if (src in random_top_k_idxs) and (dst in random_top_k_idxs):
                # idx represent the edge number code
                edge_sel.append(idx)
        # Z = self.nn(x)
        Z = copy.copy(att)
        # pi = torch.matmul(Z,torch.t(Z))
        row_ib = []
        col_ib = []

        restruct_top_k_idxs = []

        for sub_top_k_id in range(len(copy_sub_top_k_idxs)):
            sub_top_k_idxs = copy_sub_top_k_idxs[sub_top_k_id]
            sub_sig_data_batch_lst = copy_sub_sig_data_batch_lst[sub_top_k_id]
            # print(Z)

            nodes_redict = {i: id for i, id in enumerate(sub_top_k_idxs)}
            # print("nodes_redict")
            # print(nodes_redict)
            #对邻居矩阵中两个列表中的值都进行了替换，根据新节点重新开始编号。

            vector_matrix = Z
            indices = sub_top_k_idxs
            selected_vectors = vector_matrix[indices]
            norms = torch.norm(selected_vectors, dim=1)
            dot_product_matrix = torch.matmul(selected_vectors, selected_vectors.t())
            consine_sim_matrix = dot_product_matrix/(norms[:None]*norms[None,:])
            indices = torch.nonzero(consine_sim_matrix>0.8)
            edges_list = [[nodes_redict[tup[0].item()], nodes_redict[tup[1].item()]] for tup in indices]
            restruct_top_k_idxs.extend(edges_list)
            # print(restruct_top_k_idxs)
            # print(indices)
            # for sub_top_k_idxs_i in sub_top_k_idxs:
            #     for sub_top_k_idxs_j in sub_top_k_idxs:
            #         vector_matrix = torch.stack(att)
                #     x_sim = self.cos_sim(att[sub_top_k_idxs_i], att[sub_top_k_idxs_j])
                #     if x_sim > 0.6:

        # for restruct_top_k_idx_pair in restruct_top_k_idxs:
        #
        #     if (restruct_top_k_idx_pair[0], restruct_top_k_idx_pair[1]) not in edge_tuple:
        #         row_ib.append(restruct_top_k_idx_pair[0])
        #         col_ib.append(restruct_top_k_idx_pair[1])
        #     if (restruct_top_k_idx_pair[1], restruct_top_k_idx_pair[0]) not in edge_tuple:
        #         row_ib.append(restruct_top_k_idx_pair[1])
        #         col_ib.append(restruct_top_k_idx_pair[0])
        # print(self.data.edge_index.T)
        # print(e)
        # print(restruct_top_k_idxs)
        # print(e)
        restruct_top_k_idxs = torch.tensor(restruct_top_k_idxs).to(self.device)
        # print(self.data.edge_index.T)
        # print(restruct_top_k_idxs[0])
        # print(self.data.edge_index.T[0])
        elements_to_add = torch.tensor([elem.tolist() for elem in restruct_top_k_idxs if elem not in self.data.edge_index.T])
        elements_to_add = elements_to_add.T
        # print(elements_to_add)
        # print(elements_to_add)
            # print(row_ib)
                # print(row_ib)
                # print(col_ib)
            # print(sub_sig_data_batch_lst)
            # print(sub_top_k_idxs)
            # vector_ma
            # print(indices)

            # all_permute_edge = [[],[]]

        #todo try to reconnect the realtion of top k

        # for sub_copy_idxs in copy_sub_top_k_idxs:
        #     permute_head, permute_tail = self.get_permute_edge(sub_copy_idxs)
        #     all_permute_edge[0].extend(permute_head)
        #     all_permute_edge[1].extend(permute_tail)
        # all_permute_edge_tuple = list(zip(all_permute_edge[0], all_permute_edge[1]))
        # edge_index_tuple = list(zip(data_copy.edge_index[0], data_copy.edge_index[1]))


        #add permute edge
        #todo 这个判断花费了大量的时间
        sel_per_edge_tuple = []
        #for per_edge in all_permute_edge_tuple:
            #if per_edge not in edge_index_tuple:
                # pass
                #sel_per_edge_tuple.append(per_edge)
        # #print(zip(*sel_per_edge_tuple))
        #sel_per_edge_head, sel_per_edge_tail = tuple(zip(*sel_per_edge_tuple))
        #print(sel_per_edge_head)
        #sel_per_edge_head = list(sel_per_edge_head)
        #sel_per_edge_tail = list(sel_per_edge_tail)

        # data_copy.edge_attr=data_copy.edge_attr[edge_sel]
        # select correspond node batch
        random_top_k_idxs.sort()
        # print("sort:")
        # print(random_top_k_idxs)

        #data_copy.x = data_copy.x[random_top_k_idxs, :]
        # data_copy.rootindex =

        data_copy.edge_index = data_copy.edge_index[:, edge_sel]
        copy_edge_index_0 = data_copy.edge_index[0]
        copy_edge_index_1 = data_copy.edge_index[1]

        # print(self.data.edge_index)
        new_edge_index = [[],[]]
        new_edge_index_row = torch.cat((copy_edge_index_0, elements_to_add[0].to(self.device)), dim=0)
        # print(torch.cat((copy_edge_index_0, elements_to_add[0].to(self.device)), dim=0))
        # print(data_copy.edge_index[0])
        new_edge_index_col = torch.cat((copy_edge_index_1, elements_to_add[1].to(self.device)), dim=0)
        # print(new_edge_index)
        data_copy.edge_index = torch.cat((new_edge_index_row.unsqueeze(0),new_edge_index_col.unsqueeze(0)), dim=0)

        #print(sel_per_edge_head)
        #print(sel_per_edge_tail)
        #data_copy.edge_index[0].extend(sel_per_edge_head)
        #data_copy.edge_index[1].extend(sel_per_edge_tail)

        #data_copy.batch = data_copy.batch[random_top_k_idxs]

        # 这个地方很关键，这里对节点iD进行了重新排列
        # {6: 0, 7: 1, 9: 2, 10: 3, 11: 4, 14: 5, 18: 6, 20: 7, 21: 8, 23: 9, 24: 10, 27: 11, 28: 12, 29: 13, 36: 14, 37: 15, 40: 16, 42: 17, 43: 18, 44: 19, 46: 20, 48: 21, 49: 22, 50: 23, 51: 24, 53: 25, 54: 26, 55: 27, 58: 28, 60: 29, 61: 30, 62: 31, 64: 32, 66: 33, 70: 34, 71: 35}
        #nodes_redict = {id: i for i, id in enumerate(random_top_k_idxs)}
        # print("nodes_redict")
        # print(nodes_redict)
        # 对邻居矩阵中两个列表中的值都进行了替换，根据新节点重新开始编号。
        #edges_list = [[nodes_redict[tup[0].item()], nodes_redict[tup[1].item()]] for tup in data_copy.edge_index.T]
        # print("tup")
        # print(data_copy.edge_index.T[0][0].item())
        #data_copy.edge_index = torch.from_numpy(np.array(edges_list).T).to(self.device)
        #tmp_rootindex = data_copy.rootindex
        # print(tmp_rootindex)
        # print(nodes_redict)
        #new_rootindex = [nodes_redict[rootindex.item()] for rootindex in data_copy.rootindex]

        # random_top_k_idxs.append(data_copy)
        #data_copy.rootindex = new_rootindex

        return data_copy


    def reconstruct(self, top_k_idxs):
        # print(data.x.shape)
        # chain_graph = self.chain_graph(self.data.edge_index)
        all_random_top_k_idxs = []
        sub_data_copy = []
        # print(len(top_k_idxs))
        # print(top_k_idxs)
        # print(self.data.rootindex)
        # print("==========================")
        for sub_top_k_idxs_i in range(len(top_k_idxs)):
            sub_top_k_idxs = top_k_idxs[sub_top_k_idxs_i].copy()
            # print(sub_top_k_idxs.dtype)
            root_index = np.array(self.data.rootindex[sub_top_k_idxs_i].cpu())
            # 如果
            if root_index not in sub_top_k_idxs:
                sub_top_k_idxs = np.append(sub_top_k_idxs, root_index)

            random_top_k_idxs = self.graph_radom_walk(sub_top_k_idxs)
            all_random_top_k_idxs.extend(random_top_k_idxs)
        sub_data_copy = self.all_reconstruct(all_random_top_k_idxs)
        return sub_data_copy

    def reconstruct1(self, top_k_idxs, sub_top_k_idxs, sub_sig_data_batch_lst, att):
        # print(data.x.shape)
        # chain_graph = self.chain_graph(self.data.edge_index)
        all_random_top_k_idxs = []
        sub_data_copy = []
        # print(len(top_k_idxs))
        # print(top_k_idxs)
        # print(self.data.rootindex)
        # print("==========================")
        # for sub_top_k_idxs_i in range(len(top_k_idxs)):
        # sub_top_k_idxs = top_k_idxs[sub_top_k_idxs_i].copy()
        # print(sub_top_k_idxs.dtype)
        copy_top_k_idxs = top_k_idxs.copy()
        copy_sub_sig_data_batch_lst = sub_sig_data_batch_lst.copy()
        copy_sub_top_k_idxs = sub_top_k_idxs.copy()
        # root_index = np.array(self.data.rootindex[sub_top_k_idxs_i].cpu())
        # 如果
        # if root_index not in sub_top_k_idxs:
        #     sub_top_k_idxs = np.append(sub_top_k_idxs, root_index)
        #todo,这里是先进行随机有走，然后在进行联通边选择
        #don't know the root node whether in the graph
        copy_top_k_idxs.extend(np.array(self.data.rootindex.cpu()))
        random_top_k_idxs = self.graph_radom_walk1(copy_top_k_idxs)
        # all_random_top_k_idxs.extend(random_top_k_idxs)

        sub_data_copy = self.all_reconstruct(random_top_k_idxs, att, copy_top_k_idxs, copy_sub_top_k_idxs, copy_sub_sig_data_batch_lst)
        return sub_data_copy

    def sample_subset(self, att, i):
        '''
        Args:
            att (Tensor): Float Tensor of weights for each element. In gumbel mode
                these are interpreted as log probabilities
            i (int): index of batch
        '''
        att_g = self.gumbel_keys(att)
        return self.continuous_topk(att_g, i)

    def graph_sampling(self, att):

        sub_datas = [0] * self.Gnum_m
        counter_sub_graphs_idxs = []
        sub_X_features = []
        self.all_top_ks = []
        self.all_neg_top_ks = []

        for j in range(self.Gnum_m):
            top_k_idxs = []
            sig_data_batch_lst = []
            neg_top_k_idxs = []
            sub_top_k_idxs = []
            sub_neg_top_k_idxs = []
            sub_sig_data_batch_lst = []



            # counter_top_k_idxs = []
            # edge_tuple = []
            # for idx, edge in enumerate(self.data.edge_index.T):
            #     edge_tuple.append(edge)
            # 现在我们已经有了所有节点的卷积结果，接下来我们要吧不同图的node out

            for i in range(max(self.data.batch) + 1):
                # 他的意思是找到节点
                # print(self.nodes_num)
                sig_data_batch = att.index_select(1, torch.tensor(j).to(self.device)).index_select(0,
                                                                                  torch.tensor(
                                                                                      np.arange(
                                                                                          self.nodes_num[
                                                                                              i].cpu(),
                                                                                          self.nodes_num[
                                                                                              i + 1].cpu())).to(
                                                                                      self.device))
                selected_idxs = self.sample_subset(sig_data_batch, i)

                # 这个地方的数据的类型是NUMPY类型，我们要ARRAY数据类型没有EXTEND和APPEND方法
                #torch.argsort返回按照之进行排序后的下表
                top_k_idx = np.sort(np.array(torch.argsort(selected_idxs).cpu())[-self.subNnum_k[i]:]) + \
                            self.nodes_num[i].item()
                neg_top_k_idx = np.sort(np.array(torch.argsort(selected_idxs).cpu())[:-self.neg_subNnum_k[i]]) + \
                            self.nodes_num[i].item()
                # counter_top_k_idx = np.sort(np.array(torch.argsort(selected_edges).cpu())[: -self.subNnum_k[i]])+self.nodes_num[i].item()
                top_k_idxs.extend(top_k_idx)
                sig_data_batch_lst.extend(sig_data_batch)

                copy_top_k_idxs = top_k_idxs
                row_ib =[]
                col_ib = []


                # for sub_top_k_idxs_i in copy_top_k_idxs:
                #     for sub_top_k_idxs_j in copy_top_k_idxs:
                #         x_sim = self.cos_sim(att[sub_top_k_idxs_i], att[sub_top_k_idxs_j])
                #         if x_sim > 0.6:
                #             if (sub_top_k_idxs_i, sub_top_k_idxs_j) not in edge_tuple:
                #                 row_ib.append(sub_top_k_idxs_i)
                #                 col_ib.append(sub_top_k_idxs_j)
                #             if (sub_top_k_idxs_j, sub_top_k_idxs_i) not in edge_tuple:
                #                 row_ib.append(sub_top_k_idxs_j)
                #                 col_ib.append(sub_top_k_idxs_i)
                #     print(row_ib)
                #     print(col_ib)
                # vector_matrix = torch.stack()


                sub_top_k_idxs.append(top_k_idx)
                neg_top_k_idxs.extend(neg_top_k_idx)
                sub_neg_top_k_idxs.append(neg_top_k_idx)
                sub_sig_data_batch_lst.append(sig_data_batch)

                # counter_top_k_idxs.extend(counter_top_k_idx)

            # sub_X_features.append(self.X_feature[top_k_idxs,:])
            #里边存放的都是节点的下标
            self.all_top_ks.append(top_k_idxs)

            self.all_neg_top_ks.append(neg_top_k_idxs)
            # 这里要尝试return different graph
            # sub_data = self.reconstruct(sub_top_k_idxs)
            sub_data = self.reconstruct1(top_k_idxs, sub_top_k_idxs, sub_sig_data_batch_lst, att)

            sub_datas[j] = sub_data
            # counter_sub_graphs_idxs.append(counter_top_k_idxs)

        # return sub_graphs, counter_sub_graphs_idxs, sub_X_features
        # return sub_graphs, sub_X_features
        return sub_datas, self.all_neg_top_ks



    # def restruct_top_k(self,x, edgeidx):
    #     Z =self.nn(x)
    #     pi = torch.sigmoid(torch.matmul(Z, torch.t(Z)))
    #     # edgeindex_ib = []
    #     row_ib = []
    #     col_ib = []
    #     for i in range(len(pi)):
    #         for j in range(len(pi)):
    #         # u, v = row[i], col[i]
    #             if self.cos_sim(pi[i, j]) > 0.6:
    #                 # edgeindex_ib.append(i)
    #                 row_ib.append(edgeidx[i])
    #                 col_ib.append(edgeidx[j])
    #     tem_ib = col_ib
    #     col_ib.extend(row_ib)
    #     row_ib.extend(tem_ib)
    #
    #             # col_ib = [col[i] for i in edgeindex_ib]
    #     drop_edgeindex = [row_ib, col_ib]
    #     return torch.LongTensor(drop_edgeindex).cuda()
