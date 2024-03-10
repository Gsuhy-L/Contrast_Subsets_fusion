import random
import networkx as nx

# def graph_radom_walk(self, top_k_idx):
G = nx.DiGraph()

G.add_node(4)
G.add_nodes_from([0,1,2,3])

G.add_edge(0,1)

G.add_edges_from([(1,2),(2,3),(2,0),(3,0)])
chain_edge_adjs = G.adj
print(chain_edge_adjs)
# chain_edge_adjs = chain_graph(data.edge_index)
# que = root_node.children.copy()
# 原来方案中的链表中的节点ID都比模型中的下标打一
top_k_idx = [2]
graph_idx_lst = top_k_idx.copy()
graph_idx_que = top_k_idx.copy()
# que = chain_edge_adjs.copy()

while len(graph_idx_que) > 0:
    # cur is num node
    cur = graph_idx_que.pop()
    if random.random() >= 0.6:
        # mask[int(cur.idx) - 1] = 1
        current_idx_children = chain_edge_adjs[cur]
        for key, val in current_idx_children.items():
            # for child in cur.children:
            if key not in graph_idx_lst:
                graph_idx_lst.append(key)
                graph_idx_que.append(key)
print(graph_idx_lst)