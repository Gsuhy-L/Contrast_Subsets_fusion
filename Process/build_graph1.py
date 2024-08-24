import os
import random
import numpy as np
import pickle as pkl
import scipy.sparse as sp
import sys
from tqdm import tqdm
# from bert_model import BertLayer
from gensim.test.utils import datapath, get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec
import gensim
from collections import defaultdict
import torch

cwd = os.getcwd()


# def cos_sim(tensor_1, tensor_2):
#
#
#     normalized_tensor_1 = tensor_1 / tensor_1.norm(dim=-1, keepdim=True)
#     normalized_tensor_2 = tensor_2 / tensor_2.norm(dim=-1, keepdim=True)
#     return (normalized_tensor_1 * normalized_tensor_2).sum(dim=-1)





# def get_concept_windows_sim(project_name, dataname,word_id_map,sim):
#     raw_word_vec_file = '/home/ubuntu/PyProjects_gsuhyl/PyProjects/Knowledge_CKC_Bigcn/model/'+project_name+'/Process/word2vec_data/'+dataname+'/raw_word2vec.npy'
#     w2vec = np.load(raw_word_vec_file)
#     # 将词向量转化为Tensor
#     w2vec = torch.from_numpy(w2vec)
#     # CUDA接受float32，不接受float64
#     w2vec = w2vec.float()
#     doc_concept_windows = defaultdict(list)
#     doc_concept_entity=defaultdict(list)
#     content_path = '/home/ubuntu/PyProjects_gsuhyl/PyProjects/Knowledge_CKC_Bigcn/model/' + project_name + '/Twitter_data/data/' + dataname
#
#     with open(content_path + '/sig_concept.txt', 'r') as f_c:
#         for concept_line in f_c.readlines():
#             # TEXT
#             c_id = concept_line.strip('\n').split('\t')[0]
#             # print(line)
#             # SET
#             e_set = eval(concept_line.strip('\n').split('\t')[1])
#             if len(e_set) != 0:
#                 for k, c_set in e_set.items():
#                     doc_concept_entity[c_id].append(k)
#                     if len(c_set) != 0:
#                         for concept in c_set:
#                             con_sim = cos_sim(w2vec[word_id_map[k]],w2vec[word_id_map[concept]])
#                             # print(con_sim)
#                             if con_sim>= sim:
#                                 e_c = k + ' ' + concept
#
#                                 if c_id not in doc_concept_windows:
#
#                                     doc_concept_windows[c_id].append(e_c)
#                                 else:
#                                     doc_concept_windows[c_id].append(e_c)
#                         #
#                         #         print(e_set)
#                         #         print(e_c)
#                         #         print('------------------')
#                         #         print(e)
#                         # else:
#                         pass
#             else:
#                 pass
#     return doc_concept_windows, doc_concept_entity


# def get_ckc_windows_sim(project_name, dataname,word_id_map,sim,score_lim):
#     raw_word_vec_file = '/home/ubuntu/PyProjects_gsuhyl/PyProjects/Knowledge_CKC_Bigcn/model/'+project_name+'/Process/word2vec_data/'+dataname+'/raw_word2vec.npy'
#     w2vec = np.load(raw_word_vec_file)
#     # 将词向量转化为Tensor
#     w2vec = torch.from_numpy(w2vec)
#     # CUDA接受float32，不接受float64
#     w2vec = w2vec.float()
#     doc_ckc_windows_set = defaultdict(list)
#     ckc_path = cwd+'/Twitter_data'+"/data/"+dataname+'/bilearavg_condidate_score0.8'
#     ckc_set = get_ckc_word_score_lim(ckc_path,dataname)
#     # print(ckc_set)
#     # print(e)
#     for head, val in ckc_set.items():
#         # print(ckc_set)
#     # for ckc_file in ckc_files:
#     #     with open(ckc_file, 'r') as f_c:
#             # TEXT
#         for tail in val:
#             # if float(tail[1]) > float(score_lim):
#                 #print(head)
#                 #print(word_id_map[tail[0]])
#                 con_sim = cos_sim(w2vec[word_id_map[head]],w2vec[word_id_map[tail[0]]])
#                 if con_sim>= sim:
#                     e_c = head + ' ' + tail[0]
#                     if e_c not in doc_ckc_windows_set[head]:
#                         doc_ckc_windows_set[head].append(e_c)
#
#     return doc_ckc_windows_set


#要取读应iD的概念窗口
# def get_concept_windows(project_name, dataname):
#     content_path = '/home/ubuntu/PyProjects_gsuhyl/PyProjects/Knowledge_CKC_Bigcn/model/' + project_name + '/Twitter_data/data/' + dataname
#
#     doc_concept_windows = defaultdict(list)
#     with open(content_path + '/sig_concept.txt', 'r') as f_c:
#         for concept_line in f_c.readlines():
#             # TEXT
#             c_id = concept_line.strip('\n').split('\t')[0]
#             # print(line)
#             # SET
#             e_set = eval(concept_line.strip('\n').split('\t')[1])
#             if len(e_set) != 0:
#                 for k, c_set in e_set.items():
#                     if len(c_set) != 0:
#                         for concept in c_set:
#                             #con_sim = cos_sim(w2vec[k],w2vec[concept])
#
#                             e_c = k + ' ' + concept
#
#                             if c_id not in doc_concept_windows:
#
#                                 doc_concept_windows[c_id].append(e_c)
#                             else:
#                                 doc_concept_windows[c_id].append(e_c)
#                         #
#                         #         print(e_set)
#                         #         print(e_c)
#                         #         print('------------------')
#                         #         print(e)
#                         # else:
#                         pass
#             else:
#                 pass
#     return doc_concept_windows

# def get_ckc_concept_windows_list(project_name, dataname):
#     ckc_path = '/home/ubuntu/PyProjects_gsuhyl/PyProjects/Knowledge_CKC_Bigcn/model/' + project_name + '/Twitter_data/data/' +\
#                dataname+'/bilearavg_condidate_score0.8/'
#
#     doc_ckc_concept_windows = defaultdict(list)
#     ckc_files = [ckc_path + f for f in os.listdir(ckc_path)]  # 用idx.pkl中的idx排序
#     ckc_list = []
#     for ckc_file in ckc_files:
#         with open(ckc_file, 'r') as f_c:
#             for concept_line in f_c.readlines():
#                 concept_line = concept_line.strip('\n').split('\t')
#                 # print(concept_line)
#                 head, tail = concept_line[1], concept_line[2]
#                 # TEXT
#                 # c_id = concept_line.strip('\n').split('\t')[0]
#                 # print(line)
#                 # SET
#                 # e_set = eval(concept_line.strip('\n').split('\t')[1])
#                 # if len(e_set) != 0:
#                 #     for k, c_set in e_set.items():
#                 #         if len(c_set) != 0:
#                 for concept in concept_line:
#                     #con_sim = cos_sim(w2vec[k],w2vec[concept])
#
#                     e_c = head + ' ' + concept
#                     if e_c not in ckc_list:
#                         ckc_list.append(e_c)
#                     # if c_id not in doc_concept_windows:
#
#                 # pass
#
#     return ckc_list



no_url_doc_name_list = []
# no_url_doc_content_lst = []
# with open(content_path + '/source_tweets_del_url.txt', 'r') as f:
#     for line in f.readlines():
#         # print(line)
#         no_url_doc_name_list.append(line.split('\t')[0])
#         no_url_doc_content_lst.append(line.split('\t')[1].strip())

# print(doc_name_list)
# build corpus vocabulary



# word_set = set()
# flag = 0
# for doc_words in doc_content_list:
#     flag += 1
#     words = doc_words.split()
#     word_set.update(words)
# #:判断一共有多少个单词
# vocab = list(word_set)
# print(vocab)

# print(e)
# vocab_size = len(vocab)
# #每个单词给他一个编号
# word_id_map = {}
# for i in range(vocab_size):
#     word_id_map[vocab[i]] = i

# def stopwordslist():
#     """
#     创建停用词表
#     :return:
#     """
#     stopwords = [line.strip() for line in open('word2vec_data/stopword.txt', encoding='UTF-8').readlines()]
#     return stopwords

# def get_concepts_set(project_name, dataname):
#     doc_concept_set = defaultdict(list)
#     content_path = '/home/ubuntu/PyProjects_gsuhyl/PyProjects/Knowledge_CKC_Bigcn/model/' + project_name + '/Twitter_data/data/' + dataname
#
#     with open(content_path + '/sig_concept.txt', 'r') as f_c:
#         for concept_line in f_c.readlines():
#             # TEXT
#             c_id = concept_line.strip('\n').split('\t')[0]
#             # print(line)
#             # SET
#             e_set = eval(concept_line.strip('\n').split('\t')[1])
#             if len(e_set) != 0:
#                 for k, c_set in e_set.items():
#                     if len(c_set) != 0:
#                         for concept in c_set:
#
#                             if c_id not in doc_concept_set:
#
#                                 doc_concept_set[c_id].append(concept)
#                             else:
#                                 doc_concept_set[c_id].append(concept)
#                         #
#                         #         print(e_set)
#                         #         print(e_c)
#                         #         print('------------------')
#                         #         print(e)
#                         # else:
#                         pass
#             else:
#                 pass
#         return doc_concept_set

# def get_ckc_concepts_list(project_name, dataname):
#     ckc_path = '/home/ubuntu/PyProjects_gsuhyl/PyProjects/Knowledge_CKC_Bigcn/model/' + project_name + '/Twitter_data/data/' + dataname+\
#         '/bilearavg_condidate_score0.8/'
#     # ckc_list = []
#     # doc_ckc_concept_set = defaultdict(list)
#     ckc_files = [ckc_path + f for f in os.listdir(ckc_path)]  # 用idx.pkl中的idx排序
#     ckc_list = []
#     for ckc_file in ckc_files:
#         # with open(ckc_file,'r') as :
#         ckc_data = open(ckc_file,'r').readlines()
#         for line in ckc_data:
#             line = line.strip('\n').split('\t')
#             head, tail = line[1],line[2]
#             if tail not in ckc_list:
#                 ckc_list.append(tail)
#                 #print(tail)
#             # for concept in c_set:
#             #
#             #     if c_id not in doc_concept_set:
#             #
#             #         doc_concept_set[c_id].append(concept)
#             #     else:
#             #         doc_concept_set[c_id].append(concept)
#
#     return ckc_list


# def get_ckc_word(data_path,dataset):
#     ckc_files = [data_path + f for f in os.listdir(data_path)]  # 用idx.pkl中的idx排序
#     ckc_set = defaultdict(list)
#     for file in ckc_files:
#         file_data = open(file,'r').readlines()
#         for line in file_data:
#             #createdby	Durex	countable	0.95331
#             line = line.strip('\n').split('\t')
#             head, tail = line[1], line[2]
#             # if head not in ckc_set:
#             if tail not in ckc_set[head]:
#                 ckc_set[head].append(tail)
#         # file_data.c
#     return ckc_set

# def get_ckc_word_score_lim(data_path,dataset):
#     ckc_files = [data_path+"_entity/" +'entity_ckc.txt']  # 用idx.pkl中的idx排序
#     ckc_set = defaultdict(list)
#     for file in ckc_files:
#         file_data = open(file,'r').readlines()
#         for line in file_data:
#             #createdby	Durex	countable	0.95331
#             line = line.strip('\n').split('\t')
#             head, tail, score = line[0], line[1], line[2]
#             # head, tail= line[0], line[1]
#
#             # if head not in ckc_set:
#             if tail not in ckc_set[head]:
#                 ckc_set[head].append([tail,score])
#         # file_data.c
#     return ckc_set




def build_word2id(dataset):
    """
    将word2id词典写入文件中，key为word，value为索引
    :param file: word2id保存地址
    :return: None
    """
    # 加载停用词表
    # stopwords = stopwordslist()
    word2id = {'_PAD_': 0}
    # 文件路径
    #path = [Config.train_path, Config.val_path]
    # print(path)
    # 遍历训练集与验证集

    # 打开文件

    path = '../model/Twitter/data/'+dataset+'/clean_content.txt'
    file_path = '../model/Twitter/data/'+dataset+'word2id.txt'

    # concept_wins_set  = get_concept_windows(project_name,dataset)
    # con_set = get_concepts_set(project_name,dataset)
    #
    # ckc_wins_list = get_ckc_concept_windows_list(project_name,dataset)
    # ckc_list = get_ckc_concepts_list(project_name,dataset)

    with open(path, encoding='utf-8') as f:
        # 遍历文件每一行
        for line in f.readlines():
            # print(line)
            out_list = []
            # 去掉首尾空格并按照空格分割
            sp_id = line.strip('\n').split('\t')[0]
            sp = line.split('\t')[1].split()
            # 遍历文本部分每一个词
            # print(sp)
            for word in sp[:]:
                # 如果词不是停用词
                # if word not in stopwords:
                #     print(word)
                #     在字符串中找到正则表达式所匹配的所有子串，并返回一个列表，如果没有找到匹配的，则返回空列表。
                #     rt = re.findall('[a-zA-Z]+', word)
                #     # 如果word不等于制表符
                #     # print(rt)
                #     if word != '\t':
                #         # 如果词匹配的字符串为1，则继续遍历下一个词
                #         if len(rt) == 1:
                #             continue
                #         # 如果词匹配的字符串为0，则将这个词添加到out_list中
                #         else:
                    out_list.append(word)


            # 遍历out_list中的词
            # print(out_list)
            for word in out_list:
                # 如果这些词不在word2id字典的key中,则添加到word2id字典中
                if word not in word2id.keys():
                    word2id[word] = len(word2id)
            # print(word2id)

            #i添加单个单词
            # con_wins = concept_wins_set[sp_id]
            # for con_win_line in con_wins:
            #     con_win_words = con_win_line.split()
            #     for con_word in con_win_words:
            #         if con_word not in word2id.keys():
            #             word2id[con_word] = len(word2id)

            #添加整个单词组
            # doc_con_set = con_set[sp_id]
            # for sig_con in doc_con_set:
            #     if sig_con not in word2id.keys():
            #         word2id[sig_con] = len(word2id)

                    # 前边是按找文章iD进行添加的，这下边的是不用按照文章ID进行添加的，可以考虑之放在外边节约时间

                # ckc_wins = ckc_wins_list[sp_id]
            # for ckc_win_line in ckc_wins_list:
            #     ckc_win_words = ckc_win_line.split()
            #     for ckc_word in ckc_win_words:
            #         if ckc_word not in word2id.keys():
            #             word2id[ckc_word] = len(word2id)
            #
            # for sig_ckc in ckc_list:
            #     if sig_ckc not in word2id.keys():
            #         word2id[sig_ckc] = len(word2id)



            # print(word2id)

    for key in word2id:
        word2id[key] = int(word2id[key])

    # 构建id2word
    id2word = {}
    for key, val in word2id.items():
        id2word[val] = key
    # 打开输出文件并进行文件写入
    with open(file_path, 'w', encoding='utf-8') as f:
        # 遍历词典中的每一个词
        for w in word2id:
            f.write(w + '\t')
            f.write(str(word2id[w]))
            f.write('\n')
    #上面是为每个单词都, and ,we create a vetor for every short concepts.
    return word2id, id2word


def build_word2vec(dataset,word2id, save_to_path=None):
    """
    使用word2vec对单词进行编码
    :param fname: 预训练的word2vec.
    :param word2id: 语料文本中包含的词汇集.
    :param save_to_path: 保存训练语料库中的词组对应的word2vec到本地
    :return: 语料文本中词汇集对应的word2vec向量{id: word2vec}.
    """
    # 词的总数量
    n_words = max(word2id.values()) + 1
    # 加载预训练的词向量
    # glove_file = datapath('D:\\jetbrains_project\\PycharmProjects\\RDSPU\\datasets\\glove.6B.300d.txt')
    glove_file = datapath('/home/ubuntu/PyProjects_gsuhyl/PyProjects/Knowledge_CKC_Bigcn_2/model/hera_model_gat/Process/word2vec_data/Twitter16/glove.6B.300d.txt')

    tmp_file = get_tmpfile('word2vec.6B.300d.txt')
    glove2word2vec(glove_file,tmp_file)

    raw_word_vec_file = '../model/Twitter/data/'+dataset+'/raw_word2vec'

    model = gensim.models.KeyedVectors.load_word2vec_format(tmp_file,binary=False)
    # 初始化词向量
    #此项了的唯独等于个数诚意唯独
    word_vecs = np.array(np.random.uniform(-0.01, -0.01, [n_words, model.vector_size]))
    # 遍历每个单词
    for word in word2id.keys():
        try:
            # 构建词向量
            word_vecs[word2id[word]] = model[word]
            # print()
        except KeyError:
            pass
    # 将word_vecs保存到文件中

    #接下来为每个概念生成一个词向量
    # con_sets = get_concepts_set(project_name,dataset)
    # print(con_sets)
    # for doc_id, doc_con in con_sets.items():
    #
    #     for sig_con in doc_con:
    #
    #         sig_con_split = sig_con.split(' ')
    #
    #         con_vec = [word_vecs[word2id[con]] for con in sig_con_split]
    #         mean_con_vec = np.mean(con_vec,axis=0)
    #         word_vecs[word2id[sig_con]] = mean_con_vec
#为每个常识知识构建一个词向量

    # con_sets = get_concepts_set(project_name,dataset)
    # ckc_path = cwd+'/Twitter_data'+"/data/"+dataset+'/bilearavg_condidate_score0.8/'

    # ckc_sets = get_ckc_word(ckc_path,dataset)
    # for ckc_key,ckc_val in ckc_sets.items():
    #     for ckc in ckc_val:
    #         sig_ckc_split = ckc.split(' ')
    #
    #         ckc_vec = [word_vecs[word2id[con]] for con in sig_ckc_split]
    #         # print('+++++++++++++++++')
    #         mean_ckc_vec = np.mean(ckc_vec,axis=0)
    #         word_vecs[word2id[ckc]] = mean_ckc_vec

    if save_to_path:
        # with open(raw_word_vec_file, 'w', encoding='utf-8') as f:
        #     for vec in word_vecs:
        #         vec = [str(w) for w in vec]
        #         f.write(' '.join(vec))
        #         f.write('\n')
        # with open(raw_word_vec_file, 'w') as f1:
        #     f1.write(word_vecs)
        np.save(raw_word_vec_file,word_vecs)
    # 返回word_vecs数组
    # print('word_vecs',word_vecs.shape)
    return word_vecs



def get_label(dataset):
    labelPath = os.path.join('../model/Twitter/data/',\
                             dataset + "/label.txt")
    labelset_nonR, labelset_f, labelset_t, labelset_u = ['news', 'non-rumor'], ['false'], ['true'], ['unverified']

    print("loading tree label")
    event, y = [], []
    l1 = l2 = l3 = l4 = 0
    labelDic = {}
    for line in open(labelPath):
        line = line.rstrip()
        label, eid = line.split(':')[0], line.split(':')[1]
        label=label.lower()
        event.append(eid)
        if label in labelset_nonR:
            labelDic[eid]=0
            l1 += 1
        if label  in labelset_f:
            labelDic[eid]=1
            l2 += 1
        if label  in labelset_t:
            labelDic[eid]=2
            l3 += 1
        if label  in labelset_u:
            labelDic[eid]=3
            l4 += 1
    return labelDic
    # print(len(labelDic))
    # labelPath = os.path.join('.../model/Twitter/data/',\
    #                              dataset + "/label.txt")
    # labelset_nonR, labelset_t, = ['non-rumours'], ['rumours']
    #
    # print("loading tree label")
    # #event, y = [], []
    # l1 = l2 = l3 = l4 = 0
    # labelDic = {}
    # for line in open(labelPath):
    #     line = line.rstrip()
    #     label, eid = line.split(':')[0], line.split(':')[1]
    #     label=label.lower()
    #     #event.append(eid)
    #     if label in labelset_nonR:
    #         labelDic[eid]=0
    #         l1 += 1
    #
    #     if label  in labelset_t:
    #         labelDic[eid]=1
    #         l2 += 1

    # return labelDic

def build_graph(word_id_map, id_word_map, dataset, doc_content_list,doc_name_list,seq_lenth, window_size,weighted_graph,truncate,MAX_TRUNC_LEN):
    # load pre-trained word embeddings
    word_embeddings_dim = 300
    word_embeddings = {}
    # concept_windows, concept_entity = get_concept_windows_sim(project_name, dataset, word_id_map, sim_con)
    # ckc_windows = get_ckc_windows_sim(project_name,dataset,word_id_map,sim_ckc,score_lim)
    # print(ckc_windows)
#defaultdict(<class 'list'>, {'canadian': ['canadian prominent'], 'ottawa': ['ottawa people'], 'cnn': [
    # with open("../datasets/" + dataset+'/glove.6B.' + str(
    #         word_embeddings_dim) + 'd.txt', 'r') as f:
    #     for line in f.readlines():
    #         data = line.split()
    #         word_embeddings[str(data[0])] = list(map(float, data[1:]))

    # glove_file = datapath('D:\\jetbrains_project\\PycharmProjects\\RDSPU\\datasets\\glove.6B.300d.txt')
    glove_file = datapath('/home/ubuntu/PyProjects_gsuhyl/PyProjects/Knowledge_CKC_Bigcn_2/model/hera_model_gat/Process/word2vec_data/Twitter16/glove.6B.300d.txt')

    tmp_file = get_tmpfile('word2vec.6B.300d.txt')
    glove2word2vec(glove_file,tmp_file)

    # raw_word_vec_file = '../datasets/'+dataset+'/raw_word2vec'
    word_embeddings = gensim.models.KeyedVectors.load_word2vec_format(tmp_file,binary=False)
    oov = {}
    for v in range(len(word_id_map)):
        oov[v] = np.random.uniform(-0.01, 0.01, word_embeddings_dim)
    i=0
    for word in word_id_map.keys():

        if word in word_embeddings:
            # print()
            oov[word_id_map[word]] = word_embeddings[word]

            # print()
        else:
            # print(word)
            # print(i)
            i+=1
    labels_lst = get_label(dataset)
    x_adj = []
    x_feature = []
    # y = []
    doc_len_list = []
    vocab_set = set()
    # print(doc_name_list)


    for i in tqdm(range(len(doc_content_list))):
        doc_name = doc_name_list[i]
        # post_text = no_url_doc_content_lst[i]
        # print(doc_name)
        doc_words = doc_content_list[i].split()
        if truncate:
            doc_words = doc_words[:MAX_TRUNC_LEN]
        doc_len = len(doc_words)

               # print(e)

        #去冲
        doc_vocab = list(set(doc_words))
        doc_nodes = len(doc_vocab)
        # doc_len_list.append(doc_nodes)
        vocab_set.update(doc_vocab)

        doc_word_id_map = {}
        for j in range(doc_nodes):
            #在单个文件中，给每个词一个id
            doc_word_id_map[doc_vocab[j]] = j


        # sliding windows
        windows = []
        if doc_len <= window_size:
            windows.append(doc_words)
        else:
            for j in range(doc_len - window_size + 1):
                window = doc_words[j: j + window_size]
                windows.append(window)

        word_pair_count = {}
        # print(len(windows))
        for window in windows:
            for p in range(1, len(window)):
                for q in range(0, p):
                    word_p = window[p]
                    word_p_id = word_id_map[word_p]  # doc_word_id_map[word_p]
                    word_q = window[q]
                    word_q_id = word_id_map[word_q]  # doc_word_id_map[word_q]
                    if word_p_id == word_q_id:
                        continue
                    word_pair_key = (word_p_id, word_q_id)
                    # word co-occurrences as weights
                    if word_pair_key in word_pair_count:
                        word_pair_count[word_pair_key] += 1.
                    else:
                        word_pair_count[word_pair_key] = 1.
                    # bi-direction
                    word_pair_key = (word_q_id, word_p_id)
                    if word_pair_key in word_pair_count:
                        word_pair_count[word_pair_key] += 1.
                    else:
                        word_pair_count[word_pair_key] = 1.

        row = []
        col = []
        weight = []
        features = []

        for key in word_pair_count:
            p = key[0]
            q = key[1]
            #里边存放的ID是总的次表的ID，需要把它专程单个次表的ID

            row.append(doc_word_id_map[id_word_map[p]])  # p
            col.append(doc_word_id_map[id_word_map[q]])  # q
            weight.append(word_pair_count[key] if weighted_graph else 1.)
        adj = sp.csr_matrix((weight, (row, col)), shape=(doc_nodes, doc_nodes))



        temp_features_id = []
        feature_ids =[]
        tmp_features = []
        post_features = []
        #这里生成节点的特征。
        # print(doc_word_id_map)
        # print(e)
        for k, v in sorted(doc_word_id_map.items(), key=lambda x: x[1]):
            # print(k)
            # print(word_id_map[k])
            # tmp_features.append(word_embeddings[k] if k in word_embeddings else oov[k])
            temp_features_id.append(word_id_map[k])
            tmp_features.append(oov[word_id_map[k]])
            # tmp_features.append(k)
        # print(feature_ids.dtype)
        # print(e)
        feature_ids.append(temp_features_id)
        # print('=====================')
        features = tmp_features



        y = labels_lst[doc_name]
        # x_adj.append(adj)
        # x_feature.append(features)


        # return None
        # features = np.array(features.cpu())
        feature_ids = np.array(feature_ids)
        features = np.array(features)
        # post_features = np.array(feature_post.cpu())
        adj_lst = []
        adj_lst.append(adj)
        adj_lst = np.array(adj_lst)
        # print(adj.toarray())
        # print(e)
        y=np.array(y)
        doc_array = prepare_data(dataset, word_id_map, doc_name, seq_lenth)
        # print(doc_array)
        # print(feature_ids)

        # print(e)
        np.savez(os.path.join('../model/Twitter/data/'+dataset+'_Knowledge/',
                               doc_name + '.npz'), x=features, doc_array = doc_array,  edgeindex=adj_lst, \
                                  feature_ids = feature_ids, y = y)

        # data = np.load(os.path.join('/home/ubuntu/PyProjects_gsuhyl/PyProjects/Knowledge_CKC_Bigcn/Twitter_data/',\
        #                       'data/' + dataset + 'graph/' + doc_name + '.npz'), allow_pickle=True)

def text_to_array(dataset, word2id, seq_lenth, doc_id):
    """
    有标签文本转为索引数字模式
    :param word2id: word2id
    :param seq_lenth: 句子最大长度
    :param path: 文件路径
    :return:
    """
    # 存储标签
    lable_array = []
    # 句子索引初始化
    i = 0
    sa = []
    path = '../model/Twitter/data/'+dataset+'/clean_content.txt'

    # 获取句子个数
    # print(word2id)
    # print(e)
    with open(path, encoding='utf-8') as f1:
        # 打开文件并遍历文件每一行
        for l1 in f1.readlines():
            # 返回分割后的字符串列表
            # print(l1.split('\t')[0])
            # print(e)
            s = l1.split('\t')[1].strip().split()
            # 去掉标签
            s1 = s[:]
            # 单词转索引数字
            new_s = [word2id.get(word, 0) for word in s1]
            # 存储由索引数字表示的文本列表
            # print(new_s)

            sa.append(new_s)
        # print(len(sa))

    # print(label_dic)
    content_idx_dict = {}
    with open(path, encoding='utf-8') as f:
        # 初始化句子array；行：句子个数 列：句子长度
        sentences_array = np.zeros(shape=(len(sa), seq_lenth),dtype=np.int)
        # print(sentences_array)
        # print(e)
        # 遍历每一句话
        for line in f.readlines():
            # print(line)
            sl1 = line.split('\t')[1].strip().split()
            # 去掉标签
            sen = sl1[:]
            # 单词转索引数字,不存在则为0
            new_sen = [word2id.get(word, 0) for word in sen]
            # print(new_sen)
            # 转换为(1,sen_len)
            new_sen_np = np.array(new_sen).reshape(1, -1)
            # print(new_sen_np)

            # 补齐每个句子长度，少了就直接赋值,0填在前面。
            # np.size，返回沿给定轴的元素数
            # print(np.size(new_sen_np, 1))
            if np.size(new_sen_np, 1) < seq_lenth:
                sentences_array[i, seq_lenth - np.size(new_sen_np, 1):] = new_sen_np[0, :]
            # 长了进行截断
            else:
                sentences_array[i, 0:seq_lenth] = new_sen_np[0, 0:seq_lenth]
            # print(sentences_array)
            content_idx = line.split('\t')[0]
            content_idx_dict[content_idx] = sentences_array[i]
            i = i + 1


                # 标签
                # lable = int(sl1[0])
                # lable = label_dic[line.split('\t')[0]]
                # print(line.split('\t')[0])
                # print(lable)
                # print(e)
                # lable_array.append(lable)
            # else:
                # print(e)
        doc_sentences_array=[]
        # print(doc_id)


        doc_sentences_array.append(list(content_idx_dict[doc_id]))
        # 返回索引模式的文本以及标签
        # print('===============')
        # print(doc_lst)
        # print(lable_array)
        # print(e)
        # print(doc_lst)
        # print(lable_array)
        # print(e)
        # print(doc_sentences_array)
        # print(e)
    return np.array(doc_sentences_array)

def prepare_data(dataset, w2id, doc_id, seq_lenth):
    """
    得到数字索引表示的句子和标签
    :param w2id: word2id
    :param train_path: 训练文件路径
    :param val_path: 验证文件路径
    :param test_path: 测试文件路径
    :param seq_lenth: 句子最大长度
    :return:
    """
    # 对训练集、验证集、测试集处理，将文本转化为由单词索引构成的array
    train_array = text_to_array(dataset=dataset, word2id=w2id, seq_lenth=seq_lenth, doc_id = doc_id)
    # pri
    # print(train_lst)
    # print(train_lable)
    # print(e)
    # val_array = text_to_array(dataset=dataset, word2id=w2id, seq_lenth=seq_lenth, doc_id=val_lst)
    return train_array



def build_graph_twitter(dataset):

    # cwd=os.getcwd()
    # if len(sys.argv) < 2:
    #     sys.exit("Use: python knowledge_build_graph.py <dataset>")

    # settings
    # datasets = ['mr', 'ohsumed', 'R8', 'R52', 'TREC', 'ag_news', 'WebKB', 'SST1', 'SST2']


    # if dataset not in datasets:
    #     sys.exit("wrong dataset name")
# /
    seq_length = 20
    print('using default seq  lenght = 25')

    try:
        window_size = int(sys.argv[3])
    except:
        window_size = 3
        print('using default window size = 3')

    try:
        weighted_graph = bool(sys.argv[4])
    except:
        weighted_graph = False
        print('using default unweighted graph')
    # seq_length = 25
    # window_size = 3
    truncate = False  # whether to truncate long document
    MAX_TRUNC_LEN = 35

    print('loading raw data')

    doc_content_list = []
    doc_name_list = []
    content_path = '../model/Twitter/data/'+ dataset

    # for file in os.listdir(content_path):
    with open(content_path + '/clean_content.txt', 'r') as f:
        for line in f.readlines():
            # print(line)
            doc_name_list.append(line.split('\t')[0])
            doc_content_list.append(line.split('\t')[1].strip())

    print('building graphs for training')
    word_id_map, id_word_map = build_word2id(dataset)

    word2vec = build_word2vec(dataset,word_id_map, True)
#def build_graph(word_id_map, id_word_map, project_name, dataset, doc_content_list,doc_name_list,seq_lenth,window_size,weighted_graph,truncate,MAX_TRUNC_LEN):

    build_graph(word_id_map, id_word_map, dataset,doc_content_list,doc_name_list, seq_length, window_size,weighted_graph,truncate,MAX_TRUNC_LEN)

build_graph_twitter(dataset='Twitter16')