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
from RDEA_model_R5 import Net1, Classfier1
from tools.evaluate import *
import torch.nn as nn
from torch_geometric.nn import GCNConv,GATConv
import copy
from model_utils import ExtractorMLP, Net
from model.Twitter.utils import get_data_loaders, get_model, MLP
from pathlib import Path
import yaml
import torch
from sampler import Sampler
from SCLR5 import SCL
import GCL.losses as L
from GCL.models import DualBranchContrast
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

Graph_loss = DualBranchContrast(loss=L.JSD(), mode='G2G')

def train_GCN(treeDic, x_test, x_train, model_config, UDdroprate,lr, weight_decay,patience,n_epochs,batchsize,dataname,iter,contrast_lr):
    # model = Net(5000,64,64,model_config,device).to(device)
    unsup_model = Net1(64, 3).to(device)
    unsup_epoch = 0
    for unsup_epoch in range(25):

        optimizer = th.optim.Adam(unsup_model.parameters(), lr=0.0005, weight_decay=1e-4)
        unsup_model.train()
        traindata_list, _ = loadBiData1(dataname, treeDic, x_train+x_test, x_test, 0.2, 0.2)
        train_loader = DataLoader(traindata_list, batch_size=batchsize, shuffle=True, num_workers=4)
        batch_idx = 0
        loss_all = 0
        tqdm_train_loader = tqdm(train_loader)
        for Batch_data in tqdm_train_loader:
            optimizer.zero_grad()
            Batch_data = Batch_data.to(device)
            loss = unsup_model(Batch_data)
            loss_all += loss.item() * (max(Batch_data.batch) + 1)
            # loss_all += loss.item()

            loss.backward()
            optimizer.step()
            batch_idx = batch_idx + 1
        loss = loss_all / len(train_loader)
    name = "best_pre_"+dataname +"_4unsup" + ".pkl"
    th.save(unsup_model.state_dict(), name)
    print('Finished the unsuperivised training.', '  Loss:', loss)
    print("Start classify!!!")
    unsup_model.eval()
    # BU_params=list(map(id,model.BUrumorGCN.conv1.parameters()))
    # BU_params += list(map(id, model.BUrumorGCN.conv2.paramet498991()))
    # base_params=filter(lambda p:id(p) not in BU_params,model.parameters())
    # optimizer = th.optim.Adam([
    #     {'params':base_params},
    #     {'params':model.BUrumorGCN.conv1.parameters(),'lr':lr/5},
    #     {'params': model.BUrumorGCN.conv2.parameters(), 'lr': lr/5}
    # ], lr=lr, weight_decay=weight_decay)
    #
    # optimizer = torch.optim.Adam(list(extractor.parameters()) + list(model.parameters()), lr=float(model_config["lr"]),
    #                              weight_decay=float(model_config["weight_decay"]))

    # UD_params = list(map(id, model.attn_model.gnn1.parameters()))
    # UD_params += list(map(id, model.attn_model.gnn2.parameters()))
    # base_params = filter(lambda p: id(p) not in UD_params, model.parameters())
    # optimizer = th.optim.Adam([
    #     {'params': base_params},
    #     {'params': model.attn_model.gnn1.parameters(), 'lr': float(lr) / 5},
    #     {'params': model.attn_model.gnn2.parameters(), 'lr': float(lr) / 5}
    # ], lr=float(lr), weight_decay=float(weight_decay))
    model = Classfier1(64*3,64,4).to(device)
    opt = th.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)


    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    early_stopping = EarlyStopping(patience=10, verbose=True)
    for epoch in range(n_epochs):

        traindata_list, testdata_list = loadBiData1(dataname, treeDic, x_train, x_test, UDdroprate, UDdroprate)

        train_loader = DataLoader(traindata_list, batch_size=batchsize, shuffle=True, num_workers=5)
        test_loader = DataLoader(testdata_list, batch_size=batchsize, shuffle=True, num_workers=5)
        avg_loss = []
        avg_acc = []
        batch_idx = 0
        tqdm_train_loader = tqdm(train_loader)
        model.train()
        unsup_model.train()
        for Batch_data in tqdm_train_loader:
            Batch_data.to(device)
            _, Batch_embed = unsup_model.encoder(Batch_data.x, Batch_data.edge_index, Batch_data.batch)
            # out_labels,x_clf_logit, x_clf_logit_two = model(Batch_data,Batch_data.x,state="train",atts = None)
            #out_labels,x_clf_logit = model(Batch_data,Batch_data.x,state="train",atts = None)

            # loss_mix = F.mse_loss(x_clf_logit,x_clf_logit_two)
            # print(Batch_data.y)
            # print(e)
            # subdata,out_labels, init_out= model(Batch_embed, Batch_data, states = 'train', datatype="initdata")
            out_data, out_labels= model(Batch_embed, Batch_data, states = 'train')

            # out_labels = model(Batch_embed, Batch_data, states='test')

            #_, Sub_Batch_embed = unsup_model.encoder(subdata[0].x, subdata[0].edge_index, subdata[0].batch)
            # sub_out= model(Sub_Batch_embed, subdata[0], states = 'train', datatype="subdata")


            # Batch_embed_x = scatter_mean(Batch_embed, Batch_data.batch, dim=0)
            # Sub_batch_embed_x = scatter_mean(Sub_Batch_embed, subdata[0].batch, dim=0)

            #loss_CrossGraph = Graph_loss(g1=init_out, g2=sub_out, batch=Batch_data.batch)
            #loss_IntraGraph = Graph_loss(g1=Batch_embed_x, g2=Batch_embed_x, batch=Batch_data.batch)

            SCL_model = SCL(0.1).to(device)
            # loss_IntraGraph = SCL_model(out_data,Sub_batch_embed_x,Batch_data.y)

            finalloss=F.nll_loss(out_labels,Batch_data.y)

            # out_data.retain_grad()  # we need to get gradient w.r.t low-resource embeddings
            # finalloss.backward(retain_graph=True)
            # unnormalized_noise = out_data.grad.detach_()
            # for p in model.parameters():
            #     if p.grad is not None:
            #         p.grad.detach_()
            #         p.grad.zero_()
            # norm = unnormalized_noise.norm(p=2, dim=-1)
            # normalized_noise = unnormalized_noise / (norm.unsqueeze(dim=-1) + 1e-10)  # add 1e-10 to avoid Nan
            #
            # noise_norm = 1.5
            # alp = 0.5
            #
            # target_noise = noise_norm * normalized_noise
            # noise_x_ = out_data + target_noise
            zeta = 0.02
            model_copy = copy.deepcopy(model)
            with th.no_grad():
                for param, param_copy in zip(model.parameters(), model_copy.parameters()):
                    noise = th.randn_like(param)
                    noise = zeta * noise / noise.norm(p=2)
                    param_copy.data.add_(noise)
            # loss_para = F.mse_loss(model(Batch_embed, Batch_data)[0], model_copy(Batch_embed, Batch_data)[0])
            copy_out_data = model_copy(Batch_embed, Batch_data, states='train')[0]

            out_data_scloss = SCL_model(out_data, out_data, Batch_data.y)
            out_noise_scloss = SCL_model(out_data, out_data, Batch_data.y)

            noise_scloss = SCL_model(copy_out_data,copy_out_data , Batch_data.y)
            con_loss = (out_data_scloss+out_noise_scloss+noise_scloss)/3

            #这里对比的嵌入是,这种对比方法指的是同一个节点或/图在不同的视图中被视为正对，
            loss=finalloss+0.3*con_loss#+contrast_lr*loss_CrossGraph#contrast_lr*loss_IntraGraph
            opt.zero_grad()
            loss.backward()
            avg_loss.append(loss.item())
            opt.step()

            _, pred = out_labels.max(dim=-1)
            correct = pred.eq(Batch_data.y).sum().item()
            train_acc = correct / len(Batch_data.y)
            avg_acc.append(train_acc)
            print("Iter {:03d} | Epoch {:05d} | Batch{:02d} | Train_Loss {:.4f}| Train_Accuracy {:.4f}".format(iter,epoch, batch_idx,
                                                                                                 loss.item(),
                                                                                                 train_acc))
            batch_idx = batch_idx + 1

        train_losses.append(np.mean(avg_loss))
        train_accs.append(np.mean(avg_acc))

        temp_val_losses = []
        temp_val_accs = []
        temp_val_Acc_all, temp_val_Acc1, temp_val_Prec1, temp_val_Recll1, temp_val_F1, \
        temp_val_Acc2, temp_val_Prec2, temp_val_Recll2, temp_val_F2, \
        temp_val_Acc3, temp_val_Prec3, temp_val_Recll3, temp_val_F3, \
        temp_val_Acc4, temp_val_Prec4, temp_val_Recll4, temp_val_F4 = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
        model.eval()
        tqdm_test_loader = tqdm(test_loader)
        for Batch_data in tqdm_test_loader:
            Batch_data.to(device)
            # val_out = model(Batch_data)
            # val_out, x_clf_logit = model(Batch_data,Batch_data.x,state = "eval",atts = None)
            #todo
            #E_, Batch_embed = unsup_model.encoder(Batch_data.x, Batch_data.edge_index, Batch_data.batch)
            Batch_embed = unsup_model.encoder.get_embeddings(Batch_data)

            out_labels= model(Batch_embed, Batch_data, states = 'test')
            val_loss  = F.nll_loss(out_labels, Batch_data.y)

            temp_val_losses.append(val_loss.item())
            _, val_pred = out_labels.max(dim=1)
            correct = val_pred.eq(Batch_data.y).sum().item()
            val_acc = correct / len(Batch_data.y)
            Acc_all, Acc1, Prec1, Recll1, F1, Acc2, Prec2, Recll2, F2, Acc3, Prec3, Recll3, F3, Acc4, Prec4, Recll4, F4 = evaluation4class(
                val_pred, Batch_data.y)
            temp_val_Acc_all.append(Acc_all), temp_val_Acc1.append(Acc1), temp_val_Prec1.append(
                Prec1), temp_val_Recll1.append(Recll1), temp_val_F1.append(F1), \
            temp_val_Acc2.append(Acc2), temp_val_Prec2.append(Prec2), temp_val_Recll2.append(
                Recll2), temp_val_F2.append(F2), \
            temp_val_Acc3.append(Acc3), temp_val_Prec3.append(Prec3), temp_val_Recll3.append(
                Recll3), temp_val_F3.append(F3), \
            temp_val_Acc4.append(Acc4), temp_val_Prec4.append(Prec4), temp_val_Recll4.append(
                Recll4), temp_val_F4.append(F4)
            temp_val_accs.append(val_acc)
        val_losses.append(np.mean(temp_val_losses))
        val_accs.append(np.mean(temp_val_accs))
        print("Epoch {:05d} | Val_Loss {:.4f}| Val_Accuracy {:.4f}".format(epoch, np.mean(temp_val_losses),
                                                                           np.mean(temp_val_accs)))

        res = ['acc:{:.4f}'.format(np.mean(temp_val_Acc_all)),
               'C1:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc1), np.mean(temp_val_Prec1),
                                                       np.mean(temp_val_Recll1), np.mean(temp_val_F1)),
               'C2:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc2), np.mean(temp_val_Prec2),
                                                       np.mean(temp_val_Recll2), np.mean(temp_val_F2)),
               'C3:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc3), np.mean(temp_val_Prec3),
                                                       np.mean(temp_val_Recll3), np.mean(temp_val_F3)),
               'C4:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc4), np.mean(temp_val_Prec4),
                                                       np.mean(temp_val_Recll4), np.mean(temp_val_F4))]
        print('results:', res)
        early_stopping(np.mean(temp_val_losses), np.mean(temp_val_accs), np.mean(temp_val_F1), np.mean(temp_val_F2),
                       np.mean(temp_val_F3), np.mean(temp_val_F4), model, 'BiGCN', dataname)
        accs =np.mean(temp_val_accs)
        F1 = np.mean(temp_val_F1)
        F2 = np.mean(temp_val_F2)
        F3 = np.mean(temp_val_F3)
        F4 = np.mean(temp_val_F4)
        if early_stopping.early_stop:
            print("Early stopping")
            accs=early_stopping.accs
            F1=early_stopping.F1
            F2 = early_stopping.F2
            F3 = early_stopping.F3
            F4 = early_stopping.F4
            break
    return train_losses , val_losses ,train_accs, val_accs,accs,F1,F2,F3,F4


local_config = yaml.safe_load((Path("./configs/Graph_Twitter.yml")).open('r'))

model_config = local_config['model_config']
# sampler_config = local_config["sampler_config"]
# data_config= local_config["data_config"]
data_nam = "Graph_Twitter"
# lr=0.0005
# weight_decay=1e-4
# patience=10
# n_epochs=200
# batchsize=128
# TDdroprate=0
# BUdroprate=0
cuda_id = 0  # or -1 if cpu
device = torch.device(f'cuda:{cuda_id}' if cuda_id >= 0 else 'cpu')
print("[INFO] Dataset:", data_nam)
print("[INFO] Model:", model_config["model_name"])
print("[INFO] Using device:", device)
print("[INFO] lr:", model_config["lr"])
print("[INFO] batch-size:", model_config["batch_size"])
print("[INFO] Gnum:", model_config["Gnum_m"])
print("[INFO] Knum: ", model_config["Nnum_k"])
print("[INFO] hidden-size:  ", model_config["hidden_size"])
print("[INFO] pred-coef:  ", model_config["pred_coef"])
print("[INFO] sampler-coef:", model_config["sampler_coef"])
print("[INFO] UDdroprate:  ", model_config["UDdroprate"])
print("[INFO] weight_decay:  ", model_config["weight_decay"])


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.backends.cudnn.deterministic = True
    print("seed:", seed)


datasetname=sys.argv[1] #"Twitter15"、"Twitter16"
iterations=int(sys.argv[2])
model="GCN"
# device = th.device('cuda:1' if th.cuda.is_available() else 'cpu')


def run_main(contrast_lr):
    # set_seed(1)
    f = open('/home/ubuntu/PyProjects_gsuhyl/PyProjects/Contrast_Subsets_fusion/'+datasetname+'res_R5.txt','a')
    test_accs = []
    NR_F1 = []
    FR_F1 = []
    TR_F1 = []
    UR_F1 = []
    for iter in range(iterations):
        fold0_x_test, fold0_x_train, \
        fold1_x_test,  fold1_x_train,  \
        fold2_x_test, fold2_x_train, \
        fold3_x_test, fold3_x_train, \
        fold4_x_test,fold4_x_train = load5foldData(datasetname)
        treeDic=loadTree(datasetname)
        # print()
        train_losses, val_losses, train_accs, val_accs0, accs0, F1_0, F2_0, F3_0, F4_0 = train_GCN(treeDic,
                                                                                                   fold0_x_test,
                                                                                                   fold0_x_train,
                                                                                                   model_config,
                                                                                                   model_config['UDdroprate'],
                                                                                                   model_config['lr'],
                                                                                                   model_config["weight_decay"],
                                                                                                   model_config['patience'],
                                                                                                   model_config['epochs'],
                                                                                                   model_config['batch_size'],
                                                                                                   datasetname,
                                                                                                   iter, contrast_lr)
        train_losses, val_losses, train_accs, val_accs1, accs1, F1_1, F2_1, F3_1, F4_1 = train_GCN(treeDic,
                                                                                                   fold1_x_test,
                                                                                                   fold1_x_train,
                                                                                                   model_config,
                                                                                                   model_config['UDdroprate'],
                                                                                                   model_config['lr'], model_config["weight_decay"],
                                                                                                   model_config['patience'],
                                                                                                   model_config['epochs'],
                                                                                                   model_config['batch_size'],
                                                                                                   datasetname,
                                                                                                   iter, contrast_lr)
        train_losses, val_losses, train_accs, val_accs2, accs2, F1_2, F2_2, F3_2, F4_2 = train_GCN(treeDic,
                                                                                                   fold2_x_test,
                                                                                                   fold2_x_train,
                                                                                                   model_config,
                                                                                                   model_config['UDdroprate'],
                                                                                                   model_config['lr'], model_config["weight_decay"],
                                                                                                   model_config['patience'],
                                                                                                   model_config['epochs'],
                                                                                                   model_config['batch_size'],
                                                                                                   datasetname,
                                                                                                   iter, contrast_lr)
        train_losses, val_losses, train_accs, val_accs3, accs3, F1_3, F2_3, F3_3, F4_3 = train_GCN(treeDic,
                                                                                                   fold3_x_test,
                                                                                                   fold3_x_train,
                                                                                                   model_config,
                                                                                                   model_config['UDdroprate'],
                                                                                                   model_config['lr'], model_config["weight_decay"],
                                                                                                   model_config['patience'],
                                                                                                   model_config['epochs'],
                                                                                                   model_config['batch_size'],
                                                                                                   datasetname,
                                                                                                   iter, contrast_lr)
        train_losses, val_losses, train_accs, val_accs4, accs4, F1_4, F2_4, F3_4, F4_4 = train_GCN(treeDic,
                                                                                                   fold4_x_test,
                                                                                                   fold4_x_train,
                                                                                                   model_config,
                                                                                                   model_config['UDdroprate'],
                                                                                                   model_config['lr'], model_config["weight_decay"],
                                                                                                   model_config['patience'],
                                                                                                   model_config['epochs'],
                                                                                                   model_config['batch_size'],
                                                                                                   datasetname,
                                                                                                   iter, contrast_lr)
        test_accs.append((accs0+accs1+accs2+accs3+accs4)/5)
        NR_F1.append((F1_0+F1_1+F1_2+F1_3+F1_4)/5)
        FR_F1.append((F2_0 + F2_1 + F2_2 + F2_3 + F2_4) / 5)
        TR_F1.append((F3_0 + F3_1 + F3_2 + F3_3 + F3_4) / 5)
        UR_F1.append((F4_0 + F4_1 + F4_2 + F4_3 + F4_4) / 5)
    # print("Total_Test_Accuracy: {:.4f}|NR F1: {:.4f}|FR F1: {:.4f}|TR F1: {:.4f}|UR F1: {:.4f}".format(
    #     sum(test_accs) / iterations, sum(NR_F1) /iterations, sum(FR_F1) /iterations, sum(TR_F1) / iterations, sum(UR_F1) / iterations))
    res = "Total_Test_Accuracy: {:.4f}|NR F1: {:.4f}|FR F1: {:.4f}|TR F1: {:.4f}|UR F1: {:.4f}".format(
        sum(test_accs) / iterations, sum(NR_F1) /iterations, sum(FR_F1) /iterations, sum(TR_F1) / iterations, sum(UR_F1) / iterations)
    print(res)
    f.write(str(contrast_lr)+res+"\n")
    f.close()
for i in np.arange(0.5, 1.1, 0.1):
    # print(i)
    # i = 0.3
    run_main(i)
