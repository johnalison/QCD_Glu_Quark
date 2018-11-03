import numpy as np
np.random.seed(0)
import os
import time
import h5py
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import *
from sklearn.metrics import roc_curve, auc

import argparse
parser = argparse.ArgumentParser(description='Training parameters')
parser.add_argument('-e', '--epochs', default=30, type=int, help='Number of training epochs.')
parser.add_argument('-l', '--lr_init', default=5.e-4, type=float, help='Initial learning rate.')
parser.add_argument('-b', '--resblocks', default=3, type=int, help='Number of residual blocks.')
parser.add_argument('-c', '--cuda', default=0, type=int, help='Which gpuid to use.')
args = parser.parse_args()

lr_init = args.lr_init
resblocks = args.resblocks
epochs = args.epochs
os.environ["CUDA_VISIBLE_DEVICES"]=str(args.cuda)

expt_name = 'ResNet_blocks%d_RH1o100_ECAL+HCAL+Trk_lr%s_gamma0.5every10ep_epochs%d'%(resblocks, str(lr_init), epochs)

class ParquetDataset(Dataset):
    def __init__(self, filename):
        self.parquet = pq.ParquetFile(filename)
        self.cols = None
        #self.cols = ['X_jets.list.item.list.item.list.item','y'] 
    def __getitem__(self, index):
        data = self.parquet.read_row_group(index, columns=self.cols).to_pydict()
        data['X_jets'] = np.float32(data['X_jets'][0]) 
        data['y'] = np.float32(data['y'])
        data['m0'] = np.float32(data['m0'])
        data['pt'] = np.float32(data['pt'])
        # Preprocessing
        data['X_jets'][data['X_jets'] < 1.e-3] = 0. # Zero-Suppression
        data['X_jets'][-1,...] = 25.*data['X_jets'][-1,...] # For HCAL: to match pixel intensity distn of other layers
        data['X_jets'] = data['X_jets']/100. # To standardize
        return dict(data)
    def __len__(self):
        return self.parquet.num_row_groups

decay = 'QCDToGGQQ_IMGjet_RH1all'
decays = ['%s_run0_n215556'%decay, '%s_run1_n297980'%decay, '%s_run2_n280364'%decay]
expt_name = '%s_%s'%(decay, expt_name)
for d in ['MODELS', 'METRICS']:
    if not os.path.isdir('%s/%s'%(d, expt_name)): 
        os.mkdir('%s/%s'%(d, expt_name))

ntrain_total = sum([int(d.split('_')[-1][1:]) for d in decays]) #793900
idxs = np.random.permutation(ntrain_total)
train_cut = 2*384*1000 

dset_train = ConcatDataset([ParquetDataset("IMG/%s.train.snappy.parquet"%d) for d in decays])
train_sampler = sampler.SubsetRandomSampler(idxs[:train_cut])
train_loader = DataLoader(dataset=dset_train, batch_size=32, num_workers=10, sampler=train_sampler, pin_memory=True)

dset_val = ConcatDataset([ParquetDataset("IMG/%s.train.snappy.parquet"%d) for d in decays])
val_sampler = sampler.SubsetRandomSampler(idxs[train_cut:])
val_loader = DataLoader(dataset=dset_val, batch_size=120, num_workers=10, sampler=val_sampler)

import torch_resnet_single as networks
resnet = networks.ResNet(3, resblocks, [16, 32])
resnet.cuda()
optimizer = optim.Adam(resnet.parameters(), lr=lr_init)
lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20], gamma=0.5)

def do_eval(resnet, val_loader, f, roc_auc_best, epoch):
    global expt_name
    loss_, acc_ = 0., 0.
    y_pred_, y_truth_, m0_, pt_ = [], [], [], []
    now = time.time()
    for i, data in enumerate(val_loader):
        X, y, m0, pt = data['X_jets'].cuda(), data['y'].cuda(), data['m0'], data['pt']
        logits = resnet(X)
        loss_ += F.binary_cross_entropy_with_logits(logits, y).item()
        pred = logits.ge(0.).byte()
        acc_ += pred.eq(y.byte()).float().mean().item()
        y_pred = torch.sigmoid(logits)
        # Store batch metrics:
        y_pred_.append(y_pred.tolist())
        y_truth_.append(y.tolist())
        m0_.append(m0.tolist())
        pt_.append(pt.tolist())

    now = time.time() - now
    y_pred_ = np.concatenate(y_pred_)
    y_truth_ = np.concatenate(y_truth_)
    m0_ = np.concatenate(m0_)
    pt_ = np.concatenate(pt_)
    s = '%d: Val time:%.2fs in %d steps'%(epoch+1, now, len(val_loader))
    print(s)
    f.write('%s\n'%(s))
    s = '%d: Val loss:%f, acc:%f'%(epoch+1, loss_/len(val_loader), acc_/len(val_loader))
    print(s)
    f.write('%s\n'%(s))

    fpr, tpr, _ = roc_curve(y_truth_, y_pred_)
    roc_auc = auc(fpr, tpr)
    s = "VAL ROC AUC: %f"%(roc_auc)
    print(s)
    f.write('%s\n'%(s))

    #if roc_auc > roc_auc_best:
    #    roc_auc_best = roc_auc
    #    f.write('Best ROC AUC:%.4f\n'%roc_auc_best)
    #    score_str = 'epoch%d_auc%.4f'%(epoch+1, roc_auc_best)

    #    filename = 'MODELS/%s/model_%s.pkl'%(expt_name, score_str)
    #    model_dict = {'model': resnet.state_dict(), 'optim': optimizer.state_dict()}
    #    torch.save(model_dict, filename)

    #    h = h5py.File('METRICS/%s/metrics_%s.hdf5'%(expt_name, score_str), 'w')
    #    h.create_dataset('fpr', data=fpr)
    #    h.create_dataset('tpr', data=tpr)
    #    h.create_dataset('y_truth', data=y_truth_)
    #    h.create_dataset('y_pred', data=y_pred_)
    #    h.create_dataset('m0', data=m0_)
    #    h.close()

    return roc_auc_best

# MAIN #
#eval_step = 1000
print_step = 1000
roc_auc_best = 0.5
print(">> Training <<<<<<<<")
f = open('%s.log'%(expt_name), 'w')
for epoch in range(epochs):

    s = '>> Epoch %d <<<<<<<<'%(epoch+1)
    print(s)
    f.write('%s\n'%(s))

    # Run training
    lr_scheduler.step()
    resnet.train()
    now = time.time()
    for i, data in enumerate(train_loader):
        X, y = data['X_jets'].cuda(), data['y'].cuda()
        optimizer.zero_grad()
        logits = resnet(X)
        loss = F.binary_cross_entropy_with_logits(logits, y).cuda()
        loss.backward()
        optimizer.step()
        if i % print_step == 0:
            pred = logits.ge(0.).byte()
            acc = pred.eq(y.byte()).float().mean()
            s = '%d: Train loss:%f, acc:%f'%(epoch+1, loss.item(), acc.item())
            print(s)
        # For more frequent validation:
        #if epoch+1 > 1 and i % eval_step == 0:
        #    resnet.eval()
        #    roc_auc_best = do_eval(resnet, val_loader, f, roc_auc_best, epoch)
        #    resnet.train()

    f.write('%s\n'%(s))
    now = time.time() - now
    s = '%d: Train time:%.2fs in %d steps'%(epoch+1, now, len(train_loader))
    print(s)
    f.write('%s\n'%(s))

    # Run Validation
    resnet.eval()
    roc_auc_best = do_eval(resnet, val_loader, f, roc_auc_best, epoch)

f.close()
