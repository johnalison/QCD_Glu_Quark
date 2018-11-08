import numpy as np
np.random.seed(0)
import os, glob, re
import time
import h5py
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from torch.utils.data import *
from sklearn.metrics import roc_curve, auc

import argparse
parser = argparse.ArgumentParser(description='Evaluation parameters.')
parser.add_argument('-e', '--epoch', required=True, type=int, help='Training epoch to use for evaluation.')
parser.add_argument('-s', '--expt_name', required=True, type=str, help='String corresponding to expt_name.')
parser.add_argument('-c', '--cuda', default=0, type=int, help='Which gpuid to use.')
args = parser.parse_args()

epoch = args.epoch
expt_name = args.expt_name
nblocks = int(re.search('blocks([0-9]+?)_', expt_name).group(1))
os.environ["CUDA_VISIBLE_DEVICES"]=str(args.cuda)

# NOTE: The ParquetDataset class here has to exactly match the one used during training time!!! 
class ParquetDataset(Dataset):
    def __init__(self, filename):
        self.parquet = pq.ParquetFile(filename)
        self.cols = None # read all columns
        #self.cols = ['X_ECAL_stacked.list.item.list.item.list.item','y']
    def __getitem__(self, index):
        data = self.parquet.read_row_group(index, columns=self.cols).to_pydict()
        data['X_ECAL_stacked'] = np.float32(data['X_ECAL_stacked'][0])
        data['y'] = np.float32(data['y'])
        # Preprocessing
        data['X_ECAL_stacked'][data['X_ECAL_stacked'] < 1.e-3] = 0. # Zero-Suppression
        data['X_ECAL_stacked'][-1,...] = 25.*data['X_ECAL_stacked'][-1,...] # For HCAL: to match pixel intensity distn of other layers
        data['X_ECAL_stacked'] = data['X_ECAL_stacked']/100. # To standardize
        return dict(data)
    def __len__(self):
        return self.parquet.num_row_groups

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

model_file = glob.glob('MODELS/%s/model_epoch%d_auc*.pkl'%(expt_name, epoch))
assert len(model_file) == 1
model_file = model_file[0]
print(">> Model file:", model_file)

for d in ['METRICS_TEST']:
    if not os.path.isdir('%s/%s'%(d, expt_name)):
        os.makedirs('%s/%s'%(d, expt_name))

decay = 'QCDToGGQQ_IMG_RH1all'
decays = glob.glob('IMG/%s_run?_n*.test.snappy.parquet'%decay)
print(">> Input files:",decays)
assert len(decays) == 3, "len(decays) = %d"%(len(decays))

dset_test = ConcatDataset([ParquetDataset(d) for d in decays])
test_loader = DataLoader(dataset=dset_test, batch_size=120, num_workers=10)

import torch_resnet_single as networks
resnet = networks.ResNet(3, nblocks, [16, 32])
resnet.cuda()
resnet.load_state_dict(torch.load('%s'%model_file)['model'])
print('>> N model params (trainable):', count_parameters(resnet))

def do_eval_test(resnet, test_loader, epoch):
    print_step = 100
    loss_, acc_ = 0., 0.
    y_pred_, y_truth_, m0_, pt_ = [], [], [], []
    now = time.time()
    for i, data in enumerate(test_loader):
        if i % print_step == 0:
            print('(%d/%d)'%(i, len(test_loader)))
        X, y = data['X_ECAL_stacked'].cuda(), data['y'].cuda()
        logits = resnet(X)
        loss_ += F.binary_cross_entropy_with_logits(logits, y).item()
        pred = logits.ge(0.).byte()
        acc_ += pred.eq(y.byte()).float().mean().item()
        y_pred = F.sigmoid(logits)
        # Store batch metrics:
        y_pred_.append(y_pred.tolist())
        y_truth_.append(y.tolist())

    now = time.time() - now
    y_pred_ = np.concatenate(y_pred_)
    y_truth_ = np.concatenate(y_truth_)
    s = '%d: Test time:%.2fs in %d steps'%(epoch, now, len(test_loader))
    print(s)
    s = '%d: Test loss:%f, acc:%f'%(epoch, loss_/len(test_loader), acc_/len(test_loader))
    print(s)

    fpr, tpr, _ = roc_curve(y_truth_, y_pred_)
    roc_auc = auc(fpr, tpr)
    s = "TEST ROC AUC: %f"%(roc_auc)
    print(s)

    score_str = 'epoch%d_auc%.4f'%(epoch, roc_auc)
    h = h5py.File('METRICS_TEST/%s/metrics_%s.hdf5'%(expt_name, score_str), 'w')
    h.create_dataset('fpr', data=fpr)
    h.create_dataset('tpr', data=tpr)
    h.create_dataset('y_truth', data=y_truth_)
    h.create_dataset('y_pred', data=y_pred_)
    h.close()

resnet.eval()
do_eval_test(resnet, test_loader, epoch)
