import random
random.seed(1337)
import h5py
import time
import numpy as np
import re

import pyarrow as pa
import pyarrow.parquet as pq

def np2arrowArray(x):
    if len(x.shape) > 1:
        x = np.transpose(x, [2,0,1])
        return pa.array([x.tolist()])
    else:
        return pa.array([x.tolist()])

def convert_to_Parquet(decays, start, stop, chunk_size, expt_name, set_name):
    
    # Open the input HDF5 file
    dsets = [h5py.File('%s.hdf5'%decay) for decay in decays]
    keys = ['X_jets', 'jetPt', 'jetM', 'y_jets'] # key names in in put hdf5
    row0 = [np2arrowArray(dsets[0][key][0]) for key in keys]
    keys = ['X_jets', 'pt', 'm0', 'y'] # desired key names in output parquet
    table0 = pa.Table.from_arrays(row0, keys) 
    
    # Open the output Parquet file
    filename = '%s.%s.snappy.parquet'%(expt_name,set_name)
    writer = pq.ParquetWriter(filename, table0.schema, compression='snappy')

    # Loop over file chunks of size chunk_size
    nevts = stop - start
    #for i in range(nevts//chunk_size):
    for i in range(int(np.ceil(1.*nevts/chunk_size))):
        
        begin = start + i*chunk_size
        end = begin + chunk_size

        # Load array chunks into memory
        X = np.concatenate([dset['X_jets'][begin:end] for dset in dsets])
        pt = np.concatenate([dset['jetPt'][begin:end] for dset in dsets])
        m = np.concatenate([dset['jetM'][begin:end] for dset in dsets])
        y = np.concatenate([dset['y_jets'][begin:end] for dset in dsets])
        
        # Shuffle
        l = list(zip(X, pt, m, y))
        random.shuffle(l)
        X, pt, m, y = zip(*l)

        # Convert events in the chunk one-by-one
        print('Doing events: [%d->%d)'%(begin,end))
        for j in range(len(y)):

            # Create a list for each sample
            sample = [
                np2arrowArray(X[j]),
                np2arrowArray(pt[j]),
                np2arrowArray(m[j]),
                np2arrowArray(y[j]),
            ]

            table = pa.Table.from_arrays(sample, keys)

            writer.write_table(table)

    writer.close()
    return filename
    
# MAIN
runId = 0
decays = [
    #'QCD_Pt_80_170_00000_IMGjet_n107778_label0_run0',
    #'QCD_Pt_80_170_00000_IMGjet_n107778_label1_run0'
    #'QCD_Pt_80_170_00000_IMGjet_n148990_label0_run1',
    #'QCD_Pt_80_170_00000_IMGjet_n148990_label1_run1'
    #'QCD_Pt_80_170_00000_IMGjet_n140182_label0_run2',
    #'QCD_Pt_80_170_00000_IMGjet_n140182_label1_run2'

    'QCD_Pt_80_170_00001_IMGjet_n18136_label0_run0',
    'QCD_Pt_80_170_00001_IMGjet_n18136_label1_run0'
    #'QCD_Pt_80_170_00001_IMGjet_n23770_label0_run1',
    #'QCD_Pt_80_170_00001_IMGjet_n23770_label1_run1'
    #'QCD_Pt_80_170_00001_IMGjet_n27747_label0_run2',
    #'QCD_Pt_80_170_00001_IMGjet_n27747_label1_run2'
    ]
assert runId == int(decays[0][-1])
train_total = decays[0].split("_")[-3][1:] 
train_total = int(train_total) 
print('total events per file:', train_total)

train_chunk_size = 3200 
train_start, train_stop = 0, train_total 
#assert (train_stop-train_start) % train_chunk_size == 0

#val_chunk_size = 3200
#val_start, val_stop = train_stop, 288000 
#assert (val_stop-val_start) % val_chunk_size == 0

expt_name_ = 'QCDToGGQQ_IMGjet_RH1all_run%d'%runId

#for set_name in list(['train', 'val']):
for set_name in list(['train']):

    print('Doing %s...'%set_name)

    if set_name == 'train':
        start = train_start
        stop = train_stop
        chunk_size = train_chunk_size
    else:
        start = val_start
        stop = val_stop
        chunk_size = val_chunk_size

    #expt_name = '%s_n%dk'%(expt_name_, 2*(stop-start)//1000.)
    expt_name = '%s_n%d'%(expt_name_, 2*(stop-start))

    now = time.time()
    f = convert_to_Parquet(decays, start, stop, chunk_size, expt_name, set_name)
    print('%s time: %.2f'%(set_name,time.time()-now))
    reader = pq.ParquetFile(f)
    for i in range(10):
        print(i, reader.read_row_group(i).to_pydict()['y'])
    print('total events written:',reader.num_row_groups)
