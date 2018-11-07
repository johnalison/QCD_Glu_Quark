import random
random.seed(1337)
import h5py
import time
import numpy as np
import glob, re

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
    dsets = [h5py.File('%s'%decay) for decay in decays]
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
chunk_size = 3200
jetId = 0

for set_name in list(['train', 'test']):

    print('>> Doing %s...'%set_name)

    if set_name == 'train':
        list_idx = '00000'
    else:
        list_idx = '00001'

    for runId in range(3):

        print(' >> Doing runId: %d'%runId)

        decays = glob.glob('QCD_Pt_80_170_%s_IMGjet_n*_label?_jet%d_run%d.hdf5'%(list_idx, jetId, runId))
        print(' >>',decays)
        assert len(decays) == 2
        nevts_total = decays[0].split("_")[-4][1:]
        nevts_total = int(nevts_total)
        print(' >> Total events per file:', nevts_total)

        start, stop = 0, nevts_total

        expt_name = 'QCDToGGQQ_IMGjet_RH1all_jet%d_run%d_n%d'%(jetId, runId, len(decays)*(stop-start))

        now = time.time()
        f = convert_to_Parquet(decays, start, stop, chunk_size, expt_name, set_name)
        print(' >> %s time: %.2f'%(set_name,time.time()-now))

        reader = pq.ParquetFile(f)
        for i in range(10):
            print(i, reader.read_row_group(i).to_pydict()['y'])
        print(' >> Total events written:',reader.num_row_groups)
