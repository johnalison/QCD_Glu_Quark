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

def convert_to_Parquet(decays, start, stop, chunk_size, expt_name):
    
    # Open the input HDF5 file
    dsets = [h5py.File('%s'%decay) for decay in decays]
    keys = ['X_jets', 'jet_pT', 'jet_truthLabel'] # key names in in put hdf5
    row0 = [np2arrowArray(dsets[0][key][0]) for key in keys]
    keys = ['X_jets', 'pt', 'y'] # desired key names in output parquet
    table0 = pa.Table.from_arrays(row0, keys) 
    
    # Open the output Parquet file
    filename = '%s.snappy.parquet'%(expt_name)
    writer = pq.ParquetWriter(filename, table0.schema, compression='snappy')

    # Loop over file chunks of size chunk_size
    nevts = stop - start
    #for i in range(nevts//chunk_size):
    for i in range(int(np.ceil(1.*nevts/chunk_size))):
        
        begin = start + i*chunk_size
        end = begin + chunk_size

        # Load array chunks into memory
        X = np.concatenate([dset['X_jets'][begin:end] for dset in dsets])
        pt = np.concatenate([dset['jet_pT'][begin:end] for dset in dsets])
        y = np.concatenate([dset['jet_truthLabel'][begin:end] for dset in dsets])
        
        # Shuffle
        l = list(zip(X, pt, y))
        random.shuffle(l)
        X, pt, y = zip(*l)

        # Convert events in the chunk one-by-one
        print('Doing events: [%d->%d)'%(begin,end))
        for j in range(len(y)):

            # Create a list for each sample
            sample = [
                np2arrowArray(X[j]),
                np2arrowArray(pt[j]),
                np2arrowArray(y[j]),
            ]

            table = pa.Table.from_arrays(sample, keys)

            writer.write_table(table)

    writer.close()
    return filename
    
# MAIN
chunk_size = 3200
#jetId = "BJetsAll"
jetId = "NonBJetsAll"

#test_jets_BJetsAll.hdf5
#test_jets_NonBJetsAll.hdf5

inputFileNames = glob.glob('test_jets_%s.hdf5'%(jetId))
print(inputFileNames)
inputFile =  h5py.File(inputFileNames[0],"r")
X_jets = inputFile.get("X_jets")
nevts_total = int(X_jets.shape[0])
inputFile.close()
print(' >> Total events per file:', nevts_total)
start, stop = 0, nevts_total

expt_name = '%s_IMGjet_n%d'%(jetId, len(inputFileNames)*(stop-start))

now = time.time()
f = convert_to_Parquet(inputFileNames, start, stop, chunk_size, expt_name)
print(' >> time: %.2f'%(time.time()-now))

reader = pq.ParquetFile(f)
for i in range(10):
    print(i, reader.read_row_group(i).to_pydict()['y'])
print(' >> Total events written:',reader.num_row_groups)
