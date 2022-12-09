import h5py
import logging

logging.getLogger('datalad').setLevel(logging.WARNING)

def write_h5(h5_file, dataset, data, overwrite):
    with h5py.File(h5_file, 'a') as f:
                    try:
                        ds = f.require_dataset(dataset, shape=data.shape, dtype=data.dtype, data=data, chunks=True,
                                               maxshape=(None,)*data.ndim)
                    except TypeError:
                        if overwrite:
                            ds = f[dataset]
                            ds.resize(data.shape)
                            ds.write_direct(data)
                            