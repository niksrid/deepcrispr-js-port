from pkgutil import get_data
import tensorflow as tf
import sonnet as snt
import numpy as np
import pandas as pd
from operator import add
from functools import reduce

np.set_printoptions(threshold=np.inf)


__all__ = ['Sgt', 'Episgt', 'Epiotrt']

ntmap = {'A': (1, 0, 0, 0),
         'C': (0, 1, 0, 0),
         'G': (0, 0, 1, 0),
         'T': (0, 0, 0, 1)
         }
epimap = {'A': 1, 'N': 0}


def get_seqcode(seq):
    return np.array(reduce(add, map(lambda c: ntmap[c], seq.upper()))).reshape(
        (1, len(seq), -1))

def get_epicode(eseq):
    return np.array(list(map(lambda c: epimap[c], eseq))).reshape(1, len(eseq), -1)



class Episgt:
    def __init__(self, fpath, num_epi_features, with_y=True):
        self._fpath = fpath
        self._ori_df = pd.read_csv(fpath, sep='\t', index_col=None, header=None)
        self._num_epi_features = num_epi_features
        self._with_y = with_y
        self._num_cols = num_epi_features + 2 if with_y else num_epi_features + 1
        self._cols = list(self._ori_df.columns)[-self._num_cols:]
        self._df = self._ori_df[self._cols]
        # print(self._df[self._cols])
        # print([self._cols])

    @property
    def length(self):
        return len(self._df)

    def get_dataset(self, x_dtype=np.float32, y_dtype=np.float32):
        x_seq = np.concatenate(list(map(get_seqcode, self._df[self._cols[0]])))
        x_epis = np.concatenate([np.concatenate(list(map(get_epicode, self._df[col]))) for col in
                                 self._cols[1: 1 + self._num_epi_features]], axis=-1)
        x = np.concatenate([x_seq, x_epis], axis=-1).astype(x_dtype)
       
        # print(x.shape)
        x = x.transpose(0, 2, 1)
        if self._with_y:
            y = np.array(self._df[self._cols[-1]]).astype(y_dtype)
            print(x,y)

            
            return x, y
        else:
            return x


file_path = 'examples/eg_cls_on_target.episgt'
input_data = Episgt(file_path, num_epi_features=4, with_y=True)
input_data.get_dataset()

