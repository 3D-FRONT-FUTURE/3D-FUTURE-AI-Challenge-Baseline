from torch.utils.data.sampler import Sampler
from ..utils import common_functions as c_f
import numpy as np

# modified from
# https://raw.githubusercontent.com/bnulihaixia/Deep_metric/master/utils/sampler.py
class MPerClassSampler(Sampler):
    """
    At every iteration, this will return m samples per class. For example,
    if dataloader's batchsize is 100, and m = 5, then 20 classes with 5 samples
    each will be returned
    Args:
        labels_to_indices: a dictionary mapping dataset labels to lists of
                            indices that have that label
        m: the number of samples per class to fetch at every iteration. If a
                    class has less than m samples, then there will be duplicates
                    in the returned batch
        hierarchy_level: which level of labels will be used to form each batch.
                        The default is 0, because most use-cases will have
                        1 label per datapoint. But for example, iNat has 7
                        labels per datapoint, in which case hierarchy_level could
                        be set to a number between 0 and 6.
    """

    def __init__(self, labels_to_indices, m, hierarchy_level=0):
        self.m_per_class = int(m)
        self.labels_to_indices = labels_to_indices
        self.set_hierarchy_level(hierarchy_level)
        self.length_of_single_pass = self.m_per_class*len(self.labels)
        self.list_size = 1000000
        if self.length_of_single_pass < self.list_size:
            self.list_size -= (self.list_size) % (self.length_of_single_pass)

    def __len__(self):
        return self.list_size

    def __iter__(self):
        idx_list = [0]*self.list_size
        i = 0
        num_iters = self.list_size // self.length_of_single_pass if self.length_of_single_pass < self.list_size else 1
        for _ in range(num_iters):
            c_f.NUMPY_RANDOM_STATE.shuffle(self.labels)
            for label in self.labels:
                t = self.curr_labels_to_indices[label]
                idx_list[i:i+self.m_per_class] = c_f.safe_random_choice(t, size=self.m_per_class)
                i += self.m_per_class
        return iter(idx_list)

    def set_hierarchy_level(self, hierarchy_level):
        self.curr_labels_to_indices = self.labels_to_indices[hierarchy_level]
        self.labels = list(self.curr_labels_to_indices.keys())