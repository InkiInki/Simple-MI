"""
@author: Inki
@contact: inki.yinji@gmail.com
@version: Created in 2020 1119, last modified in 2020 1119.
@note: The refer paper is Multiple instance classification: Review, taxonomy and comparative study.
"""
import numpy as np
from Prototype import MIL


class SimpleMi(MIL):
    """
    The Simple-MI algorithms.
    @param:
        Please refer the algorithm named MIL.
    @attribute:
        vector:
            The mean vector of bag.
    @example:
        See main.
    """

    def __init__(self, path, k=10):
        """
        The constructor.
        """
        super(SimpleMi, self).__init__(path)
        self.k = k
        self.vector = []
        self.tr_idx = []
        self.te_idx = []
        self.__initialize_simple_mi()

    def __initialize_simple_mi(self):
        """
        The initialize of Simple-MI.
        """
        self.vector = np.zeros((self.num_bags, self.dimensions))
        for i in range(self.num_bags):
            temp_bag = self.bags[i, 0][:, :self.dimensions]
            self.vector[i] = np.average(temp_bag, 0)

    def get_mapping(self):
        """
        Split training set and test set.
        """
        self.tr_idx, self.te_idx = self.get_index(self.k)
        for loop in range(self.k):
            yield self.vector[self.tr_idx[loop]], self.bags_label[self.tr_idx[loop]], \
                  self.vector[self.te_idx[loop]], self.bags_label[self.te_idx[loop]], None
