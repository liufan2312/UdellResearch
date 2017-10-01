from node import node
import numpy as np

class decisionTree(object):

    def __init__(self, data, label, type_list, predict_min_err, min_data_size, max_split, dis_continuous_flag):
        '''
        :param data: table with row and features
        :param label: response
        :param type_list: type of feature in teh data
        :param predict_min_err: function to predict the minimal err
        :param min_data_size: minimal size for each node
        '''
        n_r, n_c = data.shape
        self.data = data
        self.type_list = type_list
        self.label = label
        self.root = node(range(1,n_r), 0)
        self.predict_min_err = predict_min_err
        self.min_data_size = min_data_size
        self.max_split = max_split
        self.leaves = [self.root]
        self.nodes = [self.root]
        self.dis_continous_flag = dis_continuous_flag

    def get_nodes(self):
        return self.nodes

    def check_at_least_two_classes(self, node):

        '''
        :param node: node after split
        :return: whether this node contains at least two classes
        '''

        label = (self.data[node.get_row_index_list()])
        return len(np.unique(label)) >= 2


    def split(self, node, feature_index, split_value):
        '''
        :param node:  node to be splitted
        :param col_index:
        :param split_value:
        :return:
        '''
        data = self.data[node.row_index_list(),:]
        left_index_list = np.where(data[:, feature_index] <= split_value)
        right_index_list = np.where(data[:, feature_index] > split_value)
        node.left_kid = node(left_index_list, node.depth+1)
        node.right = node(right_index_list, node.depth+1)

        return node.left_kid, node.right_kid


    def build_tree_uniform_split(self):

        '''
        :return: list of nodes after max_depth random split
        '''

        _, n_f = self.data.shape

        for i in range(self.max_split):
            # choose leaf node
            split = True
            while split:
                leaf_ind = np.random.randint(0, len(self.leaves))
                cur_leaf = self.leaves[leaf_ind]
                feature_index = np.random.randint(0, n_f)
                values = self.data[cur_leaf.get_row_index_list(), feature_index]

                # just be more tolerant in the random split case
                # if the size of the leaf is too small restart the process
                if 3 * self.minimal_data_size >= len(values) - 3 * self.minimal_data_size:
                    continue
                values.sort()
                v_ind = np.random.randint(3 * self.minimal_data_size, len(values) - 3 * self.minimal_data_size)
                split_value = values[v_ind]
                left_kid, right_kid = self.split(cur_leaf, feature_index, split_value)
                if not self.check_at_least_two_classes(left_kid) or not self.check_at_least_two_classes(right_kid):
                    continue
                nd = self.leaves[leaf_ind]
                self.leaves.pop(leaf_ind)
                self.leaves.append(left_kid)
                self.leaves.append(right_kid)
                self.nodes.append(nd)
                split = False

        return self.get_nodes()

    def build_greedy_tree(self):









