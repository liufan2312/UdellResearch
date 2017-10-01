class node(object):

    def __init__(self, row_index_list, depth):
        '''
        :param data:  data is a numpy array with (sample_size x number of columns, [data point features])
        the last column of the data is the label(or response)
        '''
        self.row_index_list = row_index_list
        self.left_kid = None
        self.right_kid = None
        self.depth = depth
        self.split_value = None
        self.split_feature_index = None

    def set_split_information(self, split_feature_index, split_value):
        self.split_feature_index = split_feature_index
        self.split_value = split_value

    def get_row_index_list(self):
        return node.row_index_list





