import os

class DataLoader:
    def __init__(self, data_path, label_file='training_label.txt', no_label_file='training_nolabel.txt', dictionary_file='dictionary.txt'):
        """
        初始化数据加载器
        
        Args:
            data_path (str): 数据路径
            label_file (str): 包含标签数据的文件名
            no_label_file (str): 不包含标签数据的文件名
            dictionary_file (str): 包含基因数据字典的文件名
        """
        self.data_path = data_path
        self.label_file = label_file
        self.no_label_file = no_label_file
        self.dictionary_file = dictionary_file
        
    def load_training_data(self, with_labels=True):
        """
        加载训练数据
        
        Args:
            with_labels (bool): 是否加载标签数据。默认为True，表示加载标签数据。
        
        Returns:
            y: 基因名称
            z: 标签数据
        """
        file_name = self.label_file if with_labels else self.no_label_file
        file_path = os.path.join(self.data_path, file_name)
        
        # 读取文件
        with open(file_path, 'r+', encoding='utf-8') as f:
            lines = f.readlines()  # 读取所有行
            lines = [line.strip('\n').split(' ') for line in lines]  # 按空格分割
            
        y = [line[1] for line in lines]  # 第二列是基因名称
        y = [site2.split(',') for site2 in y]  # 基因名称可能包含多个，以逗号分割
        
        z = [line[0] for line in lines]  # 第0列是标签
        z = [[int(part) for part in site3.split('_')] for site3 in z]  # 标签拆分成整数
        
        return y, z

    def load_dictionary_data(self):
        """
        加载基因数据字典
        
        Returns:
            lines: 基因数据，按空格分割的每一行
        """
        file_path = os.path.join(self.data_path, self.dictionary_file)
        
        with open(file_path, 'r+', encoding='utf-8') as f:
            lines = f.readlines()  # 读取所有行
            lines = [line.lstrip().strip('\n').split(' ') for line in lines]  # 按空格分割并去除空白字符
            
        return lines

