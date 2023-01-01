import torch
import numpy as np
import os
import soundfile
import glob
import json
import wave
import yaml
from speech_features import speech_features

pth = r'config.json'

def load_config_file(pth):
    with open(pth, "r", encoding="utf8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


DEFAULT_CONFIG_FILENAME = r'config_win.yml'

# pinyin_dict = r'dict.txt'
# train_data_dict = r'/Users/liufucong/Downloads/ltxm/ContextNet-master/datalist/st-cmds/train.wav.txt'
# dev_data_dict = r'/Users/liufucong/Downloads/ltxm/ContextNet-master/datalist/st-cmds/dev.wav.txt'
# test_data_dict = r'/Users/liufucong/Downloads/ltxm/ContextNet-master/datalist/st-cmds/test.wav.txt'
# data_pth = r'/Users/liufucong/Downloads/ltxm/ST-CMDS-20170001_1-OS'
# label = r'/Users/liufucong/Downloads/ltxm/Speech/st-cmds/train.syllable.txt'


def read_wav_data(filename: str) -> tuple:
    """
    读取一个wav文件，返回声音信号的时域谱矩阵和播放时间
    """
    wav = wave.open(filename,"rb") # 打开一个wav格式的声音文件流
    num_frame = wav.getnframes() # 获取帧数
    num_channel=wav.getnchannels() # 获取声道数
    framerate=wav.getframerate() # 获取帧速率
    num_sample_width=wav.getsampwidth() # 获取实例的比特宽度，即每一帧的字节数
    str_data = wav.readframes(num_frame) # 读取全部的帧
    wav.close() # 关闭流
    wave_data = np.fromstring(str_data, dtype = np.short) # 将声音文件数据转换为数组矩阵形式
    wave_data.shape = -1, num_channel # 按照声道数将数组整形，单声道时候是一列数组，双声道时候是两列的矩阵
    wave_data = wave_data.T # 将矩阵转置
    return wave_data, framerate, num_channel, num_sample_width



def load_pinyin_dict(filename: str) -> tuple:
    """
    加载拼音列表和拼音字典

    拼音列表：用于下标索引转拼音 \\
    拼音字典：用于拼音索引转下标
    """
    # global _pinyin_list, _pinyin_dict
    # if _pinyin_dict is not None and _pinyin_list is not None:
    #     return _pinyin_list, _pinyin_dict

    _pinyin_list = list()
    _pinyin_dict = dict()
    with open(filename, 'r', encoding='utf-8') as file_pointer:
        lines = file_pointer.read().split('\n')
        # print(len(lines))
    for line in lines:
        # print(line)
        if len(line) == 0:
            continue
        tokens = line.split('\t')
        _pinyin_list.append(tokens[0])
        _pinyin_dict[tokens[0]] = len(_pinyin_list) - 1#对应的位置
    return _pinyin_list, _pinyin_dict




class DataLoader:
    def __init__(self, dataset_type = 'train'):
        self.dataset_type = dataset_type
        # self.PINYIN = pinyin_dict

        self.data_list = list()
        self.wav_dict = dict()
        self.label_dict = dict()
        self.pinyin_list = list()#拼音索引
        self.pinyin_dict = dict()#汉字段长度
        self._load_data()
        self.data_count = self.get_data_count()

    def _load_data(self):
        config = load_config_file(DEFAULT_CONFIG_FILENAME)

        self.pinyin_list, self.pinyin_dict = load_pinyin_dict(config['dic_filename'])
        print('self.pinyin_dict-----{}'.format(len(self.pinyin_dict)))

        # for index in range(len(config['dataset'][self.dataset_type])):
        # for index in DATA_SET_NAME:
        # idx = index+'_'+self.dataset_type
        # print(type(idx),idx)
        # filename_datalist = config['dataset'][self.dataset_type][idx]['data_list']
        # filename_datapath = config['dataset'][self.dataset_type][idx]['data_pth']
        filename_datalist = config['st-cmds']['train']['data_list']
        filename_datapath = config['st-cmds']['train']['data_pth']
        with open(filename_datalist, 'r', encoding='utf-8') as file_pointer:
            lines = file_pointer.read().split('\n')
            for line in lines:
                if len(line) == 0:
                    continue
                tokens = line.split(' ')
                self.data_list.append(tokens[0])
                self.wav_dict[tokens[0]] = os.path.join(filename_datapath, tokens[1].split('/')[-1])

        filename_labellist = config['st-cmds']['train']['label_list']
        with open(filename_labellist, 'r', encoding='utf-8') as file_pointer:
            lines = file_pointer.read().split('\n')
            for line in lines:
                if len(line) == 0:
                    continue
                tokens = line.split(' ')
                self.label_dict[tokens[0]] = tokens[1:]
        # print('self.data_list---------{}'.format(self.data_list[:2]))
        # print('self.label_dict---------{}'.format(self.label_dict['20170001P00001A0001']))
        # print('self.wav_dict---------{}'.format(self.wav_dict['20170001P00001A0001']))


    def get_data_count(self) -> int:
        """
        获取数据集总数量
        """
        return len(self.data_list)


    def get_data(self, index: int) -> tuple:
        """
        按下标获取一条数据
        """
        mark = self.data_list[index]

        wav_signal, sample_rate, _, _ = read_wav_data(self.wav_dict[mark])
        labels = list()
        # print('label_dict[mark]',self.label_dict[mark])
        for item in self.label_dict[mark]:
            if len(item) == 0:
                continue
            labels.append(self.pinyin_dict[item])

        data_label = np.array(labels)
        return wav_signal, sample_rate, data_label




# class MyDataset(torch.utils.data.Dataset):  # 创建自己的类：MyDataset,这个类是继承的torch.utils.data.Dataset
#     def __init__(self, root, pinyin_pth, datatxt, transform=None, target_transform=None):  # 初始化一些需要传入的参数
#         _, self.label = load_pinyin_dict(pinyin_pth)
#
#         fh = open(root + datatxt, 'r')  # 按照传入的路径和txt文本参数，打开这个文本，并读取内容
#         imgs = []  # 创建一个名为img的空列表，一会儿用来装东西
#         for line in fh:  # 按行循环txt文本中的内容
#             line = line.rstrip()  # 删除 本行string 字符串末尾的指定字符，这个方法的详细介绍自己查询python
#             words = line.split()  # 通过指定分隔符对字符串进行切片，默认为所有的空字符，包括空格、换行、制表符等
#             imgs.append((words[0], int(words[1])))  # 把txt里的内容读入imgs列表保存，具体是words几要看txt内容而定
#             # 很显然，根据我刚才截图所示txt的内容，words[0]是图片信息，words[1]是lable
#         self.imgs = imgs
#         self.transform = transform
#         self.target_transform = target_transform
#
#     def __getitem__(self, index):  # 这个方法是必须要有的，用于按照索引读取每个元素的具体内容
#         fn, label = self.imgs[index]  # fn是图片path #fn和label分别获得imgs[index]也即是刚才每行中word[0]和word[1]的信息
#         img = Image.open(root + fn).convert('RGB')  # 按照path读入图片from PIL import Image # 按照路径读取图片
#
#         if self.transform is not None:
#             img = self.transform(img)  # 是否进行transform
#         return img, label  # return很关键，return回哪些内容，那么我们在训练时循环读取每个batch时，就能获得哪些内容
#
#     def __len__(self):  # 这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
#         return len(self.imgs)


# 根据自己定义的那个勒MyDataset来创建数据集！注意是数据集！而不是loader迭代器



# train_data = MyDataset(txt=root + 'train.txt', transform=transforms.ToTensor())
# test_data = MyDataset(txt=root + 'test.txt', transform=transforms.ToTensor())



# if __name__ == '__main__':

    # a,b = load_pinyin_dict(pinyin_dict)

    # cl = DataLoader()
    # # cl._load_data()
    #
    # wav_signal, sample_rate, data_label = cl.get_data(5)
    # # print('wav_signal-----{}'.format(len(wav_signal[0])))
    # # print('sample_rate-----{}'.format(sample_rate))
    # # print('data_label-----{}'.format(len(data_label)))
    #
    #
    # ll = speech_features.MFCC()
    # data_input = ll.run(wav_signal)
    #
    # print('data_input-----{}'.format(data_input.shape))
    # print('data_label_shape------{}'.format(data_label))
    # # config = load_config_file(pth)
    # # print(config['dic_filename'])
    #
    # print('done')

