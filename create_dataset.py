import torch
import numpy as np
import random
import os
import yaml
import wave
from torch.utils.data import DataLoader
from speech_features import speech_features

import os.path

class Mydataset(torch.utils.data.Dataset):
    def __init__(self):
        super(Mydataset, self).__init__()
        self.data_list = list()
        self.wav_dict = dict()
        self.label_dict = dict()
        self.pinyin_list = list()  # 拼音索引
        self.pinyin_dict = dict()  # 汉字段长度


        self.DEFAULT_CONFIG_FILENAME = r'config_win.yml'



    def __getitem__(self, index):
        config = load_config_file(self.DEFAULT_CONFIG_FILENAME)

        self.pinyin_list, self.pinyin_dict = load_pinyin_dict(config['dic_filename'])


        filename_datalist = config['st-cmds']['train']['data_list']
        # print(filename_datalist)
        filename_datapath = config['st-cmds']['train']['data_pth']
        # print(filename_datapath)
        with open(filename_datalist, 'r', encoding='utf-8') as file_pointer:
            lines = file_pointer.read().split('\n')
            for line in lines:
                if len(line) == 0:
                    continue
                tokens = line.split(' ')
                self.data_list.append(tokens[0])
                self.wav_dict[tokens[0]] = os.path.join(filename_datapath, tokens[1].split('/')[-1])
            # print(self.wav_dict)
            # print('open filename is done')
        filename_labellist = config['st-cmds']['train']['label_list']
        with open(filename_labellist, 'r', encoding='utf-8') as file_pointer:
            lines = file_pointer.read().split('\n')
            for line in lines:
                if len(line) == 0:
                    continue
                tokens = line.split(' ')
                self.label_dict[tokens[0]] = tokens[1:]

        mark = self.data_list[index]
        # print('mark is here', mark)
        # print('self.wav_dict', self.wav_dict[mark])
        # print(os.path.isfile(self.wav_dict[mark]))
        wav_signal, sample_rate, _, _ = read_wav_data(self.wav_dict[mark])
        labels = list()
        # print('label_dict[mark]',self.label_dict[mark])
        for item in self.label_dict[mark]:
            if len(item) == 0:
                continue
            labels.append(self.pinyin_dict[item])

        data_label = np.array(labels)
        print('__data_label is done')
        mfcc_feature = speech_features.MFCC()
        data_input = mfcc_feature.run(wav_signal, sample_rate)
        print('__get_items is done')
        return data_input, data_label

    def __len__(self):
        return len(self.data_list)
    # @staticmethod



def load_config_file( pth):
    with open(pth, "r", encoding="utf8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

# @staticmethod
def load_pinyin_dict( filename: str) -> tuple:
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
        _pinyin_dict[tokens[0]] = len(_pinyin_list) - 1  # 对应的位置
    return _pinyin_list, _pinyin_dict

# @staticmethod
def read_wav_data( filename: str) -> tuple:
    """
    读取一个wav文件，返回声音信号的时域谱矩阵和播放时间
    """
    wav = wave.open(filename, "rb")  # 打开一个wav格式的声音文件流
    num_frame = wav.getnframes()  # 获取帧数
    num_channel = wav.getnchannels()  # 获取声道数
    framerate = wav.getframerate()  # 获取帧速率
    num_sample_width = wav.getsampwidth()  # 获取实例的比特宽度，即每一帧的字节数
    str_data = wav.readframes(num_frame)  # 读取全部的帧
    wav.close()  # 关闭流
    wave_data = np.fromstring(str_data, dtype=np.short)  # 将声音文件数据转换为数组矩阵形式
    wave_data.shape = -1, num_channel  # 按照声道数将数组整形，单声道时候是一列数组，双声道时候是两列的矩阵
    wave_data = wave_data.T  # 将矩阵转置
    return wave_data, framerate, num_channel, num_sample_width



if __name__ == '__main__':
    data_set = Mydataset()
    data_input, data_label = data_set.__getitem__(5)
    # print(data_input.shape)
    # print(data_label.shape)



    train_loader = DataLoader(dataset=data_set, batch_size=1, shuffle=False)
    # print('hh')
    for step, (the_input, the_label) in enumerate(train_loader):
        print(step)
        # print('hhhh')
        if step < 3:
            print('data_input.shape-------{}'.format(the_input.shape))
            print('data_label.shape-------{}'.format(the_label.shape))
        else:
            break

