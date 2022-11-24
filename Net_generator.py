import torch
import numpy as np
from dataset import DataLoader



def _data_generator(self, batch_size, data_loader):


    labels = np.zeros((batch_size, 1), dtype=np.float)
    data_count = data_loader.get_data_count()
    index = 0

    while True:
        X = np.zeros((batch_size,) + self.speech_model.input_shape, dtype=np.float)  # 定义了一个（batchsize，1600, 200, 1)的矩阵
        y = np.zeros((batch_size, self.max_label_length), dtype=np.int16)
        input_length = []
        label_length = []

        for i in range(batch_size):
            wavdata, sample_rate, data_labels = data_loader.get_data(index)
            data_input = self.speech_features.run(wavdata, sample_rate)
            data_input = data_input.reshape(data_input.shape[0], data_input.shape[1], 1)
            # 必须加上模pool_size得到的值，否则会出现inf问题，然后提示No valid path found.
            # 但是直接加又可能会出现sequence_length <= xxx 的问题，因此不能让其超过时间序列长度的最大值，比如200
            pool_size = self.speech_model.input_shape[0] // self.speech_model.output_shape[0]
            inlen = min(data_input.shape[0] // pool_size + data_input.shape[0] % pool_size,
                        self.speech_model.output_shape[0])
            input_length.append(inlen)

            X[i, 0:len(data_input)] = data_input
            y[i, 0:len(data_labels)] = data_labels
            label_length.append([len(data_labels)])

            index = (index + 1) % data_count

        label_length = np.matrix(label_length)
        input_length = np.array([input_length]).T

        yield [X, y, input_length, label_length], labels