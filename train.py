import numpy as np

from contextnet.model import ContextNet
import create_dataset
import torch


def create_modle():
    batch_size = 3
    seq_length = 500
    input_size = 80
    vocab_size =0
    vocab_pth ='dict.txt'
    data_set = create_dataset.Mydataset()
    data_input, data_label = data_set.__getitem__(5)
    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')

    data_input = data_input[np.newaxis,:]
    data_label = data_label[np.newaxis,:]
    print(data_input.shape)
    print(data_label.shape)

    model = ContextNet(
        model_size='medium',
        num_vocabs=10,
    ).to(device)
    print(data_input.shape[1])
    print(data_label.shape[1])
    outputs = model(data_input, data_input.shape[2], data_label, data_label.shape[1])
    print(outputs.zize())

    # cuda = torch.cuda.is_available()
    # device = torch.device('cuda' if cuda else 'cpu')
if __name__ == '__main__':
    create_modle()