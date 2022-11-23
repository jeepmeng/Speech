from contextnet.model import ContextNet
import torch


def create_modle():
    batch_size = 3
    seq_length = 500
    input_size = 80
    vocab_size =0
    vocab_pth ='dict.txt'

    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
if __name__ == '__main__':
    create_modle()