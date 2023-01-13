
from torch import nn
import torchaudio

class RNN_T_LOSS(nn.Module):
    def __init__(self):
        super(RNN_T_LOSS, self).__init__()

        # self.ctc = nn.CTCLoss(blank=0, reduction='sum', zero_infinity=True)
        self.ctc = torchaudio.transforms.RNNTLoss()
    def forward(self, logits, targets, preds_lengths, target_length):
        ctc_loss = self.ctc(logits = logits,
                            targets = targets,
                            logit_lengths = preds_lengths,
                            target_lengths = target_length)

        return ctc_loss


def build_criterion(conf):
    loss_type = conf['setting']['loss_type']
    if loss_type == 'RNN_T_LOSS':
        criterion = RNN_T_LOSS().to('cuda')
        # criterion: RNN_T_LOSS = RNN_T_LOSS().to('cuda')

    return criterion