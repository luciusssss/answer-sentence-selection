import torch
import torch.nn as nn
import torch.nn.functional as F


class ESIM(nn.Module):
    def __init__(self, pretrained_embeddings, vocab_size, embedding_dim, hidden_dim, linear_size, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(pretrained_embeddings)
        self.lstm1 = nn.LSTM(embedding_dim, hidden_dim, num_layers=1,
                            bidirectional=True)
        self.lstm2 = nn.LSTM(8 * hidden_dim, hidden_dim, num_layers=1,
                            bidirectional=True)
        self.fc = nn.Sequential(
            nn.BatchNorm1d(hidden_dim * 8),
            nn.Linear(hidden_dim * 8, linear_size),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(linear_size),
            nn.Dropout(dropout),
            nn.Linear(linear_size, linear_size),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(linear_size),
            nn.Dropout(dropout),
            nn.Linear(linear_size, 2),
        )
    
    def forward(self, x1, x2):
        mask1 = x1.permute(1, 0).eq(1) 
        mask2 = x2.permute(1, 0).eq(1)
        # mask = [batch size, sent len]
        x1_emb = self.embedding(x1)
        x2_emb = self.embedding(x2)
        # x_emb = [sent len, batch size, emb dim]
        output1, _ = self.lstm1(x1_emb)
        output2, _ = self.lstm1(x2_emb)
        # output = [sent len, batch size, 2*hidden size]
        output1 = output1.permute(1, 0, 2)
        output2 = output2.permute(1, 0, 2)
        # output = [batch size, sent len, 2*hidden size]
        x1_align, x2_align = self.soft_attention_align(output1, output2, mask1, mask2)
        # x_align = [batch size, sent len, 2 * hidden dim]
        x1_concat = torch.cat((output1, x1_align, output1 - x1_align, output1 * x1_align), dim=-1)
        x2_concat = torch.cat((output2, x2_align, output2 - x2_align, output2 * x2_align), dim=-1)
        # x_concat = [batch size, sent len, 8 * hidden dim]
        composition1, _ = self.lstm2(x1_concat.permute(1, 0, 2))
        composition2, _ = self.lstm2(x2_concat.permute(1, 0, 2))
        # compostion = [sent len, batch size, 2 * hidden dim]
        avep1 = F.avg_pool1d(composition1.permute(1, 2, 0), composition1.size(0)).squeeze(-1)
        maxp1 = F.max_pool1d(composition1.permute(1, 2, 0), composition1.size(0)).squeeze(-1)
        aggr1 = torch.cat((avep1, maxp1), dim=1)
        avep2 = F.avg_pool1d(composition2.permute(1, 2, 0), composition2.size(0)).squeeze(-1)
        maxp2 = F.max_pool1d(composition2.permute(1, 2, 0), composition2.size(0)).squeeze(-1)
        aggr2 = torch.cat((avep2, maxp2), dim=1)
        # aggr = [batch size, 4 * hidden dim]
        
        result = self.fc(torch.cat((aggr1, aggr2), dim=1))
        return result



    def soft_attention_align(self, x1, x2, mask1, mask2):
        # x1 = [batch size, sent len 1, 2 * hidden dim]
        # x2 = [batch size, sent len 2, 2 * hidden dim]

        attention = torch.matmul(x1, x2.transpose(1, 2))
        # attention = [batch size, sent len 1, sent len 2]

        mask1 = mask1.float().masked_fill_(mask1, float('-inf'))
        # mask1 = [batch size, sent len 1]
        mask2 = mask2.float().masked_fill_(mask2, float('-inf'))
        # mask1 = [batch size, sent len 2]

        weight1 = F.softmax(attention + mask2.unsqueeze(1), dim=-1)
        # weight1 = [batch size, sent len 1, sent len 2]
        x1_align = torch.matmul(weight1, x2)
        # x1_align = [batch size, sent len 1, 2 * hidden dim]
        weight2 = F.softmax(attention.transpose(1, 2) + mask1.unsqueeze(1), dim=-1)
        x2_align = torch.matmul(weight2, x1)
        # x2_align = [batch size, sent len 2, 2 * hidden dim]

        return x1_align, x2_align


# https://github.com/marvis/pytorch-yolo2/blob/master/FocalLoss.py
class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in 
        Focal Loss for Dense Object Detection.
            
            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])
    
        The losses are averaged across observations for each minibatch.
        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = torch.ones(class_num, 1).cuda()
        else:
            if isinstance(alpha, torch.tensor):
                self.alpha = alpha
            else:
                self.alpha = torch.tensor(alpha).cuda()
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = torch.tensor(class_mask).cuda()
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]
        
        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss
