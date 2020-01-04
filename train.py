import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchtext import data
import random
from model import ESIM
from model import FocalLoss
import os
import time
from torchtext import vocab
import re

SEED = 2020
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


def tokenizer(text):
    tokenized = []
    for i in text:
        if i != ' ':
            tokenized.append(i)
    if len(tokenized) == 0:
        tokenized.append("无")
    return tokenized

TEXT = data.Field(lower=True, tokenize=tokenizer)
LABEL = data.LabelField()


print("start loading data")
train_data, valid_data = data.TabularDataset.splits(
    path='data', train='train.csv',
    validation='valid.csv', 
    format='csv', skip_header=False,
    csv_reader_params={'delimiter':'\t'},
    fields=[('text1',TEXT), ('text2',TEXT), ('label',LABEL)]
)

print('train_data[0]', vars(train_data[1]))


vectors = vocab.Vectors(
    name = "sgns.merge.char",    #args.pretrainedEmbeddingName由参数指定
    cache = "."    #args.pretrainedEmbeddingPath由参数指定
)

TEXT.build_vocab(train_data, max_size=80000, vectors=vectors)
print("Unique tokens in TEXT vocabulary:", len(TEXT.vocab))
LABEL.build_vocab(train_data)
print(LABEL.vocab.stoi)


N_EPOCHS = 80
VOCAB_SIZE = len(TEXT.vocab)
EMBEDDING_DIM = 300
HIDDEN_DIM = 100
LINEAR_SIZE = 200
DROPOUT = 0.5
BATCH_SIZE = 128


device = torch.device('cuda')

train_iterator, valid_iterator = data.BucketIterator.splits(
    (train_data, valid_data),
    batch_size=BATCH_SIZE, sort_key=lambda x: len(x.text1),
    device=device, shuffle=True)


pretrained_embeddings = TEXT.vocab.vectors
model = ESIM(pretrained_embeddings, VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, LINEAR_SIZE, DROPOUT)

optimizer = optim.Adam(model.parameters())
criterion = FocalLoss(2)
model = model.to(device)
criterion = criterion.to(device)

def categorical_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    max_preds = preds.argmax(dim=1, keepdim=True) # get the index of the max probability
    correct = max_preds.squeeze(1).eq(y)
    return correct.sum()/torch.FloatTensor([y.shape[0]])

def train(model, iterator, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    cnt = 0
    
    model.train()
    
    for batch in iterator:
        
        optimizer.zero_grad()
        
        predictions = model(batch.text1, batch.text2)
        
        loss = criterion(predictions, batch.label)
        # # l1正则化
        # L1_reg = 0
        # for param in model.parameters():
        #     L1_reg += torch.sum(torch.abs(param))
        # loss += 0.001 * L1_reg  # lambda=0.001

        acc = categorical_accuracy(predictions, batch.label)
        
        loss.backward()
        
        optimizer.step()
        
        cnt += predictions.size(0)
        if (cnt % 10000 == 0):
            print("cnt:", cnt)
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
    
        for batch in iterator:

            predictions = model(batch.text1, batch.text2)
            
            loss = criterion(predictions, batch.label)
            
            acc = categorical_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


print("start training!")

best_valid_loss = float('inf')
best_valid_acc = 0

for epoch in range(N_EPOCHS):

    start_time = time.time()
    
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
    
    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    
    f_log = open("log_esim.txt", 'a')
    f_log.write('Epoch: %d | Epoch Time: %dm %ds\n' % (epoch+1, epoch_mins, epoch_secs))
    f_log.write('\tTrain Loss: %.3f | Train Acc: %.2f %%\n' % (train_loss, train_acc*100))
    f_log.write('\t Val. Loss: %.3f |  Val. Acc: %.2f %%\n' % (valid_loss, valid_acc*100))
    print('Epoch: %d | Epoch Time: %dm %ds' % (epoch+1, epoch_mins, epoch_secs))
    print('\tTrain Loss: %.3f | Train Acc: %.2f %%' % (train_loss, train_acc*100))
    print('\t Val. Loss: %.3f |  Val. Acc: %.2f %%' % (valid_loss, valid_acc*100))

    if valid_loss <  best_valid_loss:
        best_valid_loss = valid_loss
        best_valid_acc = valid_acc
        torch.save(model.state_dict(), './saved_model/esim.pt')
        print("New model saved!")
        f_log.write("New model saved!\n")
    
    f_log.flush()
    f_log.close()


model.load_state_dict(torch.load('./saved_model/esim.pt'))
model.eval()

f_valid = open("data/test-set.data", "r", encoding='utf-8')
f_res = open('prediction.txt', 'w')
for i, rowlist in enumerate(f_valid):
    rowlist = rowlist[:-1].split('\t')
    input_sent = []
    for sent in rowlist[:2]:
        tokenized = tokenizer(sent)
        indexed = [TEXT.vocab.stoi[t] for t in tokenized]
        tensor = torch.LongTensor(indexed).to(device)
        tensor = tensor.unsqueeze(1)
        input_sent.append(tensor)
    ans = F.softmax(model(input_sent[0], input_sent[1])[0])[1].item()
    f_res.write(str(ans) + '\n')
f_valid.close()