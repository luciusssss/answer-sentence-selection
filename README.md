# Character Embedding + ESIM + Focal Loss for Chinese Answer Sentence Selection 
This is a course project for Web Data Mining. The task is to decide whether a sentence contains the answer to the questions.
We use ESIM ([Enhanced LSTM for Natural Language Inference](https://arxiv.org/abs/1609.06038)) as our main model. [Pretrained Chinese character embedding](https://github.com/Embedding/Chinese-Word-Vectors) is adopted to faciliate character-level matching between questions and answers. We employ focal loss to address the unbalanced label.
A PowerPoint slide is attached in which we further explain our method.

## Requirement
+ Python (>= 3.6)
+ PyTorch (>= 1.0)
+ torchtext

## Dataset 
The dataset for this project is NLPCC DBQA 2016.

## Result
|      |  MAP   | MRR  |
| :-----| ----: | :----: |
| All-0 |  25.30  | 25.81  |
| BERT  | 93.73 |  93.83 | 
| Ours  | 90.33 |  90.48 | 

## Reference
+ [Enhanced LSTM for Natural Language Inference](https://arxiv.org/abs/1609.06038)
+ [神经网络不收敛的11个常见问题](https://zhuanlan.zhihu.com/p/36369878)
+ [自用Pytorch笔记(十五:防止过拟合的几种方法)](https://zhuanlan.zhihu.com/p/69339955)
+ [如何处理不平衡数据集的分类任务](https://zhuanlan.zhihu.com/p/67650069)
+ https://github.com/pengshuang/Text-Similarity/blob/master/models/ESIM.py
+ https://github.com/marvis/pytorch-yolo2/blob/master/FocalLoss.py
