# Sberbank_project

input文件夹里是输入文件

model文件夹里是比赛过程中保留的模型

output文件夹里是比赛中得到的输出文件

script文件夹里是比赛的代码

其中random.py是随机森林的实现

gradient.ipynb是GBDT的实现

averaging.ipynb是模型加权融合+对结果线性标定

getNewDataset.ipynb将特征工程和数据处理后得到的数据集写入文件new_train.csv和

new_test.csv

stacking.ipynb使用了新的数据集进行stacking模型融合，这份代码得到了最好的结果。

代码参考出处：

https://www.kaggle.com/mgierlach/gradient-in-python-2-0-313?scriptVersionId=1194545

https://www.kaggle.com/saeedt2977/latest-iteration-in-this-silly-game-lb-0-31039

https://dnc1994.com/2016/04/rank-10-percent-in-first-kaggle-competition/

