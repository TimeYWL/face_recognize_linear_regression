import scipy.io as sio
import random
import numpy as np
from time import sleep
from tqdm import tqdm
import matplotlib.pyplot as plt 

# load dataset
mat = sio.loadmat('AR.mat')
img = mat['alls']
labels = mat['gnd']
label = np.unique(labels)
print(len(labels[0]))

# init
count = 0 # compute acc
acc_sum = 0 # compute tol_acc
k = 2 # k-fold
# 正则参数
## 正则项 为0退化为普通线性回归，大于零均为协同表示（Collaborative Representation）
b = [0]
for i in range(-5, 10, 2):
    b.append(pow(10, i))
# 定义实验结果列表
L1_acc=[]
L2_acc=[]
Linf_acc=[]
Res_acc=[]

print('starting...')
for i in range(k):
    print(i, '-fold:')
    # 将每类样本集K分，并拼接
    test_ind = []
    train_ind = []
    for j in label:
        _, index = np.where(labels == j)
        testind = index[int(i*(len(index)/k)):int((i+1)*(len(index)/k))]
        test_ind.append(testind)
        trainind = list(set(index).difference(set(testind)))
        train_ind.append(trainind)

    test_ind = np.array(test_ind)
    test_ind = test_ind.reshape(len(test_ind)*len(test_ind[0]))

    train_ind = np.array(train_ind)
    train_ind = train_ind.reshape(len(train_ind)*len(train_ind[0]))

    # 取出测试集
    test_img = img[:, test_ind]
    test_label = labels[0, test_ind]
    ## 转换为矩阵
    ### 注意转换数据类型，numpy默认的整型uint8会导致计算溢出
    teMat = np.mat(test_img/1.0)

    # 取出训练集
    #train_ind = test_ind + (5-10*i)
    train_img = img[:, train_ind]
    train_label = labels[0, train_ind]
    ## 转换为矩阵
    trMat = np.mat(train_img/1.0)
    # 计算不同正则参数b下的回归系数
    coeff = []
    for j in b:
        regress_w = (trMat.T * trMat + j * np.identity(len(train_ind))).I * trMat.T * teMat
        coeff.append(regress_w)

    # 计算不同正则参数下的预测结果
    print('computing predict results...')
    for j in tqdm(range(len(coeff))):
        L1_score = [] # 存储1范数度量分值
        L2_score = [] # 存储2范数度量分值
        Linf_score = [] # 存储无穷范数度量分值
        res_score = [] # 存储残差度量分值
        # 计算不同类别的预测结果
        for m in label:
            temp_ind = [l for l,x in enumerate(train_label) if x == m]
            Mat = (coeff[j])[temp_ind, :]
            # 利用重构样本残差度量
            XMat = trMat[:, temp_ind]
            tePre = XMat*Mat
            Re_dist = teMat - tePre
            res_score.append(sum(np.power(Re_dist, 2)))

            # 利用1范数度量
            j1scores = sum(map(abs, Mat))
            L1_score.append(j1scores)

            # 利用2范数度量
            j2scores = sum(np.power(Mat, 2))
            L2_score.append(j2scores)

            # 利用无穷范数度量
            jinfscores = np.max(np.abs(Mat), axis=0)
            Linf_score.append(jinfscores)
        
        res_score = np.array(res_score).reshape(len(label), len(test_ind))
        L1_score = np.array(L1_score).reshape(len(label), len(test_ind))
        L2_score = np.array(L2_score).reshape(len(label), len(test_ind))
        Linf_score = np.array(Linf_score).reshape(len(label), len(test_ind))
                
        # 度量分值降序排序，进行分类（采用1范数度量分值），计算分类精度
        index=[]
        for i in range(len(test_ind)):
            index.append(np.argmax(L1_score[:, i]))
        predicts = label[index]
        difs = predicts - test_label
        L1_acc.append(np.sum(difs==0)/len(test_ind))

        # 度量分值降序排序，进行分类（采用2范数度量分值），计算分类精度
        index=[]
        for i in range(len(test_ind)):
            index.append(np.argmax(L2_score[:, i]))
        predicts = label[index]
        difs = predicts - test_label
        L2_acc.append(np.sum(difs==0)/len(test_ind))

        # 度量分值降序排序，进行分类（采用无穷范数度量分值），计算分类精度
        index=[]
        for i in range(len(test_ind)):
            index.append(np.argmax(Linf_score[:, i]))
        predicts = label[index]
        difs = predicts - test_label
        Linf_acc.append(np.sum(difs==0)/len(test_ind))

        # 度量分值降序排序，进行分类（采用残差度量分值），计算分类精度
        index=[]
        for i in range(len(test_ind)):
            index.append(np.argmin(res_score[:, i]))
        predicts = label[index]
        difs = predicts - test_label
        Res_acc.append(np.sum(difs==0)/len(test_ind))

# 保存
results=[L1_acc, L2_acc,Linf_acc, Res_acc]
np.savetxt("results.txt", results, fmt="%6.4f")

results=np.array([L1_acc, L2_acc,Linf_acc, Res_acc]).reshape(4,k,9)

# 画图
print('plting...')
labels=['L1', 'L2', 'Linf', 'Res']
mark=['s', 'o', '^', 'x']
b=range(-7,10,2)
k=0 # 选择第k折结果作图
for i in range(4):
    plt.plot(b, results[i,k,:], Marker=mark[i], label=labels[i])
plt.xticks(b)
plt.ylim(0,1)
plt.legend()
plt.savefig('results.png')
