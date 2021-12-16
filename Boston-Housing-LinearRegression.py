# !usr\bin\python
# encoding: utf-8
# Author: Tracy TAo (Dasein)
# Date: 2021/11/05
import pandas as pd
import numpy as np
import random
from random import random
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline
import warnings
warnings.filterwarnings('ignore')
sns.set(style='darkgrid',font_scale=1.2)
plt.rcParams['font.family']='SimHei'
plt.rcParams['axes.unicode_minus']=False
import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split #数据集训练集划分
from sklearn import metrics
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report,precision_score,recall_score,f1_score,r2_score #分类报告
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
def leo(x):
    train[x]=LabelEncoder().fit_transform(train[x])


train = pd.read_csv('train.csv',index_col='Id') # 读取训练集
test = pd.read_csv('test.csv',index_col='Id')   # 读取测试集

if __name__ =="__main__":
    print("Train.Head:",train.head(),"\n", "Test.Head", test.head()) # 查看数据集
    print("Train.Shape:", train.shape,"\n", "Test.Shape", test.shape)   # 查看数据量
    # MSSubClass 用于分类, 强制类型转换成字符型数据
    train['MSSubClass'] = train['MSSubClass'].astype(str)
    test['MSSubClass'] = test['MSSubClass'].astype(str)
    print(train['MSSubClass'].value_counts(), "\n", test['MSSubClass'].value_counts())
    # 区分指标性数据和定性数据，并查看详情
    quant = [x for x in train.columns if train[x].dtypes != object]
    quanli = [x for x in train.columns if train[x].dtypes == object]
    print('quant: {}, counts: {}'.format(quant, len(quant)))
    print('----------------------------------------------------------------------------------')
    print('quanli: {}, counts: {}'.format(quanli, len(quanli)))
    count_na = train.isnull().sum().sort_values(ascending=False)
    print("NA:",count_na)   # 查看是否有空数据
    # 查看房价和建造年份的相关性，并且同时观察房价分布区间
    plt.figure(figsize=(30, 10), dpi=100)
    sns.boxplot(train.YearBuilt, train.SalePrice)
    plt.title('YearBuilt - SalePrice Boxplot')
    plt.xticks(rotation=90)
    plt.savefig('YearBuilt - SalePrice Boxplot.png', dpi=80)
    plt.show()
    print('房价描述性统计：',train['SalePrice'].describe())
    # 数据规约：规约房价数据范围
    train = train[train.SalePrice >= 40000]
    train = train[train.SalePrice <= 500000]
    # 将房价按照年份分类，并计算年份的房价平均值
    df1 = train.groupby('YearBuilt').agg({'SalePrice':'mean'})
    plt.figure(figsize=(20,10),dpi=100)
    plt.plot(df1,"*",color="#00338D")
    plt.plot(df1,color="gray")
    plt.title("YearBuild - Price Level")
    plt.savefig('YearBuild - Price Level', dpi =100)
    plt.show()
    print("LotFrontage描述性统计：", train.LotFrontage.describe())
    plt.figure(figsize=(15, 10), dpi=60)
    sns.distplot(train.LotFrontage, color='#00338D')
    plt.title('LotFrontage Distplot')
    plt.savefig('LotFrontage Distplot.png', dpi=100)
    plt.show()
    train = train[train['LotFrontage'] <= 200]
    train.fillna({'LotFrontage': train.LotFrontage.median()}, inplace=True)     # 偏态分布中位数填充
    # print(train.LotFrontage.isnull().sum())
    print('LotArea描述性统计：', train.LotArea.describe())  # [1.3k,215k+]
    plt.figure(figsize=(15, 10), dpi=60)
    sns.distplot(train.LotArea, color='#00338D')
    plt.title('LotArea Distplot')
    plt.show()  # 重复规约数据的过程
    train = train[train['LotArea'] <= 50000]
    df2 = train.groupby('OverallQual').agg({'SalePrice': 'mean'})
    # df2.plot()
    null_var = [i for i in quant if train[i].isnull().sum() > 0]
    print("有数据确实的特征：",null_var)
    plt.figure(figsize=(15, 10), dpi=60)
    sns.distplot(train.GarageYrBlt, color='#00338D')
    plt.title('GarageYrBlt Distplot')
    plt.show()
    train.fillna({'MasVnrArea': train.MasVnrArea.median(), 'GarageYrBlt': train.GarageYrBlt.median()}, inplace=True)
    # null_var = [i for i in quant if train[i].isnull().sum() > 0]
    # null_var
    print("数据清洗结束——————————！")
    for i in quanli:
        print(leo(i))
    test.drop('Utilities', axis=1, inplace=True)
    train.drop('Utilities', axis=1, inplace=True)
    corr = train.corr()
    print(corr)
    # 可视化相关系数矩阵
    plt.figure(figsize=(40, 40), dpi=100)
    sns.set(font_scale=1.2)
    ax = sns.heatmap(corr,
                     xticklabels=corr.columns,
                     yticklabels=corr.columns,
                     linewidths=0.9, annot=True,
                     cbar=True, cmap="rainbow", fmt='.2f',
                     annot_kws={'size': 8})
    plt.title("Corr Heatmap")
    plt.savefig("Corr Heatmap.png")
    plt.show()
    # 特征工程
    strong_var = ['OverallQual', 'YearBuilt', 'YearRemodAdd', 'ExterQual', 'BsmtQual', 'TotalBsmtSF', '1stFlrSF',
                  'GrLivArea', 'KitchenQual',
                  'GarageCars', 'GarageArea']
    k = 15
    var_ = corr.nlargest(k, 'SalePrice')['SalePrice'].index # 不用手动输入的方法
    cm = np.corrcoef(train[var_].values.T)
    # 重新生成热力图
    plt.figure(figsize=(15, 15), dpi=80)
    sns.set(font_scale=1.4)
    hm = sns.heatmap(cm, cbar=True, linewidths=0.5,
                     annot=True, square=True,
                     fmt='.2f', annot_kws={'size': 12}, cmap="rainbow",
                     yticklabels=var_.values, xticklabels=var_.values)
    plt.title("Corr_v2 Heatmap")
    plt.savefig("Corr_v2 Heatmap.png")
    plt.show()
    vars_ = list(var_)
    vars_.remove('TotRmsAbvGrd')
    vars_.remove('GarageArea')
    vars_.remove('1stFlrSF')
    vars_.remove('GarageYrBlt')
    vars_.remove('SalePrice')
    x_train_fnl = train[vars_]
    y_train_fnl = train['SalePrice']
    x_test_fnl = test[vars_]
    x_test_fnl = LabelEncoder().fit_transform(x_test_fnl)
    scaler = StandardScaler(copy=False)
    scaler.fit_transform(x_train_fnl)
    train2 = scaler.transform(x_train_fnl)
    scaler2 = StandardScaler(copy=False)
    scaler2.fit_transform(x_test_fnl)
    x_test = scaler2.transform(x_test_fnl)
    lr = LinearRegression()
    x_train,x_test,y_train,y_test=train_test_split(x_train_fnl,y_train_fnl,test_size=.2,random_state=22)
    lr.fit(x_train, y_train)
    pred = lr.predict(x_test)
    y_test = pd.DataFrame(y_test)
    pred = pd.DataFrame(pred)
    print('r2_score:',r2_score(pred,y_test))
    xx = np.arange(len(y_test))
    plt.figure(figsize=(10, 10), dpi=80)
    plt.scatter(xx, y_test, color="#00338D", s=20, marker='*')
    plt.scatter(xx, pred, color="red", s=20, marker='o')
    plt.show()
    lr.fit(train2, y_train_fnl.values)
    y_train_pred = lr.predict(x_test)
    ypred = pd.DataFrame(y_train_pred)
    # ypred.to_csv('Bostion ypred v1.csv')







