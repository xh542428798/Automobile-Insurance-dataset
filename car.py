# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # 交通事故理赔审核
# ## 本比赛为个人练习赛，适用于入门二元分类模型，主要针对于数据新人进行自我练习、自我提高，与大家切磋。
# 
# 任务类型：二元分类
# 
# 背景介绍:
# 在交通摩擦（事故）发生后，理赔员会前往现场勘察、采集信息，这些信息往往影响着车主是否能够得到保险公司的理赔。训练集数据包括理赔人员在现场对该事故方采集的36条信息，信息已经被编码，以及该事故方最终是否获得理赔。我们的任务是根据这36条信息预测该事故方没有被理赔的概率。   
# 
# 数据文件（三个）：
# train.csv 训练集，文件大小 15.6mb  
# test.csv 预测集, 文件大小 6.1mb  
# sample_submit.csv 提交示例 文件大小 1.4mb  
# 
# 
# 训练集中共有200000条样本，预测集中有80000条样本。
# %% [markdown]
# ### 变量说明：  
# 
# |变量名| 解释  |  
# |  :----:  | :----:  |  
# |  CaseId  |  案例编号，没有实际意义  |  
# | Q1  | 理赔员现场勘察采集的信息，Q1代表第一个问题的信息。信息被编码成数字，数字的大小不代表真实的关系。 |  
# | Qk  | 同上，Qk代表第k个问题的信息。一共36个问题。 |    
# |Evaluation|表示最终审核结果。0表示授予理赔，1表示未通过理赔审核。在test.csv中，这是需要被预测的标签|

# %%
# 读取数据
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
train = pd.read_csv('./train.csv')
validation = pd.read_csv('./test.csv')

# %% [markdown]
# ### test数据目前不在我们的关注范围内，现阶段只分析train数据，首先对train数据进行分割成训练集和测试集。
# ### 对train数据进行观察与可视化

# %%
train.head(3)

# %% [markdown]
# #### 去除Caseid，提取Evaluation

# %%
train = train.drop(columns='CaseId')
train.head(3)


# %%
y = train['Evaluation']
x = train.drop(columns='Evaluation')
x[:3]

# %% [markdown]
# ### 统计各列分布

# %%
x.describe()
# 我们发现每一列不是0和1的关系


# %%
x.Q1.value_counts() # 统计Q1列的频数


# %%
for i in x.columns:
    # print(i)
    print(x[i].value_counts()) # 得到所有列的值频数


# %%
y.value_counts()

# %% [markdown]
# # 主成分分析，降维

# %%
from sklearn.decomposition import PCA
pca = PCA(n_components=2) # 实例化一个pca
x2d = pca.fit_transform(x) # 应用到数据集,得到一个2维数据
pca.components_.T[:,0] # 查看第一个主成分向量矩阵


# %%
print(pca.explained_variance_ratio_) # 查看其中成分的贡献度


# %%
plt.scatter(x2d[:,0],x2d[:,1], c=y, marker="o", cmap="bwr_r")
plt.show() # 从结果看，pca无法把两类分开

# %% [markdown]
# ### 因为该数据全部是因子类型，因此需要制作哑变量

# %%
x.info()
# get_dummies()能够自动转化object属性和category属性的列，int64的不会自动转换
# get_dummies(data, prefix=None, prefix_sep='_', dummy_na=False, columns=None, sparse=False, drop_first=False, dtype=None)
#     Convert categorical variable into dummy/indicator variables
    
#     Parameters
#     ----------
#     data : array-like, Series, or DataFrame
#     prefix : string, list of strings, or dict of strings, default None
#         String to append DataFrame column names.
#         Pass a list with length equal to the number of columns
#         when calling get_dummies on a DataFrame. Alternatively, `prefix`
#         can be a dictionary mapping column names to prefixes.
#     columns : list-like, default None
#         Column names in the DataFrame to be encoded.
#         If `columns` is None then all the columns with
#         `object` or `category` dtype will be converted.


# %%
dump_x = pd.get_dummies(x,prefix = x.columns,columns = x.columns)
dump_x[:3] # 全部转化为哑变量


# %%
from sklearn.model_selection import train_test_split
# help(train_test_split)
x_train, x_test, y_train, y_test = train_test_split(x , y, test_size = 0.2 , random_state = 1)

# %% [markdown]
# # Sklearn线性模型拟合

# %%
from sklearn import linear_model
reg = linear_model.LinearRegression() # 实例化
reg.fit(x_train,y_train) # fit()数据


# %%
reg.coef_

# %% [markdown]
# # 计算误差

# %%
y_pred = reg.predict(x_test)


