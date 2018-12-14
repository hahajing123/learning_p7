
# coding: utf-8

# In[1]:



import pandas as pd
from sklearn.cross_validation import train_test_split
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy  as np
#import lightgbm as lgb


# In[2]:


orial_data = pd.read_csv('data/train.csv',parse_dates=[2])
orial_data.head()
orial_data.shape[0]


# In[3]:


#数据拷贝一份进行处理，避免 对原始数据的改变
data_data = orial_data.copy()
#data_data.describe()
data_data.shape[0]


# In[4]:


# 加载store 数据
store = pd.read_csv('data/store.csv')
data_store = store.copy()
#异常值处理
fill_values = {'CompetitionOpenSinceYear': 0, 'CompetitionDistance': 1, 'CompetitionOpenSinceMonth': 0, 'CompetitionOpenSinceYear': 0,'Promo2SinceWeek':0,'Promo2SinceYear':0,'PromoInterval':'None' }
data_store.fillna(value=fill_values,inplace = True)
store_drop_columns = ['CompetitionOpenSinceMonth','CompetitionOpenSinceYear','Promo2SinceWeek','Promo2SinceYear','PromoInterval']
data_store.drop(store_drop_columns,axis=1,inplace=True)
data_store.head(20)
data_store.shape[0]


# In[5]:


#加载test 数据
test = pd.read_csv('data/test.csv')
data_test = test.copy()
data_test.fillna(value={'Open':1},inplace=True)
data_test.head(10)


# In[6]:


#将字符的属性转换成数字
replace_data = {'a':1,'b':2,'c':3,'d':4}
print(type(data_store))
print(type(data_store['Assortment']))
print(type(data_store.Assortment))
data_store['Assortment'].replace(replace_data,inplace=True)
data_store['StoreType'].replace(replace_data,inplace=True)

data_store.head(10)

data_data['StateHoliday'].replace(replace_data,inplace=True)
data_data['StateHoliday'] = data_data['StateHoliday'].apply(pd.to_numeric)
data_data.shape[0]

data_test['StateHoliday'].replace(replace_data,inplace=True)
data_test['StateHoliday'] = data_test['StateHoliday'].apply(pd.to_numeric)
print(data_test['StateHoliday'].unique())


# In[7]:


# 进行归一化 对CompetitionDistance

scaler = MinMaxScaler()
x = data_store['CompetitionDistance'].values.reshape(-1,1)
data_store['CompetitionDistance'] = scaler.fit_transform(x)
data_store.head(5)


# In[8]:


#pd.get_dummies(data_store)
data_test.head(1)


# In[9]:


data_data.head(3)


# In[10]:


# 提取出月份、年份,数据增加年份和月份的列
import time, datetime
def conver_date(data):
    print(data.head(5))
    data_month = pd.to_datetime(data['Date'],format='%Y-%m-%d %H:%M:%S')
    data_date = pd.to_datetime(data['Date'],format='%Y/%m/%d')
    data['year'] = data_date.dt.year
    data['month'] = data_date.dt.month
    data['day'] = data_date.dt.day
    data.drop('Date',axis=1,inplace=True)
    return data
data_data = conver_date(data_data)
data_test = conver_date(data_test)
data_test.head(5)
    
#print(time.strftime('%Y-%m-%d %H:%M:%S'))
#data_month = pd.to_datetime(data_data['Date'],format='%Y-%m-%d %H:%M:%S')

#record['ym']=record['HAPPEN_TIME']
#data_year =data_data['Date'].apply(lambda x:x.split('-')[0])
#data_date = pd.to_datetime(data_data['Date'],format='%Y/%m/%d')
#data_data['year'] = data_date.dt.year
#data_data['month'] = data_date.dt.month
#data_data['day'] = data_date.dt.day
#data_data.drop('Date',axis=1,inplace=True)


#month2str = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',7:'Jul',8:'Aug',9:'Sept',10:'Oct',11:'Nov',12:'Dec'}
#data_data['month2str'] = data_data.month.map(month2str)
#data_data.head(25)
#data_data.shape[0]


# In[11]:


data_data.shape[0]
#选取open = 1 的数据
data_data = data_data.loc[(data_data['Open'] == 1)]
print(data_data.shape[0])
data_data  = data_data.loc[(data_data['Sales'] > 0)]
print(data_data.shape[0])


# In[12]:


#选取某一个店的ID,获取其销售记录 并显示其每个月的销售情况
def get_month_sales_by_id(id):
    store_data = data_data.loc[(data_data['Store'] == 1)]
    store_data_2013 = store_data.loc[store_data['year'] == 2013]
    store_data_2014 = store_data.loc[store_data['year'] == 2014]
    store_data_2015 = store_data.loc[store_data['year'] == 2015]
    
    #计算每年每个月的销售情况
    store_data_2013_month = store_data_2013.groupby(by=['month'])['Sales'].sum()
    store_data_2014_month = store_data_2014.groupby(by=['month'])['Sales'].sum()
    store_data_2015_month = store_data_2015.groupby(by=['month'])['Sales'].sum()
    store_data_2015_month.rename(columns={"month":"sum_of_value"},inplace=True)
    
    #print(store_data_2015_month.to_frame().columns())
    #print(store_data_2015_month[1])
    #print(store_data_2015_month.head(20))
    store_data_month={}
    store_data_month['2013'] = store_data_2013_month
    store_data_month['2014'] = store_data_2014_month
    store_data_month['2015'] = store_data_2015_month
    
    return store_data_month
a=get_month_sales_by_id(5)
print("2014 1",a['2014'][1])


# In[13]:


month_info = get_month_sales_by_id(5)
month_info['2015'][8] = month_info['2015'][9] = month_info['2015'][10] =  month_info['2015'][11] = month_info['2015'][12] = 0
for i in month_info:
    print(i)
    print(month_info[i])


# In[14]:




#解决中文乱码的问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.family']='sans-serif'
#解决负号'-'显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 22

#X=[0,1,2,3,4,5,6,7,8,9,10,11]
x_labels = ['1月','2月','3月','4月','5月','6月','7月','8月','9月','10月','11月','12月']
X = np.arange(12)
bar_width = 0.2
Y=[222,42,455,664,454,334,222,42,455,664,454,334]

fig = plt.figure(figsize=(10,15))
ax = plt.subplot()
bars = []
years = []
ij = 0
for i in month_info:
    ij = ij +1
    bar_i = ax.bar(X+bar_width*ij,month_info[i],bar_width)
    bars.append(bar_i)
    years.append(i)

#bar_1 = ax.bar(X+bar_width,month_info['2013'],bar_width)
#bar_2 = ax.bar(X+bar_width*2,month_info['2014'],bar_width)
#bar_3 = ax.bar(X+bar_width*3,month_info['2015'],bar_width)
ax.set_xticklabels(x_labels)
plt.xticks(X+bar_width*2)
plt.xlim(0,len(x_labels))
plt.title('2013-2015年月销售额')
handles, labels = ax.get_legend_handles_labels()

plt.legend(bars, years,loc = 'best')
#plt.set_xticks(x_labels)
#plt.savefig('./year_month.jpg')
plt.show()


# In[15]:


#归一化处理
scaler = MinMaxScaler()
x = data_data['Customers'].values.reshape(-1,1)
data_data['Customers'] = scaler.fit_transform(x)
data_data.head(5)


# In[16]:


data_data = data_data.merge(data_store,left_on = 'Store',right_on = 'Store',how="left")
print(data_data.shape[0])
data_data.head(10)
#


# In[17]:


#test 数据获取open=1 的数据
print(data_test.shape[0])
data_test_noOpen = data_test.loc[(data_test['Open'] == 0)]
data_test_noPenIds = data_test_noOpen['Id']

data_test = data_test.loc[(data_test['Open'] == 1)]
data_test_ids = data_test['Id']
data_test.drop(['Id'],axis=1,inplace = True)
print(data_test.shape[0])
print(data_test_noOpen.shape[0])
print(data_test_noOpen)
#data=data_test[~((data_test['Open']==0)|(data_test['Open']==1))]
#print(data)


# In[18]:


data_test = data_test.merge(data_store,left_on = 'Store',right_on = 'Store',how="left")
print(data_test.shape[0])
data_test.head(10)


# In[19]:


#test 


# In[20]:


#dummies 独热编码
print(data_data.shape[0])
if 'StoreType_1'  not in data_data.columns:
    data_data = pd.get_dummies(data_data,columns=['StoreType','StateHoliday'])
    print(data_data.shape)
    print(data_data.columns.values)
    print(data_data.head(5))
    print("*************")
    print(data_test.shape[0])
    data_test = pd.get_dummies(data_test,columns=['StoreType','StateHoliday'])
    data_test['StateHoliday_2'] = 0
    data_test['StateHoliday_3'] = 0
    data_data.sort_index(axis=1,inplace=True)
    data_test.sort_index(axis=1,inplace=True)
    print(data_test.shape)
    print(data_test.columns.values)
    print(data_test.head(5))


# In[21]:



#duplicate_columns = data_data.columns[data_data.columns.duplicated()]
#print(duplicate_columns)
data_data.shape[0]

#print(type(data_data['stateHoliday'][63557]))
 


# In[22]:


import math
def rmspe_xgboost(preds, dtrain):       # written by myself
    labels = dtrain.get_label()
    # return a pair metric_name, result
    # since preds are margin(before logistic transformation, cutoff at 0)
    err = np.mean(((labels-preds)/labels)**2)
    return 'rmspe_xgboost',math.sqrt(err)


# In[23]:


from math import sqrt
def rmspe(y,y_pre):
    print("y-y_pre/y",(y-y_pre)/y)
    print("(y-y_pre/y)**2",((y-y_pre)/y)**2)
    a = np.mean(((y-y_pre)/y)**2)
    return sqrt(a)
    #print('y:',y)
    #print('y_pre:',y_pre)
    #print("y/y_pre:",(y_pre / y - 1)**2)
    #print("mean:",np.mean((y_pre / y - 1)**2))
    #return np.sqrt(np.mean((y_pre / y - 1) ** 2))


# In[24]:


'''
# 训练集和测试集随机划分
#train_data = data_data.drop(['Sales'],axis=1)
if 'Customers' in data_data.columns:
    data_data.drop(['Customers'],axis=1,inplace=True)
X_train, X_valid = train_test_split(data_data, test_size=0.2, random_state=10)
#print("+++++++++++++++++++++++++++++++++++")
#print(X_train)
#print(X_valid)
y_train = np.log1p(X_train.Sales)
y_valid = np.log1p(X_valid.Sales)
#y_train = X_train.Sales
#y_valid = X_valid.Sales

print(type(X_train.Sales))
print("x_TRAIN:",X_train.Sales.shape[0])
#X_train.drop(['Sales'],axis=1,inplace=True)
print("####################")
#print(type(X_train))
#print(type(y_train))
X_train.drop(['Sales'],axis=1,inplace=True)
X_valid.drop(['Sales'],axis=1,inplace=True)

dtrain = xgb.DMatrix(X_train,label=y_train)
dvalid = xgb.DMatrix(X_valid,label=y_valid)
print(X_train.shape[0])

print(type(y_train))
print("y_train:",y_train.shape[0])
#print(X_train.Sales.unique())
num_boost_round = 50
watch_list= [(dtrain, 'train'), (dvalid, 'valid')]
params = {"objective": "reg:linear","booster": "gbtree", "eta": 0.5,"max_depth": 10,"min_child_weight":5}
print("start train data by xgboost")
xgboost_model = xgb.train(params, dtrain, num_boost_round,evals=watch_list)
print("valid....")
y_pre = xgboost_model.predict(dvalid)


print(y_pre)
print(len(y_pre))
print(len(y_valid))
print(type(y_valid))
'''


# In[25]:


# 训练集和测试集手动划分
if 'Customers' in data_data.columns:
    data_data.drop(['Customers'],axis=1,inplace=True)
X_train = data_data[0:813766]
X_valid = data_data[813766::]
#print(X_train)
#print("*********************")
#print(X_valid)
#X_train, X_valid = train_test_split(data_data, test_size=0.2, random_state=10)
#print("+++++++++++++++++++++++++++++++++++")
#print(X_train)
#print(X_valid)

y_train = np.log1p(X_train.Sales)
y_valid = np.log1p(X_valid.Sales)
#y_train = X_train.Sales
#y_valid = X_valid.Sales

print(type(X_train.Sales))
print("x_TRAIN:",X_train.Sales.shape[0])
#X_train.drop(['Sales'],axis=1,inplace=True)
print("####################")
#print(type(X_train))
#print(type(y_train))
X_valid_sales =X_valid['Sales'] 
X_train.drop(['Sales'],axis=1,inplace=True)
X_valid.drop(['Sales'],axis=1,inplace=True)

dtrain = xgb.DMatrix(X_train,label=y_train)
dvalid = xgb.DMatrix(X_valid,label=y_valid)
print(X_train.shape[0])

print(type(y_train))
print("y_train:",y_train.shape[0])
print("***********************************")
print(X_valid.shape[0])
#print(X_train.Sales.unique())


# In[26]:


#************** 训练得分的读写 ***********************
import json
def write_record(data,file):
    jsObj = json.dumps(data)
    fileObject = open(file, 'w')
    fileObject.write(jsObj)
    fileObject.close()

# 读log 记录
def read_record(file):
    fileObject = open(file, 'r')
    file_txt = fileObject.read()
    json_content = json.loads(file_txt)
    return json_content


# In[ ]:


#********************开始训练***************************
num_boost_round = 25000
min_child_weight = 5
max_depth=10
eta = 0.01
watch_list= [(dtrain, 'train'), (dvalid, 'valid')]
evals_result = dict()
params = {"objective": "reg:linear","booster": "gbtree", "eta": eta,"max_depth":max_depth,"min_child_weight":min_child_weight} #"min_child_weight":5
print("start train data by xgboost")
xgboost_model = xgb.train(params, dtrain, num_boost_round,feval=rmspe_xgboost,evals=watch_list,verbose_eval=10,early_stopping_rounds=500,evals_result=evals_result)
print("valid....")
#保存模型
model_name = "model/num"+str(num_boost_round)+"_weight"+str(min_child_weight)+"_maxdepth"+str(max_depth)+"_eta"+str(eta)+".model"
log_name= "record/num"+str(num_boost_round)+"_weight"+str(min_child_weight)+"_maxdepth"+str(max_depth)+"_eta"+str(eta)+".txt"
xgboost_model.save_model(model_name)
print("save...",model_name)
write_record(evals_result,log_name)
print(".....save log ..........",log_name)


# In[ ]:


#查看loss 记录
train_log = evals_result['train']
valid_log = evals_result['valid']
show_num = 1200
if(show_num >0):
    log_index = np.arange(show_num)#len(train_log['rmspe_xgboost'])

    plt.plot(log_index,train_log['rmspe_xgboost'][0:show_num],linewidth = 1,label = 'train_rmspe')
    plt.legend()
    plt.plot(log_index,valid_log['rmspe_xgboost'][0:show_num],linewidth = 1,label = 'valid_rmspe')
    plt.legend()
else:
    log_index = np.arange(len(train_log['rmspe_xgboost']))
    plt.plot(log_index,train_log['rmspe_xgboost'],linewidth = 1,label = 'train_rmspe')
    plt.legend()
    plt.plot(log_index,valid_log['rmspe_xgboost'],linewidth = 1,label = 'valid_rmspe')
    plt.legend()
    
    


# In[ ]:


print("best best_ntree_limit",xgboost_model.best_ntree_limit)
print("best_score:",xgboost_model.best_score)
print("bst.best_iteration",xgboost_model.best_iteration)


# In[ ]:


#lightgbm
'''
if 'Customers' in data_data.columns:
    data_data.drop(['Customers'],axis=1,inplace=True)
X_train = data_data[0:813766]
X_valid = data_data[813766::]

y_train = np.log1p(X_train.Sales)
y_valid = np.log1p(X_valid.Sales)

X_valid_sales =X_valid['Sales'] 
X_train.drop(['Sales'],axis=1,inplace=True)
X_valid.drop(['Sales'],axis=1,inplace=True)

lgb_train = lgb.Dataset(X_train,y_train)
lgb_eval = lgb.Dataset(X_valid,y_valid,reference=lgb_train)

num_boost_round = 100
params = {
    'task':'train',
    'boosting_type':'gbdt',
    'objective':'regression',
    'metric':{'l2','mae'},
    'num_leaves':31,
    'learning_rate':0.2,
    'feature_fraction':0.9,
    'bagging_fraction':0.8,
    'bagging_freq':5,
    'verbose':0,
    'max_depth':10
}

gbm = lgb.train(params,lgb_train,num_boost_round=num_boost_round,valid_sets=lgb_eval,early_stopping_rounds=5)
 
y_pre = gbm.predict(X_valid,num_iteration=gbm.best_iteration)
#print(mean_squared_error(y_valid,lgb_predit) ** 0.5)
print("............... valid ...................")
print(y_pre)
'''


# In[ ]:


#预测数据
#model_name="model/num20000_weight3_maxdepth6_eta0.01.model"
print(model_name)
xgboost_model = xgb.Booster(model_file=model_name)
y_pre = xgboost_model.predict(dvalid)


print(y_pre)
print('y_pre',len(y_pre))
print(len(y_valid))
print('X_valid_sales',len(X_valid_sales))
print(type(y_valid))

#load_model = booster.load_model('model/num20_w3_eta001.model')
#error = rmspe(np.expm1(y_valid.values), np.expm1(y_pre))
error = rmspe(np.expm1(y_valid.values),np.expm1(y_pre))
#print(type(np.expm1(y_pre)))
#print(type(y_valid.values))
print("*********************")
print((y_pre))
#print("pre:",y_pre)
print("valid_value:",y_valid.values)
#print(error)
print('RMSPE: {:.6f}'.format(error))


# In[ ]:


''' 

#lightgbm 整合预测数据
print("********************* test ***************************")
y_test =  gbm.predict(data_test,num_iteration=gbm.best_iteration)
y_test = np.expm1(y_test)
print(y_test)
noOpen_nums = data_test_noPenIds.shape[0]
test_noOpen_data = np.zeros(noOpen_nums)

result_noOpen={'Id':data_test_noPenIds,'Sales':test_noOpen_data}
result_noOpen = pd.DataFrame(result_noOpen)
print(result_noOpen['Sales'])
print("&&&&&&&&&&&&&&&&&&&&&&&")
print(data_test_noPenIds.shape)
print(result_noOpen)

result={'Id':data_test_ids,'Sales':y_test}
print(type(result))
result = pd.DataFrame(result)
print(result)

#数据整合
result_total = result.append(result_noOpen)
print(result_total.shape)
print(result_total)
print("%%%%%%%%%%%%%%%%%%")
result_total.sort_values(by = 'Id',axis = 0,ascending = True,inplace=True)

result_total.to_csv ("data/result_light.csv" , encoding = "utf-8",index=False)
print(result_total.index)
'''


# In[ ]:


'''
learn_model = xgb.cv(params, dtrain, 1500,feval=rmspe_xgboost,verbose_eval=10)
print("sss")
'''


# In[ ]:


'''
#print(learn_model)
learn_model.loc[30:,["test-rmspe_xgboost-mean", "train-rmspe_xgboost-mean"]].plot()
print('y_pre',len(y_pre))
print(len(y_valid))
print('X_valid_sales',len(X_valid_sales))
'''


# In[ ]:


#特征重要性
import operator
importance = xgboost_model.get_fscore()
importance = sorted(importance.items(), key=operator.itemgetter(1))
df = pd.DataFrame(importance, columns=['feature', 'fscore'])
df['fscore'] = df['fscore'] / df['fscore'].sum()

plt.figure()
df.plot()
df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(16, 10))
plt.title('XGBoost Feature Importance')
plt.xlabel('relative importance')

plt.show()


# In[ ]:


#测试数据集预测
#dtest_ids = data_test['Id']
#data_test.drop(['Id'],axis=1,inplace=True)
dtest = xgb.DMatrix(data_test)
print(data_test.shape)
#data_test_s0 = data_test['SchoolHoliday_0']
#data_test_s1 = data_test['SchoolHoliday_1']
#data_test.drop('SchoolHoliday_0',axis=1,inplace=True)
#data_test.drop('SchoolHoliday_1',axis=1,inplace=True)
#data_test.insert(18,'SchoolHoliday_0',data_test_s0)
#data_test.insert(19,'SchoolHoliday_1',data_test_s1)

print(data_test.columns.values)
print("%%%%%%%%%%%%%%%%%%%")
print(X_train.shape)
print(X_train.columns.values)
y_test = xgboost_model.predict(dtest)


y_test = np.expm1(y_test)
print(y_test)

print('y_pre',y_pre.shape)
print(len(y_valid))
print('X_valid_sales',X_valid_sales.shape)


# In[ ]:


print('y_pre',y_pre.shape)
print(len(y_valid))
print('X_valid_sales',X_valid_sales.shape)


# In[ ]:


result={'Id':data_test_ids,'Sales':y_test}
print(type(result))
result = pd.DataFrame(result)
print(result)


# In[ ]:


#处理open=0 的数据
print(result.loc[(result['Id']==544)])
print("******************************")
noOpen_nums = data_test_noPenIds.shape[0]
test_noOpen_data = np.zeros(noOpen_nums)

result_noOpen={'Id':data_test_noPenIds,'Sales':test_noOpen_data}
result_noOpen = pd.DataFrame(result_noOpen)
print(result_noOpen['Sales'])
print("&&&&&&&&&&&&&&&&&&&&&&&")
print(data_test_noPenIds.shape)
print(result_noOpen)


# In[ ]:


result_total = result.append(result_noOpen)
print(result_total.shape)
print(result_total)
print("%%%%%%%%%%%%%%%%%%")
result_total.sort_values(by = 'Id',axis = 0,ascending = True,inplace=True)

result_total.to_csv ("data/result.csv" , encoding = "utf-8",index=False)
print(result_total.index)


# In[ ]:


#print(result.loc[(result['Id']==41088)])


# In[ ]:


#从验证集合
import random
#print(type(X_valid_sales))
random_i = random.randint(0,150)
data_size = 200
print("random_i",random_i)
print(X_valid_sales.shape)
print(y_pre.shape)
print(np.expm1(y_pre[random_i*data_size:(random_i+1)*data_size]))
#print(y_valid[random_i*data_size:(random_i+1)*data_size])
fig = plt.figure(figsize=(20,8))
x_index = np.arange(200)
print(x_index)
plt.plot(x_index,X_valid_sales[random_i*data_size:(random_i+1)*data_size],linewidth = 1,label = 'valid_label')
plt.legend()
plt.plot(x_index,np.expm1(y_pre[random_i*data_size:(random_i+1)*data_size]),linewidth = 1,label = 'valid_predict')
plt.legend()
plt.title('随机获取200条记录验证集预测情况')
plt.xlabel('数据ID')
plt.ylabel('销售额')
plt.savefig('predict_1.jpg')


# In[ ]:


#残差图
valid_sub = np.expm1(y_pre) - X_valid_sales
#print(valid_sub)
print('min:',math.ceil(min(valid_sub)))
print('max:',math.floor(max(valid_sub)))
bins = np.linspace(math.ceil(min(valid_sub)),
                   math.floor(max(valid_sub)),40) # fixed number of bins
plt.xlim([-4000, 4000])
 
plt.hist(valid_sub, bins=bins, alpha=0.5)
plt.title('valid data 残差图-直方图')
plt.xlabel('y_pre-y_label (40 evenly spaced bins)')
plt.ylabel('count')


# In[ ]:


#散点图
print(X_valid_sales.index)
plt.scatter(X_valid_sales.index,valid_sub,alpha=0.5,label='检验集残差')
plt.xlabel("index")
plt.ylabel("残差")
plt.title("检验集的残差散点图")

