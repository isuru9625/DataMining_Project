#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


train_x_csv = pd.read_csv('training_set_features.csv')


# In[3]:


train_y_csv = pd.read_csv('training_set_labels.csv')


# In[4]:


test_csv=pd.read_csv('test_set_features.csv')


# In[5]:


train_x_csv.columns


# In[6]:


def show_null_count(csv):
    idx = csv.isnull().sum()
    idx = idx[idx>0]
    idx.sort_values(inplace=True)
    idx.plot.bar()


# In[7]:


def get_corr(col, csv):
    corr = csv.corr()[col]
    idx_gt0 = corr[corr>0].sort_values(ascending=False).index.tolist()
    return corr[idx_gt0]


# In[8]:


show_null_count(train_x_csv)


# In[9]:


sns.heatmap(train_x_csv.corr(), vmax=.8, square=True)


# In[10]:


print(train_x_csv.head(n=10))


# In[11]:


train_x_csv.describe()


# In[12]:


train_x_csv.isnull().sum()


# In[13]:


from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imp.fit(train_x_csv[['h1n1_concern']])
train_x_csv[['h1n1_concern']]=imp.transform(train_x_csv[['h1n1_concern']])


# In[14]:


train_x_csv.isnull().sum()


# In[15]:


get_ipython().run_line_magic('pylab', 'inline')
sns.pairplot(train_x_csv[['h1n1_concern']].dropna())


# In[16]:


print(get_corr('h1n1_knowledge', train_x_csv))


# In[17]:


sns.pairplot(train_x_csv[['h1n1_knowledge']].dropna())


# In[18]:


imp.fit(train_x_csv[['h1n1_knowledge']])
train_x_csv[['h1n1_knowledge']]=imp.transform(train_x_csv[['h1n1_knowledge']])


# In[19]:


train_x_csv.isnull().sum()


# In[20]:


sns.pairplot(train_x_csv[['behavioral_antiviral_meds']].dropna())


# In[21]:


imp.fit(train_x_csv[['behavioral_antiviral_meds']])
train_x_csv[['behavioral_antiviral_meds']]=imp.transform(train_x_csv[['behavioral_antiviral_meds']])


# In[22]:


train_x_csv.isnull().sum()


# In[23]:


sns.pairplot(train_x_csv[['behavioral_avoidance']].dropna())


# In[24]:


imp.fit(train_x_csv[['behavioral_avoidance']])
train_x_csv[['behavioral_avoidance']]=imp.transform(train_x_csv[['behavioral_avoidance']])


# In[25]:


train_x_csv.isnull().sum()


# In[26]:


sns.pairplot(train_x_csv[['behavioral_face_mask']].dropna())


# In[27]:


print(get_corr('behavioral_face_mask', train_x_csv))


# In[28]:


imp.fit(train_x_csv[['behavioral_face_mask']])
train_x_csv[['behavioral_face_mask']]=imp.transform(train_x_csv[['behavioral_face_mask']])


# In[29]:


train_x_csv.isnull().sum()


# In[30]:


print(get_corr('behavioral_wash_hands', train_x_csv))


# In[31]:


sns.pairplot(train_x_csv[['behavioral_wash_hands']].dropna())


# In[32]:


imp.fit(train_x_csv[['behavioral_wash_hands']])
train_x_csv[['behavioral_wash_hands']]=imp.transform(train_x_csv[['behavioral_wash_hands']])


# In[33]:


train_x_csv.isnull().sum()


# In[34]:


print(get_corr('behavioral_large_gatherings', train_x_csv))


# In[35]:


sns.pairplot(train_x_csv[['behavioral_large_gatherings']].dropna())


# In[36]:


imp.fit(train_x_csv[['behavioral_large_gatherings']])
train_x_csv[['behavioral_large_gatherings']]=imp.transform(train_x_csv[['behavioral_large_gatherings']])


# In[37]:


train_x_csv.isnull().sum()


# In[38]:


sns.pairplot(train_x_csv[['behavioral_outside_home']].dropna())


# In[39]:


imp.fit(train_x_csv[['behavioral_outside_home']])
train_x_csv[['behavioral_outside_home']]=imp.transform(train_x_csv[['behavioral_outside_home']])


# In[40]:


train_x_csv.isnull().sum()


# In[41]:


sns.pairplot(train_x_csv[['behavioral_touch_face']].dropna())


# In[42]:


imp.fit(train_x_csv[['behavioral_touch_face']])
train_x_csv[['behavioral_touch_face']]=imp.transform(train_x_csv[['behavioral_touch_face']])


# In[43]:


train_x_csv.isnull().sum()


# In[44]:


sns.pairplot(train_x_csv[['doctor_recc_h1n1']].dropna())


# In[45]:


imp.fit(train_x_csv[['doctor_recc_h1n1']])
train_x_csv[['doctor_recc_h1n1']]=imp.transform(train_x_csv[['doctor_recc_h1n1']])


# In[46]:


train_x_csv.isnull().sum()


# In[47]:


sns.pairplot(train_x_csv[['doctor_recc_seasonal']].dropna())


# In[48]:


print(get_corr('doctor_recc_seasonal', train_x_csv))


# In[49]:


imp.fit(train_x_csv[['doctor_recc_seasonal']])
train_x_csv[['doctor_recc_seasonal']]=imp.transform(train_x_csv[['doctor_recc_seasonal']])


# In[50]:


train_x_csv.isnull().sum()


# In[51]:


sns.pairplot(train_x_csv[['chronic_med_condition']].dropna())


# In[52]:


imp.fit(train_x_csv[['chronic_med_condition']])
train_x_csv[['chronic_med_condition']]=imp.transform(train_x_csv[['chronic_med_condition']])


# In[53]:


train_x_csv.isnull().sum()


# In[54]:


sns.pairplot(train_x_csv[['child_under_6_months']].dropna())


# In[55]:


imp.fit(train_x_csv[['child_under_6_months']])
train_x_csv[['child_under_6_months']]=imp.transform(train_x_csv[['child_under_6_months']])


# In[56]:


train_x_csv.isnull().sum()


# In[57]:


sns.pairplot(train_x_csv[['health_worker']].dropna())


# In[58]:


imp.fit(train_x_csv[['health_worker']])
train_x_csv[['health_worker']]=imp.transform(train_x_csv[['health_worker']])


# In[59]:


train_x_csv.isnull().sum()


# In[60]:


sns.pairplot(train_x_csv[['health_insurance']].dropna())


# In[61]:


imp.fit(train_x_csv[['health_insurance']])
train_x_csv[['health_insurance']]=imp.transform(train_x_csv[['health_insurance']])


# In[62]:


train_x_csv.isnull().sum()


# In[63]:


sns.pairplot(train_x_csv[['opinion_h1n1_vacc_effective']].dropna())


# In[64]:


imp.fit(train_x_csv[['opinion_h1n1_vacc_effective']])
train_x_csv[['opinion_h1n1_vacc_effective']]=imp.transform(train_x_csv[['opinion_h1n1_vacc_effective']])


# In[65]:


train_x_csv.isnull().sum()


# In[66]:


sns.pairplot(train_x_csv[['opinion_h1n1_risk']].dropna())


# In[67]:


imp.fit(train_x_csv[['opinion_h1n1_risk']])
train_x_csv[['opinion_h1n1_risk']]=imp.transform(train_x_csv[['opinion_h1n1_risk']])


# In[68]:


train_x_csv.isnull().sum()


# In[69]:


sns.pairplot(train_x_csv[['opinion_h1n1_sick_from_vacc']].dropna())


# In[70]:


imp.fit(train_x_csv[['opinion_h1n1_sick_from_vacc']])
train_x_csv[['opinion_h1n1_sick_from_vacc']]=imp.transform(train_x_csv[['opinion_h1n1_sick_from_vacc']])


# In[71]:


train_x_csv.isnull().sum()


# In[72]:


sns.pairplot(train_x_csv[['opinion_seas_vacc_effective']].dropna())


# In[73]:


imp.fit(train_x_csv[['opinion_seas_vacc_effective']])
train_x_csv[['opinion_seas_vacc_effective']]=imp.transform(train_x_csv[['opinion_seas_vacc_effective']])


# In[74]:


train_x_csv.isnull().sum()


# In[75]:


sns.pairplot(train_x_csv[['opinion_seas_risk']].dropna())


# In[76]:


imp.fit(train_x_csv[['opinion_seas_risk']])
train_x_csv[['opinion_seas_risk']]=imp.transform(train_x_csv[['opinion_seas_risk']])


# In[77]:


train_x_csv.isnull().sum()


# In[78]:


sns.pairplot(train_x_csv[['opinion_seas_sick_from_vacc']].dropna())


# In[79]:


imp.fit(train_x_csv[['opinion_seas_sick_from_vacc']])
train_x_csv[['opinion_seas_sick_from_vacc']]=imp.transform(train_x_csv[['opinion_seas_sick_from_vacc']])


# In[80]:


train_x_csv.isnull().sum()


# In[81]:


train_x_csv['age_group'].value_counts()


# In[82]:


train_x_csv['education'].value_counts()


# In[83]:


train_x_csv['race'].value_counts()


# In[84]:


train_x_csv['sex'].value_counts()


# In[85]:


train_x_csv['income_poverty'].value_counts()


# In[86]:


train_x_csv['marital_status'].value_counts()


# In[87]:


train_x_csv['rent_or_own'].value_counts()


# In[88]:


train_x_csv['employment_status'].value_counts()


# In[89]:


train_x_csv['hhs_geo_region'].value_counts()


# In[90]:


train_x_csv['census_msa'].value_counts()


# In[91]:


train_x_csv['employment_industry'].value_counts()


# In[92]:


train_x_csv['employment_occupation'].value_counts()


# In[93]:


def convert_columns_to_numeric(csv):
        replace_num={"age_group":{"18 - 34 Years":0,"35 - 44 Years":1,"45 - 54 Years":2,"55 - 64 Years":3,"65+ Years":4},
    "education":{"< 12 Years":0,"12 Years":1,"Some College":2,"College Graduate":3},
            "race":{"White":0,"Black":1,"Hispanic":2,"Other or Multiple":3},
            "sex":{"Female":0,"Male":1},
             "income_poverty":{"Below Poverty":0,"<= $75,000, Above Poverty":1,"> $75,000":2},
             "marital_status":{"Married":0,"Not Married":1},
             "rent_or_own":{"Own":0,"Rent":1},
             "employment_status":{"Employed":0,"Not in Labor Force":1,"Unemployed":2},
             "hhs_geo_region":{"lzgpxyit":0,"fpwskwrf":1,"qufhixun":2,"oxchjgsf":3,"kbazzjca":4,"bhuqouqj":5,"mlyzmhmf":6,"lrircsnp":7,"atmpeygn":8,"dqpwygqj":9},
             "census_msa":{"MSA, Not Principle  City":0,"MSA, Principle City":1,"Non-MSA":2},
             "employment_industry":{"fcxhlnwr":0,"wxleyezf":1,"ldnlellj":2,"pxcmvdjn":3,"atmlpfrs":4,"arjwrbjb":5,"xicduogh":6,"mfikgejo":7,"vjjrobsf":8,"rucpziij":9,"xqicxuve":10,"saaquncn":11,"cfqqtusy":12,"nduyfdeo":13,"mcubkhph":14,"wlfvacwt":15,"dotnnunm":16,"haxffmxo":17,"msuufmds":18,"phxvnwax":19,"qnlwzans":20},
             "employment_occupation":{"xtkaffoo":0,"mxkfnird":1,"emcorrxb":2,"cmhcxjea":3,"xgwztkwe":4,"hfxkjkmi":5,"qxajmpny":6,"xqwwgdyp":7,"kldqjyjy":8,"uqqtjvyb":9,"tfqavkke":10,"ukymxvdu":11,"vlluhbov":12,"oijqvulv":13,"ccgxvspp":14,"bxpfxfdn":15,"haliazsg":16,"rcertsgn":17,"xzmlyyjv":18,"dlvbwzss":19,"hodpvpew":20,"dcjcmpih":21,"pvmttkik":22}
            }
        csv.replace(replace_num, inplace=True)
    


# In[94]:


convert_columns_to_numeric(train_x_csv)


# In[95]:


train_x_csv.head(n=10)


# In[96]:


train_x_csv.isnull().sum()


# In[97]:


sns.pairplot(train_x_csv[['education']].dropna())


# In[98]:


imp.fit(train_x_csv[['education']])
train_x_csv[['education']]=imp.transform(train_x_csv[['education']])


# In[99]:


train_x_csv.isnull().sum()


# In[100]:


sns.pairplot(train_x_csv[['income_poverty']].dropna())


# In[101]:


imp.fit(train_x_csv[['income_poverty']])
train_x_csv[['income_poverty']]=imp.transform(train_x_csv[['income_poverty']])


# In[102]:


train_x_csv.isnull().sum()


# In[103]:


sns.pairplot(train_x_csv[['marital_status']].dropna())


# In[104]:


imp.fit(train_x_csv[['marital_status']])
train_x_csv[['marital_status']]=imp.transform(train_x_csv[['marital_status']])


# In[105]:


train_x_csv.isnull().sum()


# In[106]:


sns.pairplot(train_x_csv[['rent_or_own']].dropna())


# In[107]:


imp.fit(train_x_csv[['rent_or_own']])
train_x_csv[['rent_or_own']]=imp.transform(train_x_csv[['rent_or_own']])


# In[108]:


train_x_csv.isnull().sum()


# In[109]:


sns.pairplot(train_x_csv[['employment_status']].dropna())


# In[110]:


imp.fit(train_x_csv[['employment_status']])
train_x_csv[['employment_status']]=imp.transform(train_x_csv[['employment_status']])


# In[111]:


train_x_csv.isnull().sum()


# In[112]:


sns.pairplot(train_x_csv[['household_adults']].dropna())


# In[113]:


imp.fit(train_x_csv[['household_adults']])
train_x_csv[['household_adults']]=imp.transform(train_x_csv[['household_adults']])


# In[114]:


train_x_csv.isnull().sum()


# In[115]:


sns.pairplot(train_x_csv[['household_children']].dropna())


# In[116]:


imp.fit(train_x_csv[['household_children']])
train_x_csv[['household_children']]=imp.transform(train_x_csv[['household_children']])


# In[117]:


train_x_csv.isnull().sum()


# In[118]:


sns.pairplot(train_x_csv[['employment_industry']].dropna())


# In[119]:


imp.fit(train_x_csv[['employment_industry']])
train_x_csv[['employment_industry']]=imp.transform(train_x_csv[['employment_industry']])


# In[120]:


train_x_csv.isnull().sum()


# In[121]:


sns.pairplot(train_x_csv[['employment_occupation']].dropna())


# In[122]:


imp.fit(train_x_csv[['employment_occupation']])
train_x_csv[['employment_occupation']]=imp.transform(train_x_csv[['employment_occupation']])


# In[123]:


train_x_csv.isnull().sum()


# In[124]:


print(train_x_csv.head(n=15))


# In[125]:


train_x_csv=train_x_csv.astype(int)
train_x_csv.head(n=5)


# In[126]:


train_x_csv.drop(['respondent_id'],axis=1,inplace=True)


# In[127]:


sns.heatmap(train_x_csv.corr(),annot=True,cmap='RdYlGn',linewidths=0.2)
fig=plt.gcf()
fig.set_size_inches(20,12)
plt.show()


# In[128]:


train_y_csv.head(n=5)


# In[142]:


train_csv1=train_x_csv.copy()
train_csv2=train_x_csv.copy()


# In[143]:


train_csv1["h1n1_vaccine"]=train_y_csv["h1n1_vaccine"]


# In[144]:


train_csv2["seasonal_vaccine"]=train_y_csv["seasonal_vaccine"]
train_csv2.drop(['h1n1_concern','h1n1_knowledge','doctor_recc_h1n1'],axis=1,inplace=True)


# In[145]:


train_csv2.head(n=5)


# In[146]:


from sklearn.model_selection import train_test_split
x1_train,x1_test,y1_train,y1_test = train_test_split(train_csv1.drop('h1n1_vaccine', axis=1),
                                                 train_csv1['h1n1_vaccine'],
                                                 test_size=0.2,
                                                 random_state=123)


# In[147]:


from sklearn.model_selection import train_test_split
x2_train,x2_test,y2_train,y2_test = train_test_split(train_csv2.drop('seasonal_vaccine', axis=1),
                                                 train_csv2['seasonal_vaccine'],
                                                 test_size=0.2,
                                                 random_state=123)


# In[148]:


from sklearn.metrics import accuracy_score


# In[149]:


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD


# In[150]:


def baselineNN(dims):
    model = Sequential()
    model.add(Dense(10, input_dim=dims, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    #model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# In[151]:


def use_keras_nn_model(x, y, xx, yy, epochs):
    model = baselineNN(x.shape[1])
    model.fit(x.as_matrix(), y.as_matrix(), epochs=epochs)
    y_pred = model.predict(xx.as_matrix()).reshape(xx.shape[0],)
    return y_pred, model


# In[153]:


y1_pred_nn, model_nn1 = use_keras_nn_model(x1_train, y1_train, x1_test, y1_test, 500)


# In[154]:


y2_pred_nn, model_nn2 = use_keras_nn_model(x2_train, y2_train, x2_test, y2_test, 500)


# In[155]:


import xgboost as xgb
from xgboost import plot_importance


# In[156]:


params = {
    'objective': 'binary:logistic',
    'gamma': 0.1,
    'max_depth': 5,
    'lambda': 3,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'min_child_weight': 3,
    'silent': 1,
    'eta': 0.1,
    'seed': 1000,
    'nthread': 4,
}

num_round = 10


# In[157]:


d1train = xgb.DMatrix(x1_train, label=y1_train)
d1test = xgb.DMatrix(x1_test, label=y1_test)
watchlist = [(d1train, 'train'), (d1test, 'test')]
bst = xgb.train(params, d1train, num_round, watchlist)
y1_pred_xgb = bst.predict(d1test)


# In[158]:


plot_importance(bst)


# In[159]:


train_csv1.isnull().sum()


# In[160]:


test_csv.isnull().sum()


# In[161]:


imp.fit(test_csv[['h1n1_concern']])
test_csv[['h1n1_concern']]=imp.transform(test_csv[['h1n1_concern']])


# In[162]:


imp.fit(test_csv[['h1n1_knowledge']])
test_csv[['h1n1_knowledge']]=imp.transform(test_csv[['h1n1_knowledge']])


# In[163]:


imp.fit(test_csv[['behavioral_antiviral_meds']])
test_csv[['behavioral_antiviral_meds']]=imp.transform(test_csv[['h1n1_knowledge']])


# In[164]:


imp.fit(test_csv[['behavioral_avoidance']])
test_csv[['behavioral_avoidance']]=imp.transform(test_csv[['behavioral_avoidance']])


# In[165]:


imp.fit(test_csv[['behavioral_face_mask']])
test_csv[['behavioral_face_mask']]=imp.transform(test_csv[['behavioral_face_mask']])


# In[166]:


imp.fit(test_csv[['behavioral_wash_hands']])
test_csv[['behavioral_wash_hands']]=imp.transform(test_csv[['behavioral_wash_hands']])


# In[167]:


imp.fit(test_csv[['behavioral_large_gatherings']])
test_csv[['behavioral_large_gatherings']]=imp.transform(test_csv[['behavioral_large_gatherings']])


# In[168]:


imp.fit(test_csv[['behavioral_outside_home']])
test_csv[['behavioral_outside_home']]=imp.transform(test_csv[['behavioral_outside_home']])


# In[169]:


imp.fit(test_csv[['behavioral_touch_face']])
test_csv[['behavioral_touch_face']]=imp.transform(test_csv[['behavioral_touch_face']])


# In[170]:


imp.fit(test_csv[['doctor_recc_h1n1']])
test_csv[['doctor_recc_h1n1']]=imp.transform(test_csv[['doctor_recc_h1n1']])


# In[171]:


imp.fit(test_csv[['doctor_recc_seasonal']])
test_csv[['doctor_recc_seasonal']]=imp.transform(test_csv[['doctor_recc_seasonal']])


# In[172]:


imp.fit(test_csv[['chronic_med_condition']])
test_csv[['chronic_med_condition']]=imp.transform(test_csv[['chronic_med_condition']])


# In[173]:


imp.fit(test_csv[['child_under_6_months']])
test_csv[['child_under_6_months']]=imp.transform(test_csv[['child_under_6_months']])


# In[174]:


imp.fit(test_csv[['health_worker']])
test_csv[['health_worker']]=imp.transform(test_csv[['health_worker']])


# In[175]:


imp.fit(test_csv[['health_insurance']])
test_csv[['health_insurance']]=imp.transform(test_csv[['health_insurance']])


# In[176]:


imp.fit(test_csv[['opinion_h1n1_vacc_effective']])
test_csv[['opinion_h1n1_vacc_effective']]=imp.transform(test_csv[['opinion_h1n1_vacc_effective']])


# In[177]:


imp.fit(test_csv[['opinion_h1n1_risk']])
test_csv[['opinion_h1n1_risk']]=imp.transform(test_csv[['opinion_h1n1_risk']])


# In[178]:


imp.fit(test_csv[['opinion_h1n1_sick_from_vacc']])
test_csv[['opinion_h1n1_sick_from_vacc']]=imp.transform(test_csv[['opinion_h1n1_sick_from_vacc']])


# In[179]:


imp.fit(test_csv[['opinion_seas_vacc_effective']])
test_csv[['opinion_seas_vacc_effective']]=imp.transform(test_csv[['opinion_seas_vacc_effective']])


# In[180]:


imp.fit(test_csv[['opinion_seas_risk']])
test_csv[['opinion_seas_risk']]=imp.transform(test_csv[['opinion_seas_risk']])


# In[181]:


imp.fit(test_csv[['opinion_seas_sick_from_vacc']])
test_csv[['opinion_seas_sick_from_vacc']]=imp.transform(test_csv[['opinion_seas_sick_from_vacc']])


# In[182]:


imp.fit(test_csv[['education']])
test_csv[['education']]=imp.transform(test_csv[['education']])


# In[183]:


imp.fit(test_csv[['income_poverty']])
test_csv[['income_poverty']]=imp.transform(test_csv[['income_poverty']])


# In[184]:


imp.fit(test_csv[['marital_status']])
test_csv[['marital_status']]=imp.transform(test_csv[['marital_status']])


# In[185]:


imp.fit(test_csv[['rent_or_own']])
test_csv[['rent_or_own']]=imp.transform(test_csv[['rent_or_own']])


# In[186]:


imp.fit(test_csv[['employment_status']])
test_csv[['employment_status']]=imp.transform(test_csv[['employment_status']])


# In[187]:


imp.fit(test_csv[['household_adults']])
test_csv[['household_adults']]=imp.transform(test_csv[['household_adults']])


# In[188]:


imp.fit(test_csv[['household_children']])
test_csv[['household_children']]=imp.transform(test_csv[['household_children']])


# In[189]:


imp.fit(test_csv[['employment_industry']])
test_csv[['employment_industry']]=imp.transform(test_csv[['employment_industry']])


# In[190]:


imp.fit(test_csv[['employment_occupation']])
test_csv[['employment_occupation']]=imp.transform(test_csv[['employment_occupation']])


# In[191]:


test_csv.isnull().sum()


# In[192]:


convert_columns_to_numeric(test_csv)


# In[193]:


final_df=test_csv.copy()
final_df.drop(['respondent_id'],axis=1,inplace=True)


# In[194]:


final_df=final_df.astype(int)


# In[195]:


final_df.head(n=5)


# In[197]:


len(x1_train.columns)


# In[199]:


final_df.columns


# In[200]:


y1_final_prob = model_nn1.predict(final_df.as_matrix()).reshape(final_df.shape[0],)


# In[201]:


print(y1_final_prob)


# In[202]:


final_df.drop(['h1n1_concern','h1n1_knowledge','doctor_recc_h1n1'],axis=1,inplace=True)


# In[203]:


y2_final_prob = model_nn2.predict(final_df.as_matrix()).reshape(final_df.shape[0],)


# In[204]:


#y2_final = throttling(y2_final_prob, .6)


# In[205]:


submission = pd.concat([test_csv['respondent_id'], pd.DataFrame(y1_final_prob), pd.DataFrame(y2_final_prob)], axis=1)
submission.columns = ['tripid', 'h1n1_vaccine',"seasonal_vaccine"]


# In[207]:


submission.to_csv('submission.csv', encoding='utf-8', index = False)


# In[ ]:





# In[ ]:




