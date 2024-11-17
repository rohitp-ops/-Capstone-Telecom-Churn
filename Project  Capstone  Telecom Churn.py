#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install pandas')


# In[2]:


get_ipython().system('pip install Imblearn import imblearn')


# In[169]:


get_ipython().system('pip install imblearn --ignore-installed scikit-learn')


# In[170]:


get_ipython().system('pip install numpy')


# In[171]:


get_ipython().system('pip install matplotlib')


# In[6]:


get_ipython().run_line_magic('pip', 'install seaborn')


# In[7]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# In[8]:


import pandas as pd
import numpy as np


# In[9]:


from sklearn.decomposition import PCA


# In[10]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x) 
import time


# In[11]:


import matplotlib.pyplot as plt


# In[12]:


from sklearn.svm import SVC


# In[13]:


from sklearn import metrics
from sklearn.metrics import confusion_matrix


# In[14]:


import seaborn as sns
plt.figure(figsize=(10,6))


# In[15]:


import matplotlib.pyplot as plt


# In[16]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


# In[17]:


import warnings
warnings.filterwarnings('ignore')


# In[18]:


import pandas as pd, numpy as np


# In[19]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


# In[20]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[21]:


pd.set_option('display.max_columns', 500)


# # Loading Data

# In[22]:


df = pd.read_csv("telecom_churn_data.csv")


# In[23]:


df.head()


# In[24]:


df.info()


# In[25]:


df.isnull().sum()


# In[26]:


df.shape


# In[27]:


df.head(2)


# In[28]:


df.isnull().sum()


# In[29]:


df.isnull().sum()


# # Clean Data

# In[30]:


import pandas as pd


# In[31]:


missing_values = df.isnull().sum()
print("Missing Values:\n", missing_values[missing_values > 0])


# In[32]:


threshold = 0.5 * len(df)
df = df.dropna(thresh=threshold, axis=1)


# In[33]:


for column in df.columns:
    if df[column].dtype == 'object':
        df[column].fillna(df[column].mode()[0], inplace=True)
    else:
        df[column].fillna(df[column].mean(), inplace=True)


# In[34]:


df = df.drop_duplicates()


# In[35]:


df = pd.get_dummies(df, drop_first=True)


# In[36]:


numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])


# In[37]:


df.isnull().sum().sum()


# In[38]:


print("Shape of cleaned dataset:", df.shape)


# In[39]:


df.head


# In[40]:


df.shape


# # Inspecting the Dataframe

# In[41]:


df.shape


# In[42]:


df.info()


# In[43]:


df.describe()


# In[44]:


df_missing_columns = (round(((df.isnull().sum()/len(df.index))*100),2).to_frame('null')).sort_values('null', ascending=False)
df_missing_columns


# In[45]:


col_list_missing_30 = list(df_missing_columns.index[df_missing_columns['null'] > 30])


# In[46]:


df = df.drop(col_list_missing_30, axis=1)


# In[47]:


df.shape


# In[48]:


df.hist(bins=20, figsize=(15, 10))
plt.show()


# In[49]:


correlation_matrix = df.corr()


# In[50]:


plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Matrix")
plt.show()


# In[51]:


for col in df.select_dtypes(include=['object']).columns:
    print(f"Value counts for {col}:\n", df[col].value_counts())
    print("\n")


# # Deleting the date columns as the date columns are not required in our analysis

# In[52]:


date_cols = [k for k in df.columns.to_list() if 'date' in k]
print(date_cols) 


# In[53]:


df = df.drop(date_cols, axis=1)


# In[54]:


df = df.drop('circle_id', axis=1)


# In[55]:


df.shape


# # Filter high-value customers

# In[56]:


df['avg_rech_amt_6_7'] = (df['total_rech_amt_6'] + df['total_rech_amt_7'])/2


# In[57]:


X = df['avg_rech_amt_6_7'].quantile(0.7)
X


# In[58]:


df = df[df['avg_rech_amt_6_7'] >= X]
df.head()


# In[59]:


df.shape


# # Handling missing values in rows
# 

# In[60]:


df_missing_rows_50 = df[(df.isnull().sum(axis=1)) > (len(df.columns)//2)]
df_missing_rows_50.shape


# In[61]:


df = df.drop(df_missing_rows_50.index)
df.shape


# In[62]:


df_missing_columns = (round(((df.isnull().sum()/len(df.index))*100),2).to_frame('null')).sort_values('null', ascending=False)
df_missing_columns


# In[63]:


print(((df_missing_columns[df_missing_columns['null'] == 5.32]).index).to_list())


# In[64]:


df_null_mou_9 = df[(df['loc_og_t2m_mou_9'].isnull()) & (df['loc_ic_t2f_mou_9'].isnull()) & (df['roam_og_mou_9'].isnull()) & (df['std_ic_t2m_mou_9'].isnull()) &
  (df['loc_og_t2t_mou_9'].isnull()) & (df['std_ic_t2t_mou_9'].isnull()) & (df['loc_og_t2f_mou_9'].isnull()) & (df['loc_ic_mou_9'].isnull()) &
  (df['loc_og_t2c_mou_9'].isnull()) & (df['loc_og_mou_9'].isnull()) & (df['std_og_t2t_mou_9'].isnull()) & (df['roam_ic_mou_9'].isnull()) &
  (df['loc_ic_t2m_mou_9'].isnull()) & (df['std_og_t2m_mou_9'].isnull()) & (df['loc_ic_t2t_mou_9'].isnull()) & (df['std_og_t2f_mou_9'].isnull()) & 
  (df['std_og_t2c_mou_9'].isnull()) & (df['og_others_9'].isnull()) & (df['std_og_mou_9'].isnull()) & (df['spl_og_mou_9'].isnull()) & 
  (df['std_ic_t2f_mou_9'].isnull()) & (df['isd_og_mou_9'].isnull()) & (df['std_ic_mou_9'].isnull()) & (df['offnet_mou_9'].isnull()) & 
  (df['isd_ic_mou_9'].isnull()) & (df['ic_others_9'].isnull()) & (df['std_ic_t2o_mou_9'].isnull()) & (df['onnet_mou_9'].isnull()) & 
  (df['spl_ic_mou_9'].isnull())]

df_null_mou_9.head()


# In[65]:


df_null_mou_9.shape


# In[66]:


df = df.drop(df_null_mou_9.index)


# In[67]:


df_missing_columns = (round(((df.isnull().sum()/len(df.index))*100),2).to_frame('null')).sort_values('null', ascending=False)
df_missing_columns


# In[68]:


print(((df_missing_columns[df_missing_columns['null'] == 0.55]).index).to_list())


# In[69]:


df_null_mou_8 = df[(df['loc_og_t2m_mou_8'].isnull()) & (df['loc_ic_t2f_mou_8'].isnull()) & (df['roam_og_mou_8'].isnull()) & (df['std_ic_t2m_mou_8'].isnull()) &
  (df['loc_og_t2t_mou_8'].isnull()) & (df['std_ic_t2t_mou_8'].isnull()) & (df['loc_og_t2f_mou_8'].isnull()) & (df['loc_ic_mou_8'].isnull()) &
  (df['loc_og_t2c_mou_8'].isnull()) & (df['loc_og_mou_8'].isnull()) & (df['std_og_t2t_mou_8'].isnull()) & (df['roam_ic_mou_8'].isnull()) &
  (df['loc_ic_t2m_mou_8'].isnull()) & (df['std_og_t2m_mou_8'].isnull()) & (df['loc_ic_t2t_mou_8'].isnull()) & (df['std_og_t2f_mou_8'].isnull()) & 
  (df['std_og_t2c_mou_8'].isnull()) & (df['og_others_8'].isnull()) & (df['std_og_mou_8'].isnull()) & (df['spl_og_mou_8'].isnull()) & 
  (df['std_ic_t2f_mou_8'].isnull()) & (df['isd_og_mou_8'].isnull()) & (df['std_ic_mou_8'].isnull()) & (df['offnet_mou_8'].isnull()) & 
  (df['isd_ic_mou_8'].isnull()) & (df['ic_others_8'].isnull()) & (df['std_ic_t2o_mou_8'].isnull()) & (df['onnet_mou_8'].isnull()) & 
  (df['spl_ic_mou_8'].isnull())]

df_null_mou_8.head()


# In[70]:


df = df.drop(df_null_mou_8.index)


# In[71]:


df_missing_columns = (round(((df.isnull().sum()/len(df.index))*100),2).to_frame('null')).sort_values('null', ascending=False)
df_missing_columns


# In[72]:


print(((df_missing_columns[df_missing_columns['null'] == 0.44]).index).to_list())


# In[73]:


df_null_mou_6 = df[(df['loc_og_t2m_mou_6'].isnull()) & (df['loc_ic_t2f_mou_6'].isnull()) & (df['roam_og_mou_6'].isnull()) & (df['std_ic_t2m_mou_6'].isnull()) &
  (df['loc_og_t2t_mou_6'].isnull()) & (df['std_ic_t2t_mou_6'].isnull()) & (df['loc_og_t2f_mou_6'].isnull()) & (df['loc_ic_mou_6'].isnull()) &
  (df['loc_og_t2c_mou_6'].isnull()) & (df['loc_og_mou_6'].isnull()) & (df['std_og_t2t_mou_6'].isnull()) & (df['roam_ic_mou_6'].isnull()) &
  (df['loc_ic_t2m_mou_6'].isnull()) & (df['std_og_t2m_mou_6'].isnull()) & (df['loc_ic_t2t_mou_6'].isnull()) & (df['std_og_t2f_mou_6'].isnull()) & 
  (df['std_og_t2c_mou_6'].isnull()) & (df['og_others_6'].isnull()) & (df['std_og_mou_6'].isnull()) & (df['spl_og_mou_6'].isnull()) & 
  (df['std_ic_t2f_mou_6'].isnull()) & (df['isd_og_mou_6'].isnull()) & (df['std_ic_mou_6'].isnull()) & (df['offnet_mou_6'].isnull()) & 
  (df['isd_ic_mou_6'].isnull()) & (df['ic_others_6'].isnull()) & (df['std_ic_t2o_mou_6'].isnull()) & (df['onnet_mou_6'].isnull()) & 
  (df['spl_ic_mou_6'].isnull())]

df_null_mou_6.head()


# In[74]:


df = df.drop(df_null_mou_6.index)


# In[75]:


df_missing_columns = (round(((df.isnull().sum()/len(df.index))*100),2).to_frame('null')).sort_values('null', ascending=False)
df_missing_columns


# # Tag churners

# In[76]:


df['churn'] = np.where((df['total_ic_mou_9']==0) & (df['total_og_mou_9']==0) & (df['vol_2g_mb_9']==0) & (df['vol_3g_mb_9']==0), 1, 0)


# In[77]:


df.head()


# # Deleting all the attributes corresponding to the churn phase

# In[78]:


col_9 = [col for col in df.columns.to_list() if '_9' in col]
print(col_9)


# In[79]:


df = df.drop(col_9, axis=1)


# In[80]:


df = df.drop('sep_vbc_3g', axis=1)


# # Checking churn percentage

# In[81]:


round(100*(df['churn'].mean()),2)


# # Outliers treatment

# In[82]:


df['mobile_number'] = df['mobile_number'].astype(object)
df['churn'] = df['churn'].astype(object)


# In[83]:


df.info()


# In[84]:


numeric_cols = df.select_dtypes(exclude=['object']).columns
print(numeric_cols)


# In[85]:


for col in numeric_cols: 
    q1 = df[col].quantile(0.10)
    q3 = df[col].quantile(0.90)
    iqr = q3-q1
    range_low  = q1-1.5*iqr
    range_high = q3+1.5*iqr
    # Assigning the filtered dataset into data
    data = df.loc[(df[col] > range_low) & (df[col] < range_high)]

data.shape


# # Derive new features
# 

# In[86]:


data['total_mou_good'] = (data['total_og_mou_6'] + data['total_ic_mou_6'])


# In[87]:


data['avg_mou_action'] = (data['total_og_mou_7'] + data['total_og_mou_8'] + data['total_ic_mou_7'] + data['total_ic_mou_8'])/2


# In[88]:


data['diff_mou'] = data['avg_mou_action'] - data['total_mou_good']


# In[89]:


data['decrease_mou_action'] = np.where((data['diff_mou'] < 0), 1, 0)


# In[90]:


data.head()


# # Deriving new column decrease_rech_num_action

# In[91]:


data['avg_rech_num_action'] = (data['total_rech_num_7'] + data['total_rech_num_8'])/2


# In[92]:


data['diff_rech_num'] = data['avg_rech_num_action'] - data['total_rech_num_6']


# In[93]:


data['decrease_rech_num_action'] = np.where((data['diff_rech_num'] < 0), 1, 0)


# In[94]:


data.head()


# # Deriving new column decrease_rech_amt_action
# 

# In[95]:


data['avg_rech_amt_action'] = (data['total_rech_amt_7'] + data['total_rech_amt_8'])/2


# In[96]:


data['diff_rech_amt'] = data['avg_rech_amt_action'] - data['total_rech_amt_6']


# In[97]:


data['decrease_rech_amt_action'] = np.where((data['diff_rech_amt'] < 0), 1, 0) 


# In[98]:


data.head()


# # Deriving new column decrease_arpu_action

# In[99]:


data['avg_arpu_action'] = (data['arpu_7'] + data['arpu_8'])/2


# In[100]:


data['diff_arpu'] = data['avg_arpu_action'] - data['arpu_6']


# In[101]:


data['decrease_arpu_action'] = np.where(data['diff_arpu'] < 0, 1, 0)


# In[102]:


data.head()


# # Deriving new column decrease_vbc_action

# In[103]:


data['avg_vbc_3g_action'] = (data['jul_vbc_3g'] + data['aug_vbc_3g'])/2


# In[104]:


data['diff_vbc'] = data['avg_vbc_3g_action'] - data['jun_vbc_3g']


# In[105]:


data['decrease_vbc_action'] = np.where(data['diff_vbc'] < 0 , 1, 0)


# In[106]:


data.head()


# # Explority Data Analytics

# Univariate analysis

# In[107]:


data['churn'] = data['churn'].astype('int64')


# In[108]:


data.pivot_table(values='churn', index='decrease_mou_action', aggfunc='mean').plot.bar()
plt.ylabel('churn rate')
plt.show()


# Analysis

# In[109]:


data.pivot_table(values='churn', index='decrease_rech_num_action', aggfunc='mean').plot.bar()
plt.ylabel('churn rate')
plt.show()


# In[110]:


data.pivot_table(values='churn', index='decrease_rech_amt_action', aggfunc='mean').plot.bar()
plt.ylabel('churn rate')
plt.show()


# In[111]:


data.pivot_table(values='churn', index='decrease_vbc_action', aggfunc='mean').plot.bar()
plt.ylabel('churn rate')
plt.show()


# In[112]:


data_churn = data[data['churn'] == 1]
data_non_churn = data[data['churn'] == 0]


# In[113]:


ax = sns.distplot(data_churn['avg_arpu_action'],label='churn',hist=False)
ax = sns.distplot(data_non_churn['avg_arpu_action'],label='not churn',hist=False)
ax.set(xlabel='Action phase ARPU')


# In[114]:


ax = sns.distplot(data_churn['total_mou_good'],label='churn',hist=False)
ax = sns.distplot(data_non_churn['total_mou_good'],label='non churn',hist=False)
ax.set(xlabel='Action phase MOU')


# Bivariate analysis
# 

# In[115]:


data.pivot_table(values='churn', index='decrease_rech_amt_action', columns='decrease_rech_num_action', aggfunc='mean').plot.bar()
plt.ylabel('churn rate')
plt.show()


# In[116]:


data.pivot_table(values='churn', index='decrease_rech_amt_action', columns='decrease_vbc_action', aggfunc='mean').plot.bar()
plt.ylabel('churn rate')
plt.show()


# In[117]:


plt.figure(figsize=(10, 6))
ax = sns.scatterplot(x='avg_rech_num_action', y='avg_rech_amt_action', hue='churn', data=data)


# In[118]:


data = data.drop(['total_mou_good','avg_mou_action','diff_mou','avg_rech_num_action','diff_rech_num','avg_rech_amt_action',
                 'diff_rech_amt','avg_arpu_action','diff_arpu','avg_vbc_3g_action','diff_vbc','avg_rech_amt_6_7'], axis=1)


# Train-Test Split

# In[119]:


from sklearn.model_selection import train_test_split


# In[120]:


X = data.drop(['mobile_number','churn'], axis=1)


# In[121]:


y = data['churn']


# In[122]:


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=100)


# Dealing with data imbalance

# In[123]:


from imblearn.over_sampling import SMOTE


# In[124]:


sm = SMOTE(random_state=27)


# Feature Scaling
# 

# In[125]:


from sklearn.preprocessing import StandardScaler


# In[126]:


scaler = StandardScaler()


# In[127]:


X_train.head()


# # Modelling
# 

# # HYPERPARAMETER 

# In[128]:


param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10]
}


# In[129]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

rf = RandomForestClassifier(random_state=42)


# In[130]:


print(y_train.unique())
print(y_train.dtype)


# In[131]:


y_train = y_train.astype(int)
y_test = y_test.astype(int)


# In[132]:


grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)


# In[133]:


print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)


# # Model Evaluation

# In[134]:


from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()

model.fit(X_train, y_train)


# In[135]:


(model.fit)


# In[136]:


y_pred = model.predict(X_test)


# In[137]:


import statsmodels.api as sm
import matplotlib.pyplot as plt


# In[138]:


y_actual = data['churn'] 


# In[139]:


X = data.drop(columns=['churn'])  


# In[140]:


print(X.dtypes)


# In[141]:


X = pd.get_dummies(X, drop_first=True)  


# In[142]:


print(y_actual.shape, X.shape)


# In[143]:


data_cleaned = data.dropna(subset=['churn'])
data_cleaned = data.fillna(data.mean())


# In[144]:


y_actual = data_cleaned['churn']
X = data_cleaned.drop(columns=['churn'])
X = sm.add_constant(X)


model = sm.OLS(y_actual, X).fit()


sm.graphics.plot_leverage_resid2(model)
plt.show()


# In[145]:


print(X.isnull().sum()) 
print(y_actual.isnull().sum())  


# In[146]:


data_cleaned = data.dropna(subset=['churn']) 


# In[147]:


X = X.fillna(X.mean())  
y_actual = y_actual.fillna(y_actual.mean()) 


# In[148]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


# In[149]:


print("Accuracy:", accuracy_score(y_test, y_pred))


# In[150]:


print("Precision:", precision_score(y_test, y_pred, average='binary'))
print("Recall:", recall_score(y_test, y_pred, average='binary'))
print("F1-Score:", f1_score(y_test, y_pred, average='binary'))


print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


print("Classification Report:\n", classification_report(y_test, y_pred))


# In[151]:


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


# In[152]:


plt.figure(figsize=(10, 6))


# In[153]:


plt.plot([0, 1], [0, 1], color='gray', linestyle='--') 


# In[154]:


plt.xlabel("False Positive Rate")


# In[155]:


plt.ylabel("True Positive Rate")


# In[156]:


plt.title("Receiver Operating Characteristic (ROC) Curve")


# In[157]:


plt.legend(loc='lower right')
plt.show()


# In[158]:


(y_pred)


# In[159]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[160]:


residuals =  y_pred


# In[161]:


plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_pred, y=residuals)
plt.axhline(0, color='red', linestyle='--')  # Add a horizontal line at y = 0
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted Values')
plt.show()


# In[162]:


plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True)
plt.xlabel('Residuals')
plt.title('Histogram of Residuals')
plt.show()


# In[163]:


import scipy.stats as stats

plt.figure(figsize=(10, 6))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Q-Q Plot of Residuals')
plt.show()


# In[164]:


from statsmodels.stats.stattools import durbin_watson

dw_stat = durbin_watson(residuals)
print(f'Durbin-Watson statistic: {dw_stat}')


# In[165]:


import statsmodels.api as sm
sm.qqplot(residuals, line ='45')
plt.title('QQ Plot')
plt.show()


# In[166]:


sm.graphics.plot_leverage_resid2(model)
plt.show()


# In[167]:


plt.figure(figsize=(8,6))
sns.scatterplot(x=y_pred, y=residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted Values Plot')
plt.show()


sm.qqplot(residuals, line ='45')
plt.title('QQ Plot')
plt.show()


sm.graphics.plot_leverage_resid2(model)
plt.show()


# # Conclusion and Recommendations for the Model

# . Model Performance
# Based on the evaluation of the residual analysis and other metrics, here's a summary of the model's performance:
# 
# Residuals Analysis:
# 
# If the Residuals vs Fitted Values Plot shows random scatter, the model is appropriate for the data. If there is a clear pattern, the model might need to be refined (e.g., by considering non-linear relationships).
# The QQ Plot helps assess the normality of residuals. If the residuals follow a normal distribution (i.e., the points roughly align with the 45-degree line), the assumptions of normality are met.
# The Leverage vs Residuals Plot helps detect outliers or influential points. If high-leverage points show large residuals, it could indicate that those data points disproportionately influence the model and might need special treatment (e.g., removal or further investigation).
# Model Accuracy:
# 
# Based on metrics like accuracy, precision, recall, and AUC (Area Under the Curve), the model might perform well for predicting the target variable (e.g., churn). High AUC and low residuals generally suggest a good model fit.
# If the AUC is not high, it may indicate that the model has limited discriminative power and might need further tuning or a more complex algorithm.
# 2. Key Findings
# If the model’s residuals exhibit any pattern or heteroscedasticity (i.e., varying variance of residuals across fitted values), it suggests that a linear regression model might not be the best choice. You might consider alternative approaches such as:
# 
# Non-linear models (e.g., Random Forest, Gradient Boosting) for capturing non-linear patterns.
# Transformations of the dependent or independent variables to stabilize variance or address skewness.
# Leverage and Outliers: The leverage vs residuals plot might highlight influential points or outliers. These points can distort the regression model, leading to biased or misleading results. You may need to either remove these points or apply robust methods (e.g., robust regression) that down-weight outliers.
# 
# 3. Recommendations
# Improve Model Fit: If the residuals suggest non-linearity or heteroscedasticity:
# 
# Explore non-linear models like decision trees, random forests, or gradient boosting.
# Feature Engineering: Create new features (e.g., polynomial features) or apply transformations (e.g., log or square root) to stabilize variance or address skewness.
# Handle Outliers: If there are influential points with high leverage:
# 
# Investigate whether those data points are errors or anomalies.
# Consider using robust regression methods (e.g., Ridge, Lasso) that are less sensitive to outliers.
# Cross-Validation: Use cross-validation techniques (e.g., k-fold cross-validation) to ensure that the model's performance is consistent across different subsets of the data. This helps prevent overfitting and improves the generalizability of the model.
# 
# Model Tuning: If you're using machine learning models, perform hyperparameter tuning (e.g., grid search or random search) to find the optimal set of parameters that maximize performance metrics.
# 
# Further Residual Checks: Recheck the residuals after making improvements to ensure that the model assumptions are being met. This helps maintain model validity and ensures reliable predictions.
# 
# 4. Final Notes
# Model Refinement: Continue refining the model iteratively by evaluating additional diagnostic plots and metrics, such as precision-recall curves, confusion matrices, and ROC curves.
# Consider Business Implications: Ensure that your final model aligns with business objectives. For example, if predicting churn, focus on balancing false positives and false negatives based on business needs (e.g., you might prioritize catching as many churn customers as possible, even at the cost of more false positives).
# 

# In[ ]:




