#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import QuantileTransformer, OneHotEncoder, Normalizer
from sklearn import set_config
set_config(display='diagram')
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from joblib import load, dump
from sklearn.inspection import permutation_importance, plot_partial_dependence


# In[15]:


df = pd.read_csv('Df_Bialetti.csv')
df.head()


# In[16]:


df['Giorno'] = pd.to_datetime(df['Giorno'],dayfirst=True, yearfirst=False)
df.head()


# In[17]:


s = df['Giorno'].dt.weekday
df['day_week'] =s
df['month'] = pd.DatetimeIndex(df['Giorno']).month
df.head()
df.columns


# In[18]:


df=df.rename(columns={'Spesa Google ':'google'})
df.sample(10)


# In[19]:


categorical= [
  'day_week', 'month'
]

numerical=[
    'Spesa Facebook','google','Organico'
]

all_features = categorical + numerical

transformers =[
  ('one hot', OneHotEncoder(handle_unknown='ignore'),categorical),
  ('scaler', QuantileTransformer(), numerical),
  ('normalizer',Normalizer(), all_features)
]


ct = ColumnTransformer(transformers,verbose_feature_names_out=True)


steps =[
 ('column_transformer',ct),
 ('model', MLPRegressor())
]

pipeline= Pipeline(steps)

pipeline
    


# In[20]:


param_space={
    'column_transformer__scaler__n_quantiles':[10,100],
    'column_transformer__normalizer':[ Normalizer(), 'passthrough' ],
    'model__hidden_layer_sizes':[(20,20),(50,50)],
    'model__alpha':[0.01, 0.001]
}

grid = GridSearchCV(pipeline, param_grid=param_space, cv=3, verbose=2)
grid


# In[21]:


X = df[all_features]
y= df['Transazioni']


# In[22]:


X_train, X_test,y_train, y_test = train_test_split(X,y )


# In[24]:


grid.fit(X_train, y_train)


# In[25]:


grid.best_params_


# In[26]:


grid.best_estimator_


# In[27]:


dump(grid.best_estimator_, 'Rete_Neurale_Bialetti.joblib')


# In[28]:


pipe = load('Rete_Neurale_Bialetti.joblib')
     
pipe.score(X_test, y_test)


# In[29]:


results = pd.DataFrame(grid.cv_results_)
results


# In[30]:


prediction = pipe.predict(df)


# In[31]:


df['prediction']= prediction
df.sample(10)


# In[32]:


df['error'] = df['Transazioni']-df['prediction']
df.tail(10)


# In[33]:


sns.scatterplot(data = df, x='error', y='Transazioni', alpha = 0.1)


# In[34]:


sns.distplot(df['error'])


# In[35]:


sns.lineplot(data= df, x='Giorno', y= 'Transazioni')
sns.lineplot(data= df, x='Giorno', y= 'prediction')


# In[36]:


rt=df[['Giorno','Transazioni', 'Spesa Facebook', 'google', 'Organico','day_week', 'month']]
rt.head()


# In[57]:


rt.to_csv('Df_Bialetti.csv')


# In[42]:


plot_partial_dependence(pipe, X_test, features=['google','Spesa Facebook', 'Organico'])


# In[ ]:




