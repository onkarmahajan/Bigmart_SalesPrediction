import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import pickle

train = pd.read_csv("Train.csv")
test = pd.read_csv("Test.csv")

train['source'] = 'train'
test['source'] = 'test'
dataset = pd.concat([train, test], ignore_index = True)

weight_avg = pd.pivot_table(dataset, values = 'Item_Weight', index = 'Item_Identifier')
miss_bool = dataset['Item_Weight'].isnull()
print(f'Missing values in Item_Weight: {sum(miss_bool)}')
dataset.loc[miss_bool, 'Item_Weight'] = dataset.loc[miss_bool, 'Item_Identifier'].apply(lambda x: weight_avg.loc[x].values[0])
print(f"Missing values in Item_Weight: {sum(dataset['Item_Weight'].isnull())}")

from scipy.stats import mode
outlet_size_mode = pd.pivot_table(dataset, values = 'Outlet_Size', columns = 'Outlet_Type', aggfunc = (lambda x: mode(x).mode[0]))
miss_bool = dataset['Outlet_Size'].isnull()
print(f'Missing values in Outlet_Size: {sum(miss_bool)}')
dataset.loc[miss_bool, 'Outlet_Size'] = dataset.loc[miss_bool, 'Outlet_Type'].apply(lambda x: outlet_size_mode[x])
print(f"Missing values in Outlet_Size: {sum(dataset['Outlet_Size'].isnull())}")

dataset['Item_Type_Combined'] = dataset['Item_Identifier'].apply(lambda x: x[0:2])

print(f"Initial Categories in Item_Fat_Content: \n{dataset['Item_Fat_Content'].value_counts()}")
dataset['Item_Fat_Content'] = dataset['Item_Fat_Content'].replace({'LF':'Low Fat', 'low fat':'Low Fat', 'reg':'Regular'})
print(f"\nAfter replacing Categories in Item_Fat_Content: \n{dataset['Item_Fat_Content'].value_counts()}")
dataset.loc[dataset['Item_Type_Combined'] == 'NC', 'Item_Fat_Content'] = 'Non Edible'
print(f"\nAdding Non Edible Category in Item_Fat_Content: \n{dataset['Item_Fat_Content'].value_counts()}")

visibility_mean = pd.pivot_table(dataset, values = 'Item_Visibility', index = 'Item_Identifier')
zero_bool = (dataset['Item_Visibility'] == 0)
print(f"Initial Categories in Item_Visibilty: {sum(zero_bool)}")
dataset.loc[zero_bool, 'Item_Visibility'] = dataset.loc[zero_bool, 'Item_Identifier'].apply(lambda x: visibility_mean.loc[x].values[0])
zero_bool = (dataset['Item_Visibility'] == 0)
print(f"After Replacing Zero Visibility values in Item_Visibilty: {sum(zero_bool)}")

print(f"Value Count of Outlet_Establishment_Year: \n{dataset['Outlet_Establishment_Year'].value_counts()}")
dataset['Outlet_Years'] = dataset['Outlet_Establishment_Year'].apply(lambda x: 2013 - x)
print(f"\nValue Count of Outlet_Years: \n{dataset['Outlet_Years'].value_counts()}")

dataset = pd.get_dummies(dataset, columns = ['Item_Fat_Content', 'Item_Type_Combined', 'Outlet_Identifier', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type'])

dataset.drop(['Item_Identifier', 'Item_Type', 'Outlet_Establishment_Year'], axis = 1, inplace = True)
#Getting back train and test
train = dataset.loc[dataset['source'] == 'train']
test = dataset.loc[dataset['source'] == 'test']
print(train.shape)
print(test.shape)

#Dropping unnecessary columns
train.drop(['source'], axis = 1, inplace = True)
test.drop(['source', 'Item_Outlet_Sales'], axis = 1, inplace = True)

X = train.drop(['Item_Outlet_Sales'], axis = 1).values
y = train['Item_Outlet_Sales'].values

#Splitting train into X_train, y_train, X_test and y_test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, y_train)
y_predlinear = reg.predict(X_test)

from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor(max_depth=15, min_samples_leaf=100)
dtr.fit(X_train, y_train)
y_preddtr = dtr.predict(X_test)

from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators=400,max_depth=6, min_samples_leaf=100,n_jobs=4)
rfr.fit(X_train, y_train)
y_predrfr = rfr.predict(X_test)

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit

def find_best_model_using_gridsearchcv(x,y):
    algos = {
        'linear_regression' : {
            'model': LinearRegression(),
            'params': {
                'normalize': [True, False]
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion' : ['mse','friedman_mse'],
                'max_depth': [None, 5, 10, 15, 20, 25],
                'min_samples_leaf': [1, 100, 200],
                'splitter': ['best','random'],
                'max_features':  ['auto', 'sqrt', 'log2']
            }
        },
        # 'random_forest': {
        #     'model': RandomForestRegressor(),
        #     'params': {
        #         'n_estimators': [300, 400, 500],
        #         'max_depth': [4, 6, 8], 
        #         'min_samples_leaf': [100, 200],
        #         'n_jobs': [4, 6]
        #     }
        # }
    }
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        gs =  GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(x,y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })

    return pd.DataFrame(scores,columns=['model','best_score','best_params'])

find_best_model_using_gridsearchcv(X,y)

pickle.dump(reg, open("LinearModel.pkl", 'wb'))
pickle.dump(dtr, open("DecisionTree.pkl", 'wb'))
pickle.dump(rfr, open("RandomForest.pkl", 'wb'))
