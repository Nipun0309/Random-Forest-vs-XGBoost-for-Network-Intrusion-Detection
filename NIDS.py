import time
import warnings
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
X = np.loadtxt('KDDTrain+.csv', delimiter=',', dtype='object')
warnings.filterwarnings('ignore')

# Settings
pd.set_option('display.max_columns', None)
sns.set(style="darkgrid")
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
z = X[np.any(X == 'normal', axis=1)]
train = pd.read_csv("KDDTrain+.csv")
test = pd.read_csv("KDDTest+.csv")
print(train.describe())

print("Training data has {} rows & {} columns".format(train.shape[0], train.shape[1]))
input()
# print(train.describe())
train.drop(['num_outbound_cmds'], axis=1, inplace=True)
test.drop(['num_outbound_cmds'], axis=1, inplace=True)
# print("Training data has {} rows & {} columns".format(train.shape[0], train.shape[1]))
scaler = StandardScaler()

# extract numerical attributes and scale it to have zero mean and unit variance
cols = train.select_dtypes(include=['float64', 'int64']).columns
sc_train = scaler.fit_transform(train.select_dtypes(include=['float64', 'int64']))
sc_test = scaler.fit_transform(test.select_dtypes(include=['float64', 'int64']))
sc_traindf = pd.DataFrame(sc_train, columns=cols)
sc_testdf = pd.DataFrame(sc_test, columns=cols)
# print(np.shape(z))
# print(z)

encoder = LabelEncoder()

# extract categorical attributes from both training and test sets
cattrain = train.select_dtypes(include=['object']).copy()
cattest = test.select_dtypes(include=['object']).copy()

# encode the categorical attributes
traincat = cattrain.apply(encoder.fit_transform)
testcat = cattest.apply(encoder.fit_transform)

# separate target column from encoded data
enctrain = traincat.drop(['class'], axis=1)
enctest = testcat.drop(['class'], axis=1)
cat_Ytrain = traincat[['class']].copy()
train_x = pd.concat([sc_traindf, enctrain], axis=1)
train_y = train['class']

le = LabelEncoder()
le.fit(train_y)
train_y = le.transform(train_y)

test_x = pd.concat([sc_testdf, enctest], axis=1)
test_y = test['class']
counter = Counter(train_y)
plt.bar(counter.keys(), counter.values())
plt.title('Training Dataset')
plt.xlabel('Attack Type')
plt.ylabel('No: of records')
plt.show()
counter1 = Counter(test_y)
plt.bar(counter1.keys(), counter1.values())
plt.title('Testing Dataset')
plt.xlabel('Attack Type')
plt.ylabel('No: of records')
plt.show()
test_y = le.transform(test_y)
pca2 = PCA()
pca2.fit(train_x)
plt.plot(np.cumsum(pca2.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()
pca = PCA(0.95)
pca.fit(train_x)
train_x = pca.transform(train_x)
test_x = pca.transform(test_x)

forest = RandomForestClassifier(random_state=1)
n_estimators = [50, 100, 150, 200, 250]
max_depth = [5, 8, 15, 25, 30]
min_samples_split = [2, 5, 10, 15, 20]
min_samples_leaf = [1, 2, 5, 10, 15]
max_features = [2, 3, 4, 5, 6]

hyperF = dict(n_estimators=n_estimators, max_depth=max_depth,
              min_samples_split=min_samples_split,
              min_samples_leaf=min_samples_leaf,
              max_features=max_features)

gridF = GridSearchCV(forest, hyperF, cv=3, verbose=1,
                     n_jobs=-1, scoring="neg_log_loss")
bestF = gridF.fit(train_x, train_y)
print("Best: %f using %s" % (bestF.best_score_, bestF.best_params_))
rfc = RandomForestClassifier(**bestF.best_params_, n_jobs=-1)

rfc.fit(train_x, train_y)
disp = plot_confusion_matrix(rfc, train_x, train_y, display_labels=["dos", "normal", "probe"],
                             cmap=plt.cm.get_cmap('Blues'))
plt.show()

start = time.time()
y_pred = rfc.predict(train_x)
stop = time.time()
print(f"Time Taken by Random Forest to Predict on Training Data: {stop - start}s")

print(metrics.confusion_matrix(train_y, y_pred))
print(metrics.classification_report(train_y, y_pred))
print(metrics.accuracy_score(train_y, y_pred))

Model = XGBClassifier()
n_estimators = [50, 100, 150, 200, 250]
learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.2]
reg_lambda = [1, 2, 3, 4, 5]
min_child_weight = [1, 5, 10, 15, 20]
param_grid = dict(learning_rate=learning_rate, n_estimators=n_estimators, min_child_weight=min_child_weight,
                  reg_lambda=reg_lambda)
grid_search = GridSearchCV(Model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=3, verbose=1)
grid_result = grid_search.fit(train_x, train_y)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

model = XGBClassifier(objective='multi:softmax', **grid_result.best_params_,
                      n_jobs=-1)

model.fit(train_x, train_y)
start1 = time.time()
y_pred2 = model.predict(train_x)
stop1 = time.time()
print(f"Time Taken by XBoost to Predict on Training Data: {stop1 - start1}s")
print(y_pred2)
print(metrics.confusion_matrix(train_y, y_pred2))
print(metrics.classification_report(train_y, y_pred2))
print("Accuracy:", metrics.accuracy_score(train_y, y_pred2))
print(model.classes_)

start2 = time.time()
y_pred3 = rfc.predict(test_x)
stop2 = time.time()
print(f"Time Taken by Random Forest to Predict on Testing Data: {stop2 - start2}s")
print(metrics.confusion_matrix(test_y, y_pred3))
print(metrics.classification_report(test_y, y_pred3))

start3 = time.time()
y_pred4 = model.predict(test_x)
stop3 = time.time()
print(f"Time Taken by XGBoost to Predict on Testing Data: {stop3 - start3}s")
print(metrics.confusion_matrix(test_y, y_pred4))
print(metrics.classification_report(test_y, y_pred4))
