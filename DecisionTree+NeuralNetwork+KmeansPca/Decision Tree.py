#任务一：决策树实现
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

data = pd.read_csv("breast cancer.csv")
# 挑选特征
x = data[['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean',
          'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se',
          'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se',
          'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst',
          'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst',
          'fractal_dimension_worst']]
# 选择目标值
y = data['diagnosis']
# 填补缺失值

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
dic = DictVectorizer(sparse=False)
x_train = dic.fit_transform(x_train.to_dict(orient="records"))  # x_train.to_dict(orient="records")将dataframe转化为字典
x_test = dic.transform(x_test.to_dict(orient="records"))
dec = DecisionTreeClassifier()
dec.fit(x_train, y_train)

print("预测的准确率", dec.score(x_test, y_test))

export_graphviz(dec, out_file="E:/lu_learning_demo/tree.dot",
                feature_names=['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
                               'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean',
                               'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se',
                               'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
                               'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst',
                               'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave points_worst',
                               'symmetry_worst', 'fractal_dimension_worst'])
# lu_learning_demo
