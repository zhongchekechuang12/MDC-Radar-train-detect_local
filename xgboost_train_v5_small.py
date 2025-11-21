import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, make_scorer, accuracy_score, recall_score
from sklearn.feature_selection import RFE,SelectFromModel
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np
import joblib
import os
import seaborn as sns
import json

import shap
from config import *
# 获取当前工作目录
current_dir = os.getcwd()

# 获取上一级目录
desired_dir = os.path.dirname(current_dir)


print("当前工作目录:", desired_dir)

# csv地址

csv_path = os.path.join(desired_dir,  'train_xc','data', 'train','radar_data_train20.csv') 
new_csv_path = os.path.join(desired_dir, 'train_xc','data','test','横穿','radar_data_xianchang_1_f_hengchuan.csv')  # 新的CSV数据路径

# scaler地址
scaler_path = os.path.join(desired_dir, 'train_xc','result', f'xgboost{ii}','scaler.pkl')
scaler_path2 = os.path.join(desired_dir,  'train_xc','result',f'xgboost{ii}','scaler.json')

# 权重地址
model_path = os.path.join(desired_dir, 'train_xc','result',f'xgboost{ii}','xgboost_model.pkl')

model_path2 = os.path.join(desired_dir, 'train_xc','result',f'xgboost{ii}','xgboost_model.json')

rfe_path = os.path.join(desired_dir, 'train_xc','result',f'xgboost{ii}','RFE.json')
# 创建所有必要的目录WS
directories = [
    os.path.dirname(scaler_path),
    os.path.dirname(scaler_path2),
    os.path.dirname(model_path),
    os.path.dirname(model_path2),
    os.path.dirname(rfe_path)
]

for directory in directories:
    os.makedirs(directory, exist_ok=True)

#=====================================加载数据========================
df = pd.read_csv(csv_path)

# 去除不需要的列

df = df.drop(columns=['frame', 'id', 'tag'])
# 检查并处理缺失值
df = df.dropna()

# 特征工程和数据预处理
X = df.drop('class', axis=1)
y = df['class']
feature_names = X.columns
# 数据标准化

scaler = StandardScaler()
X = scaler.fit_transform(X)

# 保存StandardScaler实例
joblib.dump(scaler, scaler_path)

# 将 scaler 参数保存为 JSON 格式
scaler_params = {
    'mean': scaler.mean_.tolist(),
    'scale': scaler.scale_.tolist()
}
with open(scaler_path2, 'w') as f:
    json.dump(scaler_params, f)
    


#===========================将数据分为训练集和测试集=====================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# ===============================特征选择 ==========================
# 选择模型
# ===============================特征选择 ==========================
# 选择模型
estimator = RandomForestClassifier(n_estimators=100, random_state=42)
# 首先训练模型
estimator.fit(X_train, y_train)
#selector = SelectFromModel(estimator, max_features=9,threshold=0.05)
selector = SelectFromModel(estimator, max_features=10,threshold=0.01)
#===输出特征重要性==
print(f'X_train 的维度: {X_train.shape}')
positive_count = df[df['class'] == 1].shape[0]
negative_count = df[df['class'] == 0].shape[0]

# 获取特征重要性分数
importances = estimator.feature_importances_
# 创建一个 DataFrame 来显示特征及其重要性分数
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)  # 按重要性排序
# 输出特征重要性
print(importance_df)

X_train = selector.transform(X_train)
X_test = selector.transform(X_test)

# 保存选择的特征索引
selected_features = feature_names[selector.get_support()].tolist()  # 获取选择的特征名称
with open(rfe_path, 'w') as f:
    json.dump(selected_features, f)  # 保存为 JSON 文件

# 转换数据格式为DMatrix，这是XGBoost高效的数据格式
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# 统计正负样本数量
positive_count = df[df['class'] == 1].shape[0]
negative_count = df[df['class'] == 0].shape[0]
XGBOOST_PARAMS['scale_pos_weight'] =  int(negative_count / positive_count) / SCALE_POS_WEIGHT_FACTOR

'''
param_grid = PARAM_GRID
def combined_score(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return 0.5 * accuracy + 0.5 * recall  # 调整权重以适应需求
custom_scorer = make_scorer(combined_score)

# 使用网格搜索进行超参数优化
xgb_model = xgb.XGBClassifier()
grid_search = GridSearchCV(xgb_model, param_grid, scoring='recall', cv=3, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
'''
# ==================================训练模型并记录评估结果======================
params = XGBOOST_PARAMS 
evals_result = {}
num_round = 1000
bst = xgb.train(params, dtrain, num_round, evals=[(dtrain, 'train'), (dtest, 'eval')], evals_result=evals_result)



# 保存模型权重参数
joblib.dump(bst, model_path)
# 保存模型为 JSON 格式
bst.save_model(model_path2)

# 输出每个训练数据点的预测结果和实际标签
y_train_pred_prob = bst.predict(dtrain)
y_train_pred = (y_train_pred_prob > threshold).astype(int)

# 输出测试集的预测结果
y_pred_prob = bst.predict(dtest)# config.py

y_pred = (y_pred_prob > threshold).astype(int)

# =========================输出训练集的评估指标===========================
print(f"\n训练集 Accuracy: {accuracy_score(y_train, y_train_pred)}")
print(classification_report(y_train, y_train_pred))

# 计算所有标签为1的样本的索引
indices_label_1 = (y_train == 1.0)
# 根据索引提取实际标签和预测标签
y_train_label_1 = y_train[indices_label_1]
y_train_pred_label_1 = y_train_pred[indices_label_1]
# 计算准确度
accuracy_label_1 = accuracy_score(y_train_label_1, y_train_pred_label_1)
print(f"\n训练集标签为1的样本准确度: {accuracy_label_1}")

# 输出测试集评估指标
print(f"\n测试集Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))
print(int(negative_count / positive_count))
# 计算所有标签为1的样本的索引
indices_label_2 = (y_test == 1.0)
# 根据索引提取实际标签和预测标签
y_train_label_2 = y_test[indices_label_2]
y_train_pred_label_2 = y_pred[indices_label_2]
# 计算准确度
accuracy_label_2 = accuracy_score(y_train_label_2, y_train_pred_label_2)
print(f"\n测试集标签为1的样本准确度: {accuracy_label_2}")

print(f"模型权重参数和标准化参数已保存到 {model_path}")



