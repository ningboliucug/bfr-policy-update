import pandas as pd

# 读取数据集
#dataset_path = '/Users/liuningbo/Desktop/1_under_writing/5-MLforAC/dataSet/amazon-employee-access-challenge/train.csv'
dataset_path = '/root/5_MLForAC/dataSet/uci-2.0.csv'
data = pd.read_csv(dataset_path)

# 将所有列（包括标签列）转换为字符串以便比较
data_str = data.apply(lambda row: '_'.join(row.astype(str)), axis=1)

# 创建一个字典来存储每个样本的出现次数和行号
sample_to_indices = {}

for index, sample in data_str.items():
    if sample in sample_to_indices:
        sample_to_indices[sample].append(index)
    else:
        sample_to_indices[sample] = [index]

# 找出重复的样本点
duplicates = {k: v for k, v in sample_to_indices.items() if len(v) > 1}

# 格式化输出重复的样本点
for duplicate_num, (sample, indices) in enumerate(duplicates.items(), 1):
    print(f"Duplicate_num: {duplicate_num}: {{", end="")
    for i, index in enumerate(indices):
        print(f"{index}: '{sample}'", end="")
        if i < len(indices) - 1:
            print(", ", end="")
    print("}")


# 去除标签列，剩下的作为特征列
features = data.drop(columns=['ACTION'])

# 将特征列转换为字符串以便比较
features_str = features.apply(lambda row: '_'.join(row.astype(str)), axis=1)

# 创建一个字典来存储每个特征组合对应的标签集合和行号
feature_to_actions = {}

# 使用字典来记录每个特征组合对应的标签
conflicts = {}

for index, row in data.iterrows():
    feature = features_str[index]
    action = row['ACTION']
    if feature in feature_to_actions:
        if action not in feature_to_actions[feature]['actions']:
            # 发现标签冲突的样本
            conflict_num = len(conflicts) + 1
            if conflict_num not in conflicts:
                conflicts[conflict_num] = {}
            conflicts[conflict_num][feature_to_actions[feature]['index']] = feature
            conflicts[conflict_num][index] = feature
    else:
        feature_to_actions[feature] = {'actions': {action}, 'index': index}

# 格式化输出标签冲突的样本点
for conflict_num, conflict in conflicts.items():
    print(f"Conflict_num: {conflict_num}: {conflict}")



# 去除标签列，剩下的作为特征列
features = data.drop(columns=['ACTION'])

# 将特征列转换为字符串以便比较
features_str = features.apply(lambda row: '_'.join(row.astype(str)), axis=1)

# 创建一个字典来存储每个特征组合对应的标签集合和行号
feature_to_actions = {}

# 使用字典来记录每个特征组合对应的标签
conflicts = {}

for index, row in data.iterrows():
    feature = features_str[index]
    action = row['ACTION']
    if feature in feature_to_actions:
        if action not in feature_to_actions[feature]['actions']:
            # 发现标签冲突的样本
            conflict_num = len(conflicts) + 1
            if conflict_num not in conflicts:
                conflicts[conflict_num] = []
            conflicts[conflict_num].append(feature_to_actions[feature]['index'])
            conflicts[conflict_num].append(index)
    else:
        feature_to_actions[feature] = {'actions': {action}, 'index': index}

# 获取所有冲突样本点的索引
conflict_indices = {index for indices in conflicts.values() for index in indices}

# 删除冲突样本点
data_cleaned = data.drop(index=conflict_indices)

# 保存删除后的数据集到新的 CSV 文件
data_cleaned.to_csv('./cleaned_conflict_uci-2.0.csv', index=False)

print("删除冲突样本点后的数据集已保存到 'cleaned_conflict_uci-2.0.csv'")
