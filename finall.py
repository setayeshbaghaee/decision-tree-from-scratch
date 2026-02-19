
import pandas as pd
import math
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random





full_data = pd.read_csv('bank-additional-full.csv', sep=";")

full_data.loc[full_data['age'] <= 17 , 'age']                            = 0 
full_data.loc[(full_data['age'] > 17) & (full_data['age'] <= 32), 'age'] = 1
full_data.loc[(full_data['age'] > 32) & (full_data['age'] <= 38) , 'age'] = 2
full_data.loc[(full_data['age'] > 38) & (full_data['age'] <= 47) , 'age'] = 3
full_data.loc[(full_data['age'] > 47) & (full_data['age'] <= 98) , 'age'] = 4
full_data.loc[full_data['age'] > 98 , 'age']= 5
full_data['job'] = full_data['job'].map({'housemaid':0 , 'services':1 , 'admin.':2, 'blue-collar':3, 'technician':4, 'retired':5, 'management':6, 'unemployed':7, 'self-employed':8, 'unknown':9, 'entrepreneur':10,'student':11}).astype(float)
full_data['marital'] = full_data['marital'].map({'married':0, 'single':1, 'divorced':2, 'unknown':3,}).astype(float)
full_data['education'] = full_data['education'].map({'basic.4y':0 , 'high.school':1 , 'basic.6y':2, 'basic.9y':3, 'professional.course':4, 'unknown':5,'university.degree':6, 'illiterate':7}).astype(float)
full_data['default'] = full_data['default'].map({"no": 0, "yes": 1, "unknown":2}).astype(float)
full_data['housing'] = full_data['housing'].map({"no": 0, "yes": 1, "unknown":2}).astype(float)
full_data['loan'] = full_data['loan'].map({"no": 0, "yes": 1, "unknown":2}).astype(float)
full_data['contact'] = full_data['contact'].map({"telephone": 0, "cellular": 1}).astype(float)
full_data['month'] = full_data['month'].map({'may':0 , 'jun':1 , 'jul':2, 'aug':3, 'nov':4, 'oct':5, 'dec':6, 'mar':7, 'apr':8, 'sep':9}).astype(float)
full_data['day_of_week'] = full_data['day_of_week'].map({'mon':0 , 'tue':1 , 'wed':2, 'thu':3, 'fri':4}).astype(float)
full_data.loc[full_data['duration'] <= 0 , 'duration']= 0 
full_data.loc[(full_data['duration'] > 0) & (full_data['duration'] <= 102.0), 'duration'] = 1
full_data.loc[(full_data['duration'] > 102.0) & (full_data['duration'] <= 180.0) , 'duration'] = 2
full_data.loc[(full_data['duration'] > 180.0) & (full_data['duration'] <= 319.0) , 'duration'] = 3
full_data.loc[(full_data['duration'] > 319.0) & (full_data['duration'] <= 4918) , 'duration'] = 4
full_data.loc[full_data['duration'] > 4918 , 'duration']= 5
full_data.loc[full_data['campaign'] <= 1 , 'campaign'] = 0 
full_data.loc[(full_data['campaign'] > 1) & (full_data['campaign'] <= 2), 'campaign'] = 1
full_data.loc[(full_data['campaign'] > 2) & (full_data['campaign'] <= 3) , 'campaign'] = 2
full_data.loc[(full_data['campaign'] > 3) & (full_data['campaign'] <= 56) , 'campaign'] = 3
full_data.loc[full_data['campaign'] > 56 , 'campaign']= 4
full_data.loc[full_data['pdays'] == 999, 'pdays'] = 0  
full_data.loc[(full_data['pdays'] > 0) & (full_data['pdays'] <= 5), 'pdays'] = 1  
full_data.loc[(full_data['pdays'] > 5) & (full_data['pdays'] <= 10), 'pdays'] = 2  
full_data.loc[(full_data['pdays'] > 10) & (full_data['pdays'] <= 20), 'pdays'] = 3  
full_data.loc[(full_data['pdays'] > 20), 'pdays'] = 4  
full_data["previous"] = full_data["previous"].astype(float)
full_data['poutcome'] = full_data['poutcome'].map({"nonexistent": 0, "failure": 1, "success":2}).astype(float)
full_data["emp.var.rate"] = full_data["emp.var.rate"].astype(float)
full_data.loc[full_data['cons.price.idx'] <= 92.201 , 'cons.price.idx']= 0 
full_data.loc[(full_data['cons.price.idx'] > 92.201) & (full_data['cons.price.idx'] <= 93.075), 'cons.price.idx'] = 1
full_data.loc[(full_data['cons.price.idx'] > 93.075) & (full_data['cons.price.idx'] <= 93.749) , 'cons.price.idx'] = 2
full_data.loc[(full_data['cons.price.idx'] > 93.749) & (full_data['cons.price.idx'] <= 93.994) , 'cons.price.idx'] = 3
full_data.loc[(full_data['cons.price.idx'] > 93.994) & (full_data['cons.price.idx'] <= 94.767) , 'cons.price.idx'] = 4
full_data.loc[full_data['cons.price.idx'] > 94.767 , 'cons.price.idx']= 5
full_data.loc[full_data['cons.conf.idx'] <= -50.8 , 'cons.conf.idx']= 0 
full_data.loc[(full_data['cons.conf.idx'] > -50.8) & (full_data['cons.conf.idx'] <= -42.7), 'cons.conf.idx'] = 1
full_data.loc[(full_data['cons.conf.idx'] > -42.7) & (full_data['cons.conf.idx'] <= -41.8) , 'cons.conf.idx'] = 2
full_data.loc[(full_data['cons.conf.idx'] > -41.8) & (full_data['cons.conf.idx'] <= -36.4) , 'cons.conf.idx'] = 3
full_data.loc[(full_data['cons.conf.idx'] > -36.4) & (full_data['cons.conf.idx'] <= -26.9) , 'cons.conf.idx'] = 4
full_data.loc[full_data['cons.conf.idx'] > -26.9 , 'cons.conf.idx']= 5
full_data.loc[full_data['euribor3m'] <= 0.634 , 'euribor3m']= 0 
full_data.loc[(full_data['euribor3m'] > 0.634) & (full_data['euribor3m'] <= 1.344), 'euribor3m'] = 1
full_data.loc[(full_data['euribor3m'] > 1.344) & (full_data['euribor3m'] <= 4.857) , 'euribor3m'] = 2
full_data.loc[(full_data['euribor3m'] > 4.857) & (full_data['euribor3m'] <= 4.961) , 'euribor3m'] = 3
full_data.loc[(full_data['euribor3m'] > 4.961) & (full_data['euribor3m'] <= 5.045) , 'euribor3m'] = 4
full_data.loc[full_data['euribor3m'] > 5.045 , 'euribor3m']= 5
full_data["nr.employed"] = full_data["nr.employed"].astype(float)
full_data['y'] = full_data['y'].map({"no": 0, "yes": 1}).astype(float)
feature_values_dict = {column: full_data[column].unique().tolist() for column in full_data.columns}

temp_data, test_data = train_test_split(full_data, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(full_data, test_size=0.2, random_state=42)

class Node:
    def __init__(self, feature=None, children=None, is_leaf=False, label=None):
        self.feature = feature  
        self.children = children if children is not None else {} 
        self.is_leaf = is_leaf 
        if is_leaf == True:
            self.label = label
        else:
            self.label = None  
 

class DecisionTree:
    def __init__(self):
        self.feature_values_dict = {}
        self.tree = None

    class Node:
        def __init__(self, feature=None, children=None, is_leaf=False, label=None):
            self.feature = feature  
            self.children = children if children is not None else {} 
            self.is_leaf = is_leaf 
            if is_leaf == True:
                self.label = label
            else:
                self.label = None  
 

    
    def gini(self, feature, examples):
        total_samples = len(examples)
        if total_samples == 0:
            return 0
        
        value_counts = examples[feature].value_counts()
        gini_index = 0
        
        for value, count in value_counts.items():
            p_value = count / total_samples  
            count_B = count  
            count_A_and_B = examples[(examples['y'] == True) & (examples[feature] == value)].shape[0]  
            
            p_condition = count_A_and_B / count_B if count_B > 0 else 0
            gini = 1 - (p_condition ** 2 + (1 - p_condition) ** 2)
            
            gini_index += p_value * gini
        
        return gini_index
    
    def entropy(self, feature, examples):
        total_samples = len(examples)
        if total_samples == 0:
            return 0
        
        value_counts = examples[feature].value_counts()
        entropy = 0 
        
        for value, count in value_counts.items():
            p_value = count / total_samples
            count_B = examples[examples[feature] == value].shape[0]  
            count_A_and_B = examples[(examples['y'] == True) & (examples[feature] == value)].shape[0]  
            
            p_condition = count_A_and_B / count_B if count_B > 0 else 0
            
            if 0 < p_condition < 1:
                entropy += p_value * (-p_condition * math.log(p_condition, 2) - (1 - p_condition) * math.log(1 - p_condition, 2))
        
        return entropy
    
    def entropy_y(self, examples):
        total_samples = len(examples)
        if total_samples == 0:
            return 0
        p = examples[examples['y'] == True].shape[0] / total_samples
        if p == 0 or p == 1: 
            return 0
        
        return (-p * math.log(p, 2)) - (1 - p) * math.log((1 - p), 2)
    
    def information_gain(self, examples, feature):
        return self.entropy_y(examples) - self.entropy(feature, examples)
    
    def choose_best_gini(self, examples, features):
        min_gini = float("inf")
        chosen_feature = None 
    
        for feature in features:
            if feature == "y": 
                continue
            gini_value = self.gini(feature, examples)  
            if gini_value < min_gini:
                min_gini = gini_value
                chosen_feature = feature  
        return chosen_feature
    
    def choose_best_gain(self, examples, features):
        max_gain = -100
        chosen_feature = None 
    
        for feature in features:
            if feature == "y": 
                continue
            gain_value = self.information_gain(examples, feature)  
            if gain_value > max_gain:
                max_gain = gain_value
                chosen_feature = feature  
        return chosen_feature
    
    def choose_best(self, data, features, criterion):
        if criterion == "gini":
            return self.choose_best_gini(data, features)
        elif criterion == "gain":
            return self.choose_best_gain(data, features)
    
    def split_data(self, data, feature):
        a = self.feature_values_dict[feature]
        feature_dict = {}
        for value in a:
            feature_dict[value] = data[data[feature] == value]
        return feature_dict
        
    def DT_hyperparams(self, features, data, min_samples_split, min_samples_leaf, max_depth, criterion, depth=0):
        best_feature = self.choose_best(data, features, criterion)
        
        if best_feature is None or len(features) == 0 or (max_depth is not None and depth >= max_depth):
            label_leaf = "yes" if data["y"].mean() >= 0.5 else "no"
            return self.Node(is_leaf=True, label=label_leaf)
    
        feature_dict = self.split_data(data, best_feature)
        root = self.Node(feature=best_feature, is_leaf=False)
    
        for value, filtered_data in feature_dict.items():
            if filtered_data.empty or len(filtered_data) < min_samples_split:
                root.children[value] = self.Node(is_leaf=True, label="yes" if data["y"].mean() >= 0.5 else "no")
            elif (filtered_data["y"] == 1).all() or (filtered_data["y"] == 0).all():
                root.children[value] = self.Node(is_leaf=True, label="yes" if (filtered_data["y"] == 1).all() else "no")
            elif len(filtered_data) < min_samples_leaf:
                root.children[value] = self.Node(is_leaf=True, label="yes" if data["y"].mean() >= 0.5 else "no")
            else:
                remaining_features = features.copy()
                remaining_features.remove(best_feature)
                root.children[value] = self.DT_hyperparams(remaining_features, filtered_data, min_samples_split, min_samples_leaf, max_depth, criterion, depth+1)
    
        return root
    
    def predict_sample(self, node, sample):
        if node.is_leaf:
            return 1 if node.label == "yes" else 0
        return self.predict_sample(node.children.get(sample[node.feature], node), sample)
   
    def accuracy(self, tree, data):
        predictions = []
        for _, row in data.iterrows():
            predictions.append(self.predict_sample(tree, row))
        correct = (predictions == data["y"]).sum() 
        return correct / len(data)  
    
    def train(self, train_data):
        self.feature_values_dict = feature_values_dict
        features = list(train_data.columns.drop("y"))
        self.tree = self.DT_hyperparams(features, train_data, min_samples_split=15, min_samples_leaf=1, max_depth=3, criterion="gini", depth=0)
        return self.tree

    
    def validate(self, val_data):
        return self.accuracy(self.tree, val_data)
    
    def test(self, test_data):
        return self.accuracy(self.tree, test_data)
    
    def print_tree(self, node, data, prefix="", is_last=True, parent_feature=None):
        connector = "└── " if is_last else "├── "
        
        if node.is_leaf:
            print(prefix + connector + f"🔹 Leaf → {node.label}")
        else:
            gini_value = self.gini(node.feature, data)
            gain_value = self.information_gain(data, node.feature)
            print(prefix + connector + f"🌳 {node.feature} (Gain: {gain_value:.3f}, Gini: {gini_value:.3f})")
        
        new_prefix = prefix + ("    " if is_last else "│   ")
        
        num_children = len(node.children)
        for i, (value, child) in enumerate(node.children.items()):
            is_last_child = (i == num_children - 1)  
            filtered_data = data[data[node.feature] == value]  
            print(new_prefix + f"│   [{node.feature} == {value}]")
            self.print_tree(child, filtered_data, new_prefix, is_last_child, parent_feature=node.feature)

  

dt = DecisionTree()
dt.train(train_data)
print(f"Train Accuracy: {dt.accuracy(dt.tree, train_data):.2%}")
print(f"Validation Accuracy: {dt.validate(val_data):.2%}")
print(f"Test Accuracy: {dt.test(test_data):.2%}")

root_node = dt.DT_hyperparams(train_data.columns.tolist(), train_data, min_samples_split=15, min_samples_leaf=1, max_depth=6, criterion="gini", depth=0)  
dt.print_tree(root_node, train_data)  

