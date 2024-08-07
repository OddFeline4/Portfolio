import pandas as pd
import numpy as np

class MyTreeClf:
    def __init__(self, max_depth = 5, min_samples_split = 2, max_leaf = 20):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leaf = max_leaf
        self.leafs_cnt = 0
        self.tree = dict()
        self.potential_leafs = 0
        self.roots = 1

    def __str__(self):
        return 'MyTreeClf class: '+ f'{", ".join(str(i[0])+"="+str(i[1]) for i in self.__dict__.items())}'

    def get_best_split(self, X: pd.DataFrame, y):
        columns = X.columns
        info_profit = 0
        split_value = None
        col_name = None
        start_entropy = self.calculate_entropy(y)
        for col in columns:
            uniq_values = X[col].sort_values().unique()
            uniq = [(uniq_values[i]+uniq_values[i+1])/2 for i in range(len(uniq_values)-1)]
            for uniq_value in uniq:
                indexes_left = X[X[col] <= uniq_value].index
                indexes_right = X[X[col] > uniq_value].index
                y_left = y[indexes_left]
                y_right = y[indexes_right]
                S_left = self.calculate_entropy(y_left)
                S_right = self.calculate_entropy(y_right)
                info_profit_step = start_entropy - (y_left.count()/y.count())*S_left - (y_right.count()/y.count())*S_right
                if info_profit_step > info_profit:
                    info_profit = info_profit_step
                    split_value = uniq_value
                    col_name = str(col)
        return col_name, split_value, info_profit


    def calculate_entropy(self,y):
        if y.sum() == y.count() or y.sum() == 0:
            return 0
        else:
            return -1 *((y.sum()/y.count())*np.log2(y.sum()/y.count()) + ((y.count()-y.sum())/y.count())*np.log2(((y.count()-y.sum())/y.count())))


    def fit(self, X: pd.DataFrame, y: pd.Series, depth = 0, root = []):
        self.build_tree_recursive(root,X,y,depth)
        self.tree = root
        # if self.leafs_cnt < self.max_leaf and depth < self.max_depth:
        #     cur_col, cur_value, cur_info = self.get_best_split(X, y)
        #     X_left = X[X[cur_col] <= cur_value]
        #     X_right = X[X[cur_col] > cur_value]
        #     y_left = y.iloc[X_left.index]
        #     y_right = y.iloc[X_right.index]
        #     depth += 1
        #     branch[1] = {'col': cur_col, 'operand': '<=', 'value': cur_value}
        #     if depth == self.max_depth:
        #         self.leafs_cnt += 1
        #     self.fit(X_left,y_left,depth,branch[1])
        #     self.fit(X_right,y_right,depth,branch)
        #     pass

    def build_tree_recursive(self, root, x: pd.DataFrame, y: pd.Series, depth, string='1'):
        if (depth == self.max_depth) or (y.sum()/y.count() == 0.0 or y.sum()/y.count() == 1.0) or (len(y) < self.min_samples_split) or (self.leafs_cnt + self.potential_leafs) >= self.max_leaf:
            probability = y.sum()/y.count()
            if string[-1] == '1':
                root.append([string[:-2]+'.left', probability])
            else:
                root.append([string[:-2]+'.right', probability])
                self.leafs_cnt += 1
                self.roots -= 1
                #self.potential_leafs -= 2


        else:
            depth += 1
            cur_col, cur_value, cur_info = self.get_best_split(x, y)
            x_left = x[x[cur_col] <= cur_value]
            x_right = x[x[cur_col] > cur_value]
            y_left = y[x_left.index]
            y_right = y[x_right.index]
            condition = [string, f'{cur_col} <= {cur_value}']
            self.potential_leafs = self.roots
            root.append(condition)
            self.build_tree_recursive(root,x_left,y_left,depth,string+'.1')
            self.build_tree_recursive(root,x_right,y_right,depth,string+'.2')
            self.roots += 1
            #self.leafs_cnt += 1
            self.potential_leafs += 1


    def print_tree(self):
        return self.tree







# ser1 = pd.Series([10,8,9,9,9,7,8,9])
# ser2 = pd.Series([7,8,9,8,3,2,1,3])
# y = pd.Series([1,1,0,0,1,0,1,0])
# X =pd.concat([ser1,ser2],axis='columns')
# X.columns = ['first','second']
# obj = MyTreeClf(max_depth=1)
# obj.fit(X,y)
#
# obj.print_tree()

df = pd.read_csv('data_banknote_authentication.txt', header=None)
df.columns = ['variance', 'skewness', 'curtosis', 'entropy', 'target']
X, y = df.iloc[:,:4], df['target']
obj = MyTreeClf(max_depth=3, max_leaf=5, min_samples_split=2)
obj.fit(X,y)
tree = obj.print_tree()
print(tree)
prob = sum(map(lambda y:y[1],filter(lambda x: isinstance(x[1],float), tree)))
print(prob)



import pandas as pd
import numpy as np

class MyTreeClf:
    def __init__(self, max_depth=5, min_samples_split = 2, max_leafs = 20):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.leafs_cnt = 0
        self.tree = dict()
        self.potential_leafs = 0
        self.roots = 0

    def __str__(self):
        return 'MyTreeClf class: '+ f'{", ".join(str(i[0])+"="+str(i[1]) for i in self.__dict__.items())}'

    def get_best_split(self, X: pd.DataFrame, y):
        columns = X.columns
        info_profit = 0
        split_value = None
        col_name = None
        start_entropy = self.calculate_entropy(y)
        for col in columns:
            uniq_values = X[col].sort_values().unique()
            uniq = [(uniq_values[i]+uniq_values[i+1])/2 for i in range(len(uniq_values)-1)]
            for uniq_value in uniq:
                indexes_left = X[X[col] <= uniq_value].index
                indexes_right = X[X[col] > uniq_value].index
                y_left = y[indexes_left]
                y_right = y[indexes_right]
                S_left = self.calculate_entropy(y_left)
                S_right = self.calculate_entropy(y_right)
                info_profit_step = start_entropy - (y_left.count()/y.count())*S_left - (y_right.count()/y.count())*S_right
                if info_profit_step > info_profit:
                    info_profit = info_profit_step
                    split_value = uniq_value
                    col_name = str(col)
        return col_name, split_value, info_profit


    def calculate_entropy(self,y):
        if y.sum() == y.count() or y.sum() == 0:
            return 0
        else:
            return -1 *((y.sum()/y.count())*np.log2(y.sum()/y.count()) + ((y.count()-y.sum())/y.count())*np.log2(((y.count()-y.sum())/y.count())))


    def fit(self, X: pd.DataFrame, y: pd.Series, depth = 0, root = []):
        self.build_tree_recursive(root,X,y,depth)
        self.tree = root
        self.leafs_cnt = len(list(filter(lambda x: isinstance(x[1],float), self.tree)))

    def build_tree_recursive(self, root, x: pd.DataFrame, y: pd.Series, depth, string='1'):
        self.roots += 1
        self.potential_leafs = self.roots * 2 - self.roots - 1
        if (depth == self.max_depth) or (y.sum()/y.count() == 0.0 or y.sum()/y.count() == 1.0) or (len(y) < self.min_samples_split) or (self.leafs_cnt + self.potential_leafs) >= self.max_leafs:
            probability = y.sum()/y.count()
            if string[-1] == '1':
                root.append([string[:-2]+'.left', probability])
            else:
                root.append([string[:-2]+'.right', probability])
            self.leafs_cnt += 1
            self.roots -= 1
            self.potential_leafs -= 2


        else:
            depth += 1
            cur_col, cur_value, cur_info = self.get_best_split(x, y)
            x_left = x[x[cur_col] <= cur_value]
            x_right = x[x[cur_col] > cur_value]
            y_left = y[x_left.index]
            y_right = y[x_right.index]
            condition = [string, f'{cur_col} <= {cur_value}']
            root.append(condition)
            self.build_tree_recursive(root,x_left,y_left,depth,string+'.1')
            self.build_tree_recursive(root,x_right,y_right,depth,string+'.2')
            #self.leafs_cnt += 1
            #self.potential_leafs -= 2


    def print_tree(self):
        return self.tree

# df = pd.read_csv('data_banknote_authentication.txt', header=None)
# df.columns = ['variance', 'skewness', 'curtosis', 'entropy', 'target']
# X, y = df.iloc[:,:4], df['target']
# obj = MyTreeClf(max_depth=3, max_leafs=5, min_samples_split=2)
# obj.fit(X,y)
# tree = obj.print_tree()
# print(tree)
# prob = sum(map(lambda y:y[1],filter(lambda x: isinstance(x[1],float), tree)))
# print(prob)









