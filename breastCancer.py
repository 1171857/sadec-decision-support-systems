import pandas as pd
import numpy as np
import os
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.model_selection import cross_val_score
import pickle
import matplotlib.pyplot as plt

class BreastCancer:
    test_size = 0.3
    max_depth = 4
    n_neighbors = 10
    list_of_results = []

    def __init__(self, path, max_depth, test_size, n_neighbors):
        self.path = path
        self.max_depth = max_depth
        self.test_size = test_size
        self.n_neighbors = n_neighbors
        self.load_Dataset(path)
        self.generate_plots()
        self.get_min_and_max_values()
        self.add_columns_normalized()
        self.create_dataset_with_selected_columns()
        self.divide_samples_with_same_ratio()
        self.generate_all_models()
        self.table_of_results()
        self.create_model_files()
        self.get_binary_tree()

    def load_Dataset(self, path):
        fn = os.path.join(os.path.dirname(__file__), 'brestcancer.csv')
        self.df_breastCancer = pd.read_csv(fn,delimiter=";",encoding = "ISO-8859-1")
        self.df_breastCancer = self.df_breastCancer.drop_duplicates()
        self.df_breastCancer = self.df_breastCancer.drop(columns=['id', 'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'])

    def get_min_and_max_values(self):
        df = self.df_breastCancer.drop(columns=["diagnosis"])
        self.min = df.min().values
        self.max = df.max().values

    def add_columns_normalized(self):
        self.df = self.df_breastCancer.drop(columns=["diagnosis"])
        i=0
        for column in self.df.columns:
            column_name = column + "_Interval_Level"
            self.df_breastCancer[column_name] = ((self.df_breastCancer[column] - self.min[i])/((self.max[i]-self.min[i])))
            i+=1

    def create_dataset_with_selected_columns(self):
        SelectedColumns=['radius_mean_Interval_Level', 'texture_mean_Interval_Level', 'perimeter_mean_Interval_Level', 
                            'area_mean_Interval_Level', 'smoothness_mean_Interval_Level', 'compactness_mean_Interval_Level', 
                            'concavity_mean_Interval_Level', 'concave points_mean_Interval_Level', 'symmetry_mean_Interval_Level', 
                            'fractal_dimension_mean_Interval_Level']
        self.dfML=self.df_breastCancer[SelectedColumns]
        self.dfML.columns=['radius', 'texture', 'perimeter', 'area', 'smoothness', 'compactness','concavity', 
                                        'concave points', 'symmetry', 'fractal_dimension']
        self.dfML['diagnosis'] = self.df_breastCancer['diagnosis']
    
    def divide_samples_with_same_ratio(self): 
        self.X = self.dfML.iloc[:,:-1]
        self.y = self.dfML.iloc[:,-1]
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,self.y,test_size=self.test_size, random_state=145,stratify=self.y)

        dfTrain = pd.DataFrame([self.y_train.value_counts(normalize=True)])
        dfTest = pd.DataFrame([self.y_test.value_counts(normalize=True)])
        html = dfTrain.to_html(classes="table")
        path_Test = os.path.join(os.path.join(os.path.dirname(__file__), 'templates'), 'ratio_Test.html')
        path_Train = os.path.join(os.path.join(os.path.dirname(__file__), 'templates'), 'ratio_Train.html')
        
        
        
        text_file = open(path_Train, 'w')
        text_file.write(html)
        text_file.close()
        html = dfTest.to_html(classes="table")
        text_file = open(path_Test, 'w')
        text_file.write(html)
        text_file.close()
        print('Train B/M distribution\n', self.y_train.value_counts(normalize=True))
        print('\n')
        print('Test B/M distribution\n', self.y_test.value_counts(normalize=True))

    def generate_all_models(self):
        self.generate_LogisticRegressionModel()
        self.generate_BinaryTreeModel()
        self.generate_GaussianNBModel()
        self.generate_KnnModel()
        self.generate_SVMModel()

    def generate_LogisticRegressionModel(self):
        clf = LogisticRegression()
        self.lrm= clf.fit(self.X_train,self.y_train)
        y_pred = self.lrm.predict(self.X_test)
        results = self.classifmodel_Metrics('lrm', self.lrm,self.y_test,y_pred)
        self.list_of_results.append(results)

    def generate_BinaryTreeModel(self):
        clf = tree.DecisionTreeClassifier(max_depth=self.max_depth,criterion='entropy')
        self.dtm = clf.fit(self.X_train, self.y_train)
        y_pred = self.dtm.predict(self.X_test)
        #feature_importance=pd.Series(self.dtm.feature_importances_,index=self.X.columns)
        #feature_importance.nlargest(10).plot(kind='barh')

        results= self.classifmodel_Metrics('dtm',self.dtm, self.y_test, y_pred)

        self.list_of_results.append(results)

    def generate_GaussianNBModel(self):
        self.nbm = GaussianNB()
        nbfit= self.nbm.fit(self.X_train, self.y_train)
        y_pred= nbfit.predict(self.X_test)

        results = self.classifmodel_Metrics('nbm', self.nbm, self.y_test, y_pred)
        self.list_of_results.append(results)

    
    def generate_KnnModel(self):    
        knclf= KNeighborsClassifier(n_neighbors=self.n_neighbors)
        self.knn= knclf.fit(self.X_train,self.y_train)
        y_pred= knclf.predict(self.X_test)

        results = self.classifmodel_Metrics('knn', self.knn, self.y_test, y_pred)
        self.list_of_results.append(results)

    def generate_SVMModel(self):
        self.svm = svm.SVC(kernel= 'rbf')
        svmfit= self.svm.fit(self.X_train,self.y_train)
        y_pred = svmfit.predict(self.X_test)

        results = self.classifmodel_Metrics('svm', self.svm, self.y_test, y_pred)
        self.list_of_results.append(results)
    
    def classifmodel_Metrics(self, modelName, model,actual, predicted):
        classes = list(np.unique(np.concatenate((actual,predicted))))

        confMtx = confusion_matrix(actual,predicted)

        print("Confusion Matrix")
        print(confMtx)

        #get classification
        report = classification_report(actual,predicted,output_dict=True)

        precision = report['macro avg']['precision']
        recall = report['macro avg']['recall']
        f1= report['macro avg']['f1-score']

        res = pd.Series({
            'ModelName':modelName,
            'Model': model,
            'accuracy': round(accuracy_score(predicted,actual),3),
            'precision':round(precision,3),
            'recall':round(recall,3),
            'F1':round(f1,3)
        })

        if len(classes) ==2:
            print("\naccuracy: {0:.2%}".format(round(accuracy_score(predicted,actual),3)))
            print("precision:  {0:.2%}".format(precision),", recall: {0:.2%}".format(recall),"F1:{0:.2%}".format(f1))
        elif len(classes) > 2:
            print("\n",classification_report(actual,predicted))
        return(res)

    def table_of_results(self):
        resdf = pd.DataFrame(self.list_of_results)
        resdf.iloc[:,[0,2,3,4,5]]
        resdf.sort_values(by=['F1','recall','precision'],ascending=[False,False,False],inplace=True)
        print(resdf.iloc[:,[0,2,3,4,5]])

        results=[]
        names= []

        print("\n10-fold cross Validation accuracy values")
        for name,model in zip(resdf.iloc[0:3,0],resdf.iloc[0:3,1]):
            res=cross_val_score(model,self.X,self.y,cv=10,scoring='f1_weighted')

            print('\n',res)
            print(name, 'F1 avg/std', round(res.mean(),3),'/',round(res.std(),3))
            results.append(res)
            names.append(name)
    
    def normalize_Values(self, listvalues):
        i=0
        for column in self.df.columns:
            listvalues[i] = (listvalues[i] - self.min[i])/(self.max[i]-self.min[i])
            i+=1
        return listvalues
    
    def desnormalize_Value(self, value, position):
        return value * (self.max[position]-self.min[position]) + self.min[position]
    
    def desnormalize_List(self, list_values):
        i=0
        for values in list_values:
            list_values[i] = list_values[i] * (self.max[i]-self.min[i]) + self.min[i]

        return list_values
    

    def create_model_files(self):
        pickle.dump(self.lrm, open(os.path.join(os.path.join(os.path.dirname(__file__), 'models'), 'lrm_model.pkl'),'wb'))
        pickle.dump(self.lrm, open(os.path.join(os.path.join(os.path.dirname(__file__), 'models'), 'dtm_model.pkl'),'wb'))
        pickle.dump(self.lrm, open(os.path.join(os.path.join(os.path.dirname(__file__), 'models'), 'nbm_model.pkl'),'wb'))
        pickle.dump(self.lrm, open(os.path.join(os.path.join(os.path.dirname(__file__), 'models'), 'knn_model.pkl'),'wb'))
        pickle.dump(self.lrm, open(os.path.join(os.path.join(os.path.dirname(__file__), 'models'), 'svm_model.pkl'),'wb'))
        
    def generate_plots(self):
        goal = self.df_breastCancer.diagnosis
        counts = goal.value_counts()
        percent100 = goal.value_counts(normalize=True).mul(100).round(1).astype(str) + '%'
        dftest = pd.DataFrame({'Nr Diagnosis': counts,'Percent': percent100})
        self.df_breastCancer['diagnosis'].value_counts().sort_index().plot.bar()
        plt.savefig(os.path.join(os.path.join(os.path.dirname(__file__), 'static'), 'ratio_diagnosis.png'))
        data_num = self.df_breastCancer.drop(columns=["diagnosis"])

        fig, axes = plt.subplots(len(data_num.columns)//3, 3, figsize=(50, 50))
        i = 0
        for triaxis in axes:
            for axis in triaxis:
                data_num.hist(column = data_num.columns[i], ax=axis, bins=50)
                i = i+1
        
        plt.savefig(os.path.join(os.path.join(os.path.dirname(__file__), 'static'), 'histogramas.png'))
        
        html = dftest.to_html(classes="table")
        os.path.join(os.path.join(os.path.dirname(__file__), 'templates'), 'ratio_diagnosis.html')
        text_file = open(os.path.join(os.path.join(os.path.dirname(__file__), 'templates'), 'ratio_diagnosis.html'), 'w')
        text_file.write(html)
        text_file.close()
    
    def get_binary_tree(self):
        plt.figure(figsize=(40,20))
        tree.plot_tree(self.dtm,feature_names = self.X.columns, class_names=['B','M'],filled=True)
        plt.savefig(os.path.join(os.path.join(os.path.dirname(__file__), 'static'), 'BinaryTree.png'))
        n_nodes = self.dtm.tree_.node_count
        children_left = self.dtm.tree_.children_left
        children_right = self.dtm.tree_.children_right
        feature = self.dtm.tree_.feature
        threshold = self.dtm.tree_.threshold
        
        self.predict = []
        j = 0
        for value in self.dtm.tree_.value:
            if value[0][0] >= value [0][1]:
                self.predict.append('B')
            else:
                self.predict.append('M')
            j = j + 1  

        node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
        is_leaves = np.zeros(shape=n_nodes, dtype=bool)
        stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
        while len(stack) > 0:
            # `pop` ensures each node is only visited once
            node_id, depth = stack.pop()
            node_depth[node_id] = depth

            # If the left and right child of a node is not the same we have a split
            # node
            is_split_node = children_left[node_id] != children_right[node_id]
            # If a split node, append left and right children and depth to `stack`
            # so we can loop through them
            if is_split_node:
                stack.append((children_left[node_id], depth + 1))
                stack.append((children_right[node_id], depth + 1))
            else:
                is_leaves[node_id] = True

        print(
            "The binary tree structure has {n} nodes and has "
            "the following tree structure:\n".format(n=n_nodes)
        )
        self.logic = 'import sys\n'
        self.questions = []
        for i in range(n_nodes):
            if is_leaves[i]:
                self.logic = self.logic + "if node == {node}:\n\tis_leaf = True\n\tsys.exit()\n".format(
                        node=i, 
                    )
                self.questions.append('')
            else:
                if is_leaves[children_left[i]] and is_leaves[children_right[i]]:
                    self.logic = self.logic + "if node == {node}:\n\tif response == 'Yes':\n\t\tnode={left}\n\t\tis_leaf = True\n\t\tsys.exit()\n\telse:\n\t\tnode={right}\n\t\tis_leaf = True\n\t\tsys.exit()\n".format(
                            node=i,
                            left=children_left[i],
                            feature=self.dtm.feature_names_in_[feature[i]],
                            threshold=round(self.desnormalize_Value(threshold[i],i),3),
                            right=children_right[i],
                        )
                    self.questions.append(self.dtm.feature_names_in_[feature[i]] + ' <= ' + "{threshold}".format(threshold=round(self.desnormalize_Value(threshold[i],i),3)))
                elif is_leaves[children_left[i]]:
                    self.logic = self.logic + "if node == {node}:\n\tif response == 'Yes':\n\t\tnode={left}\n\t\tis_leaf = True\n\t\tsys.exit()\n\telse:\n\t\tnode={right}\n\t\tsys.exit()\n".format(
                            node=i,
                            left=children_left[i],
                            feature=self.dtm.feature_names_in_[feature[i]],
                            threshold=round(self.desnormalize_Value(threshold[i],i),3),
                            right=children_right[i],
                        )
                    self.questions.append(self.dtm.feature_names_in_[feature[i]] + ' <= ' + "{threshold}".format(threshold=round(self.desnormalize_Value(threshold[i],i),3)))
                
                elif is_leaves[children_right[i]]:
                    self.logic = self.logic + "if node == {node}:\n\tif response == 'Yes':\n\t\tnode={left}\n\t\tsys.exit()\n\telse:\n\t\tnode={right}\n\t\tis_leaf = True\n\t\tsys.exit()\n".format(
                            node=i,
                            left=children_left[i],
                            feature=self.dtm.feature_names_in_[feature[i]],
                            threshold=round(self.desnormalize_Value(threshold[i],i),3),
                            right=children_right[i],
                        )
                    self.questions.append(self.dtm.feature_names_in_[feature[i]] + ' <= ' + "{threshold}".format(threshold=round(self.desnormalize_Value(threshold[i],i),3)))
                else:
                    self.logic = self.logic + "if node == {node}:\n\tif response == 'Yes':\n\t\tnode={left}\n\t\tsys.exit()\n\telse:\n\t\tnode={right}\n\t\tsys.exit()\n".format(
                            node=i,
                            left=children_left[i],
                            feature=self.dtm.feature_names_in_[feature[i]],
                            threshold=round(self.desnormalize_Value(threshold[i],i),3),
                            right=children_right[i],
                        )
                    self.questions.append(self.dtm.feature_names_in_[feature[i]] + ' <= ' + "{threshold}".format(threshold=round(self.desnormalize_Value(threshold[i],i),3)))
        print(self.logic)
    
    def get_logic(self):
        return self.logic

    def get_questions(self):
        return self.questions
    
    def get_Question(self, pos):
        if self.questions[pos] == '':
            return ''
        else:
            return self.questions[pos]+'?'
    
    def get_Result(self, pos):
        return 'Benign' if self.predict[pos] == 'B' else 'Malignant'

    def get_Results_From_All_Models(self, list_values):
        list_values_norm = self.normalize_Values(list_values)
        result = []
        result.append(self.lrm.predict([list_values_norm]))
        result.append(self.dtm.predict([list_values_norm]))
        result.append(self.nbm.predict([list_values_norm]))
        result.append(self.knn.predict([list_values_norm]))
        result.append(self.svm.predict([list_values_norm]))
        return result
        
        