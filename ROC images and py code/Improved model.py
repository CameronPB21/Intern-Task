import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd

from sklearn import svm
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn import decomposition
from sklearn import feature_selection
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier

# Constants
NORMAL = 0
INTERICTAL = 1
ICTAL = 2

# Creating file_list for data
data_path = "C:\\Users\\camer\\Desktop\\Neurovigil\\EEG_Data\\"
folder_names = ['Z_normal', 'O_normal', 'N_interictal', 
                'F_interictal', 'S_ictal']
state_label = [NORMAL]*200 + [INTERICTAL]*200 + [ICTAL]*100
label_list = [data_path + "Z_normal" + "\\*.txt",
             data_path + "O_normal" + "\\*.txt",
             data_path + "N_interictal" + "\\*.txt",
             data_path + "F_interictal" + "\\*.txt",
             data_path + "S_ictal" + "\\*.txt"]
file_list = []
for label in label_list:
    file_list += glob.glob(label)
    
# Pulling data
data = [[]]*500
i = 0
for file in file_list:
    arr = open(file).read()
    data[i] = ([int(x) for x in arr.split()])
    i += 1
data = np.array(data)
data_transpose = np.transpose(data)

# Variable/data organization
x = data
labels = state_label
y = label_binarize(labels, classes=[0, 1, 2])
n_classes = y.shape[1]

# Using PCA to first decide where %95 of the data is, then fit to that %95
def my_preprocessing(data):
    data = preprocessing.scale(data)
    pca1 = decomposition.PCA()
    pca1.fit(data)
    minimum_data = 0
    components = 0
    for i in pca1.explained_variance_ratio_:
        minimum_data += i
        components += 1
        if minimum_data > 0.90:
            print("minimum_data = " + str(minimum_data) + "\nafter x components:" + str(components))
            break
    pca = decomposition.PCA(n_components = components)
    data = pca.fit_transform(data)
    return data

def svc_model(x_train, y_train, x_test, y_test):
    classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True))
    classifier.fit(x_train, y_train)
    y_test_preds = classifier.predict(x_test)
    fpr, tpr, roc_auc = roc_auc_calc(n_classes, y_test, y_test_preds)
    plot_roc(fpr, tpr, roc_auc, 'Receiver Operating Characteristic: SVC Model')
    results = [classifier, roc_auc[2]] 
    return results

# Create, train, test a Decision Tree Model
def decision_tree_model(x_train, y_train, x_test, y_test):
    decision_tree = DecisionTreeClassifier(random_state=0)
    decision_tree.fit(x_train, y_train)
    y_test_preds = decision_tree.predict(x_test)  
    fpr, tpr, roc_auc = roc_auc_calc(n_classes, y_test, y_test_preds)
    plot_roc(fpr, tpr, roc_auc, 'Receiver Operating Characteristic: Decision Tree Model')
    results = [decision_tree, roc_auc[2]] 
    return results

# Create, train, test a Random Forest Model
def random_forest_model(x_train, y_train, x_test, y_test):
    random_forest = RandomForestClassifier(n_estimators=100)
    random_forest.fit(x_train, y_train)
    y_test_preds = random_forest.predict(x_test)  
    fpr, tpr, roc_auc = roc_auc_calc(n_classes, y_test, y_test_preds)
    plot_roc(fpr, tpr, roc_auc, 'Receiver Operating Characteristic: Random Foreset Model')
    results = [random_forest, roc_auc[2]] 
    print(random_forest.score(x_test, y_test))
    return results

# Compute ROC curve and ROC area for a general model
def roc_auc_calc(n_classes, y_test, y_test_preds):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_test_preds[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    return fpr, tpr, roc_auc
        
# Plot an ROC curve
def plot_roc(fpr, tpr, roc_auc, graph_title):
    plt.figure()
    lw = 2
    plt.plot(fpr[2], tpr[2], color='red',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(graph_title)
    plt.legend(loc="upper left")
    plt.show()

def find_alpha(dt, x_train, y_train, x_test, y_test):
    path = dt.cost_complexity_pruning_path(x_train, y_train)
    ccp_alphas = path.ccp_alphas
    ccp_alphas = ccp_alphas[:-1]
    all_alphas = []
    for ccp_alpha in ccp_alphas:
        new_dt = DecisionTreeClassifier(random_state=0)
        scores = model_selection.cross_val_score(new_dt, x_train, y_train, cv=8)
        all_alphas.append([ccp_alpha, np.mean(scores), np.std(scores)])
    results = pd.DataFrame(all_alphas,
                           columns=['alpha', 'mean', 'std'])
    results.plot(x='alpha', y='mean', marker='o', linestyle='--')
    
        
data = my_preprocessing(data)
x_train, x_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=0)
svc_model(x_train, y_train, x_test, y_test)
dt_results = decision_tree_model(x_train, y_train, x_test, y_test)
rfm = random_forest_model(x_train, y_train, x_test, y_test)
#alpha = find_alpha(dt_results[0], x_train, y_train, x_test, y_test)















