 #!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','exercised_stock_options', 'loan_advances',
                 'total_stock_value','bonus','from_ratio',
                 'salary','total_payments','long_term_incentive',
                 'shared_receipt_with_poi','other'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers

data_dict.pop('TOTAL')

### Task 3: Create new feature(s)
    
def get_stock_ratio(payments, stock):
    if str(payments) == 'NaN':
        payments = 0
    if str(stock) == 'NaN':
        stock = 0
    if stock == 0 and payments == 0:
        return 0
    else:
        return float(stock)/float(stock + payments)
    
def get_message_ratio(poi_messages, messages):
    if str(messages) == 'NaN' or str(poi_messages) == 'NaN':
        return 0
    return float(poi_messages)/float(messages)

for data in data_dict.values():
    data['stock_ratio'] = get_stock_ratio(data['total_payments'], data['total_stock_value'])
    data['from_ratio'] = get_message_ratio(data['from_this_person_to_poi'], data['from_messages'])
    data['to_ratio'] = get_message_ratio(data['from_poi_to_this_person'], data['to_messages'])
    


### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

from sklearn.preprocessing import MinMaxScaler

features = MinMaxScaler().fit_transform(features)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(class_weight='balanced', 
                             criterion='gini', 
                             max_depth=4,
                             max_features='auto', 
                             max_leaf_nodes=8,
                             min_samples_leaf=2, 
                             min_samples_split=4,
                             min_weight_fraction_leaf=0, 
                             presort=True, 
                             random_state=None,
                             splitter='best')

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)