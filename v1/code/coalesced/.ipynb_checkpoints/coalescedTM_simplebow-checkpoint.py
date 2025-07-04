#Multioutput PyCoalescedTM

import pickle
import gzip
import sys
sys.path.append('../../../PyCoalescedTsetlinMachineCUDA/')
from PyCoalescedTsetlinMachineCUDA.tm import MultiOutputTsetlinMachine
from sklearn.metrics import classification_report
from sklearn.model_selection  import train_test_split
from sklearn.metrics import multilabel_confusion_matrix
import argparse
import logging
import numpy as np
import os

filename = '../../processed_data/simple_bag_of_words_features.pkl.gz'
outputfile = '../../results/coalesced_simplebow.txt'
masteroutputfile = '../../results/master_results.csv'

num_clauses = 5000
T = 3950
s = 1
epochs = 10
train_epochs= 200


fo = open(outputfile, 'w')
fo.write('\n Using : PyCoalescedTsetlinMachineCUDA \nEncoding: Simple BOW \n')
fo.write('\n TM parameters : Clauses:'+str(num_clauses)+' T:'+str(T)+' s:'+str(s)+' \n')

with gzip.open(filename, 'rb') as file:
    loaded_features = pickle.load(file)


def make_label_vectors(labels_list, labels_dict):
    y_all = np.zeros((len(labels_list), len(labels_dict)), dtype=np.uint32)
    for lbl in range(len(labels_list)):
        for indv_label in labels_list[lbl]:
            y_all[lbl][indv_label] = 1
    return y_all
    


data = loaded_features['featurized'].astype("uint32") 
lb= loaded_features['labels']
labeldict = loaded_features['labels_:_labelnum']
sorted_label_names = []
sorted_label_names = sorted(labeldict, key=labeldict.get)
y_all = make_label_vectors(lb, labeldict)

print(data.shape)
print(y_all.shape)

fo.write('\n Data Shape : '+str(data.shape)+' \nNumber of Labels: '+str(len(labeldict))+' \n')

average_accuracy = 0.0


x_train, x_test, y_train, y_test = train_test_split(data, y_all)
x_train_ids=x_train[:,-1]
x_test_ids=x_test[:,-1]
x_train=x_train[:,:-1]
x_test=x_test[:,:-1]


print(x_train.shape)
print(y_train.shape)

print(x_test.shape)
print(y_test.shape)

tm = MultiOutputTsetlinMachine(num_clauses, T, s, boost_true_positive_feedback=0)

for i in range(epochs):
	print('Epoch ', i, ' training ...' )
	tm.fit(x_train, y_train, epochs=train_epochs)

	prediction = tm.predict(x_test)

	print("Accuracy:", 100*(prediction == y_test).mean())

	average_accuracy += 100*(prediction == y_test).mean()

	print("Average Accuracy:", average_accuracy/(i+1))

	#print('\n Confusion Matrix ',multilabel_confusion_matrix(y_test, prediction, labels=sorted_label_names ))

	print('\n Classification Report:\n',classification_report(y_test, prediction, target_names = sorted_label_names ))

	if i%10 == 0:
		fo.write('\n Epoch '+str(i)+': Acc:'+str(100*(prediction == y_test).mean()) +' ' )

fo.close()   


fo = open(masteroutputfile, 'a+')
#running_file_name, input_file_name, result_filename, Number_samples_total, Number_labels, preprocess, TMType, Clauses, T, s, train_epochs, run_epochs,Avg_accuracy
fo.write(os.path.basename(__file__)+','+filename+','+outputfile+','+str(len(data))+','+str(len(labeldict))+',Simple BOW, Coalesced,'+str(num_clauses)+','+str(T)+','+str(s)+','+str(train_epochs)+','+str(epochs)+','+str(average_accuracy/100))+',NA\n'
fo.close()