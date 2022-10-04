'''
CheckThat! Task 1
Subtask 1C File - ct22_1C.py
This file includes the code for Logistic Regression with TF-IDF and cost
sensitive approach which resulted in the best f1 score for the minority class
'''

from ct22_task1_base import *
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

train_df = ProcessData('CT22_english_1C_harmful_train.tsv',
                       'CT22_english_1C_harmful_dev.tsv')
train_df.print_value_counts('class_label')

training_labels_final = train_df.process_labels('class_label')
'''
#DATA AUGMENTATION - I didn't end up doing since this did not improve the performance.
import nlpaug.augmenter.word as naw

minority_text = train_df.get_minority(1)
aug = naw.SynonymAug(aug_src='wordnet', aug_max=3)

sent1 = [aug.augment(i,n=2)[0] for i in minority_text]
sent2 = [aug.augment(i,n=2)[1] for i in minority_text]

sent = np.concatenate((sent, sent1, sent2), axis=0)
print(len(sent))
new_labels =[1 for i in range(323*2)]
training_labels_final = np.concatenate((training_labels_final, new_labels), axis=0)
print(len(training_labels_final))
'''

val_df = ProcessData('CT22_english_1C_harmful_dev_test.tsv')
val_sent = val_df.process_text('tweet_text')
val_labels_final = val_df.process_labels('class_label')

test_df = ProcessData('CT22_english_1C_harmful_test.tsv')
test_sent = test_df.process_text('tweet_text')

#X = train_df.bag_of_words(val_sent, test_sent)
X = train_df.tfidf_vectorizer(val_sent, test_sent)
x = X[0]
val_x = X[1]
test_x = X[2]

#Best parameters for LG selected by grid search - solver & penalty = default
model = LogisticRegression(solver='lbfgs', class_weight='balanced',
                           C=0.001, penalty='l2')
model.fit(x, training_labels_final)
pred = model.predict(val_x)
f1 = f1_score(val_labels_final, pred, average=None)
print("F1 score: ", f1)
print(classification_report(val_labels_final, pred))
print(confusion_matrix(val_labels_final, pred))

# to create submission file with the predictions on the testing data
test_pred =  model.predict(test_x)
outfile = test_df.df
del outfile['tweet_url']
del outfile['tweet_text']
outfile['class_label'] = test_pred
outfile['run_id'] = 'Model_1'
outfile.to_csv('subtaks1C_harmful_english.tsv', index=False, sep="\t")
