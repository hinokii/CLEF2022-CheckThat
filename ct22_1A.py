'''
LING-582
Shared Task - CheckThat! Task 1
Subtask 1A File - ct22_1A.py
Name: Hinoki Crum (Hinoki's team)
Date: 05/01/2022
This file includes the code for Logistic Regression with TF-IDF and cost
sensitive approach which resulted in the best f1 score for the minority class
'''

from ct22_task1_base import *
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

train_df = ProcessData('CT22_english_1A_checkworthy_train.tsv',
                       'CT22_english_1A_checkworthy_dev.tsv')
training_labels= train_df.process_labels('class_label')
train_df.print_value_counts('class_label')
train_df.print_head()

val_df = ProcessData('CT22_english_1A_checkworthy_dev_test.tsv')
val_sent = val_df.process_text('tweet_text')
val_labels_final = val_df.process_labels('class_label')
val_df.print_value_counts('class_label')

test_df = ProcessData('CT22_english_1A_checkworthy_test.tsv')
test_sent = test_df.process_text('tweet_text')

# TF-IDF text representation applied
X = train_df.tfidf_vectorizer(val_sent, test_sent)
x = X[0]
val_x = X[1]
test_x = X[2]

#x, training_labels_final = train_df.oversample_smote(x, training_labels)
# best = best_model(MODELS, x, training_labels, val_x, val_labels_final, 'f1')
#cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
#gs = grid_search_model(model, param_grid_logistic, x, training_labels, cv)

# best parameters returned by grid search - C, penalty and solver = default
model = LogisticRegression(class_weight='balanced', C=1.0,
                                penalty='l2', solver='lbfgs')
model.fit(x, training_labels)
pred = model.predict(val_x)

f1 = f1_score(val_labels_final, pred)
print("F1 score - test: ", f1)
print(confusion_matrix(val_labels_final, pred))
print(classification_report(val_labels_final, pred))
#best_threshold(val_labels_final, pred)

# to create submission file with the predictions on the testing data
test_pred = model.predict(test_x)
outfile = test_df.df
del outfile['tweet_url']
del outfile['tweet_text']
outfile['class_label'] = test_pred
outfile['run_id'] = 'Model_1'
outfile.to_csv('subtaks1A_checkworthy_english.tsv', index=False, sep="\t")
