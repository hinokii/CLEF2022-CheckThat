'''
CheckThat! Task 1
Subtask 1D File - ct22_1D.py
Although the bidirectional GRU provided the best f1 score, SVM with BoW text
representation using the class weight (cost sensitive approach) produced the
best f1 score with better predicted class distributions.
'''

from ct22_task1_base import *
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

class MultiProcessData(ProcessData):
    def __init__(self, csv_file):
        super().__init__(csv_file)

    def process_labels(self): # convert labels to integers
        self.df.loc[(self.df[
             "class_label"] == 'no_not_interesting'), "class_label"] = 0
        self.df.loc[(self.df["class_label"] == 'harmful'), "class_label"] = 1
        self.df.loc[(self.df[
             "class_label"] == 'yes_calls_for_action'), "class_label"] = 2
        self.df.loc[(self.df[
             "class_label"] == 'yes_blame_authorities'), "class_label"] = 3
        self.df.loc[(self.df[
             "class_label"] == 'yes_discusses_cure'), "class_label"] = 4
        self.df.loc[(self.df[
            "class_label"] == 'yes_discusses_action_taken'), "class_label"] = 5
        self.df.loc[
            (self.df["class_label"] == 'yes_asks_question'), "class_label"] = 6
        self.df.loc[(self.df[
              "class_label"] == 'yes_contains_advice'), "class_label"] = 7
        self.df.loc[
            (self.df["class_label"] == 'yes_other'), "class_label"] = 8
        return self.df['class_label']

    def labels_to_categorical(self, labels): # convert class labels to binary class matrix
        labels_final = tf.keras.utils.to_categorical(labels, num_classes=9)
        return labels_final
'''
# define various parameters
vocab_size = 1000
embedding_dim = 16
max_length = 40
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
'''
train_df = MultiProcessData('CT22_english_1D_attentionworthy_train.tsv')
train_df.print_value_counts('class_label')
#pad = train_df.tokenize_padd(sent, vocab_size, oov_tok, max_length, padding_type, trunc_type)

#val_df = MultiProcessData('CT22_english_1D_attentionworthy_dev.tsv')
#val_sent = val_df.process_text('tweet_text')
#val_padded = val_df.tokenize_padd(val_sent, vocab_size, oov_tok, max_length, padding_type, trunc_type)

val_df = MultiProcessData('CT22_english_1D_attentionworthy_dev_test.tsv')
val_sent = val_df.process_text('tweet_text')
#test_padded = test_df.tokenize_padd(test_sent, vocab_size, oov_tok, max_length, padding_type, trunc_type)

test_df = MultiProcessData('CT22_english_1D_attentionworthy_test.tsv')
test_sent = test_df.process_text('tweet_text')

X = train_df.bag_of_words(val_sent, test_sent)
x = X[0]
val_x = X[1]
test_x = X[2]

val_df.print_value_counts('class_label')

y = train_df.process_labels().tolist()
#training_labels_final = train_df.labels_to_categorical(y)

#x,y = train_df.oversample_random(x,y)
#print(padded.shape)
#val_labels = val_df.process_labels()
#val_labels_final = train_df.labels_to_categorical(val_labels)

val_labels = val_df.process_labels().tolist()
#best = best_model(MODELS, x, y, val_x, val_labels, 'f1_weighted')

model = SVC(class_weight='balanced')
model.fit(x, y)
pred = model.predict(val_x)

print(f1_score(val_labels, pred, average='weighted'))
print(classification_report(val_labels, pred))
print(confusion_matrix(val_labels, pred))

# to create submission file with the predictions on the testing data
test_pred = model.predict(test_x)
outfile = test_df.df
del outfile['tweet_url']
del outfile['tweet_text']
outfile['class_label'] = test_pred
outfile['run_id'] = 'Model_1'
outfile.loc[(outfile['class_label'] == 0), 'class_label'] = 'no_not_interesting'
outfile.loc[(outfile['class_label'] == 1), 'class_label'] = 'harmful'
outfile.loc[(outfile['class_label'] == 2), 'class_label'] \
    = 'yes_calls_for_action'
outfile.loc[(outfile['class_label'] == 3), 'class_label'] \
    = 'yes_blame_authorities'
outfile.loc[(outfile['class_label'] == 4), 'class_label'] \
    = 'yes_discusses_cure'
outfile.loc[(outfile['class_label'] == 5), 'class_label'] \
    = 'yes_discusses_action_taken'
outfile.loc[(outfile['class_label'] == 6), 'class_label'] = 'yes_asks_question'
outfile.loc[(outfile['class_label'] == 7), 'class_label'] \
    = 'yes_contains_advice'
outfile.loc[(outfile['class_label'] == 8), 'class_label'] = 'yes_other'
outfile.to_csv('subtaks1D_attentionworthy_english.tsv', index=False, sep="\t")
'''
model = define_gru(0.2, vocab_size, embedding_dim, max_length, 9, 'softmax')
model.compile(loss='categorical_crossentropy', \
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              metrics='accuracy')

class_weights = compute_class_weights(y)
print(class_weights)

callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
num_epochs = 100
history = model.fit(pad, training_labels_final, epochs=num_epochs, validation_data=(val_padded, val_labels_final),
                 callbacks=callback)

graph_plots(history, "accuracy")
graph_plots(history, "loss")
y_pred = np.argmax(model.predict(test_padded), axis=1)
print(model.evaluate(test_padded, test_labels))
rounded_labels=np.argmax(test_labels_final, axis=1)
print("F1: ", f1_score(y_pred, rounded_labels, average='weighted'))
print(confusion_matrix(rounded_labels, y_pred))
print(classification_report(rounded_labels, y_pred))
'''
