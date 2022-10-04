'''
LING-582
Shared Task - CheckThat! Task 1
Draft Subtask 1B File - ct22_1B.py
Name: Hinoki Crum (Hinoki's team)
Date: 05/01/2022
This file includes the code for Transfer Learning with TensorFlow Hud, which provided the accuracy score.
'''

from ct22_task1_base import *

train_df = ProcessData('CT22_english_1B_claim_train.tsv',
                       'CT22_english_1B_claim_dev.tsv')

val_df = ProcessData('CT22_english_1B_claim_dev_test.tsv')

test_df = ProcessData('CT22_english_1B_claim_test.tsv')

y = train_df.process_labels('class_label')

model = define_hud(0.3, 128, 16, 1, 'sigmoid')
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)
model.summary()
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy']
              )
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
num_epochs = 100
history = model.fit(train_df.df['tweet_text'], train_df.df['class_label'],
            epochs=num_epochs,
            validation_data=(val_df.df['tweet_text'], val_df.df['class_label']),
            callbacks=callback,
            verbose=1)

graph_plots(history, "accuracy")
graph_plots(history, "loss")
model.evaluate(val_df.df['tweet_text'], val_df.df['class_label'])

# to create submission file with the predictions on the testing data
test_pred = np.round(model.predict(test_df.df['tweet_text'])).astype(int)
outfile = test_df.df
del outfile['tweet_url']
del outfile['tweet_text']
outfile['class_label'] = test_pred
outfile['run_id'] = 'Model_1'
outfile.to_csv('subtaks1B_claim_english.tsv', index=False, sep="\t")
