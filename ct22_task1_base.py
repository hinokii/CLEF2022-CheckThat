'''
CheckThat! Task 1
Base File - ct22_task1_base.py - each subtask file imports this base file to process data and train models
This is the base file which includes all the techniques that I have tried and described in my summary.
Please note that some of the functions and models are not exported into other files since
I only exported those which resulted in the best f1 scores or accuracy (the final versions only).
'''

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, StratifiedKFold, \
    RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import f1_score, accuracy_score, precision_recall_curve
from collections import Counter
import matplotlib.pyplot as plt
#nltk.download('omw-1.4')
stopwords = nltk.corpus.stopwords.words('english')
import warnings
warnings.filterwarnings("ignore")

class ProcessData:
    def __init__(self, train_file, val_file=None): #since the data is small and there are two validation sets provided,
        # I gave myself an option to combine one of validation sets with training set
        if val_file == None:
            self.df = pd.read_csv(train_file, error_bad_lines=False, sep="\t")
        else:
            self.df1 = pd.read_csv(train_file, error_bad_lines=False, sep="\t")
            self.df2 = pd.read_csv(val_file, error_bad_lines=False, sep="\t")
            self.df = pd.concat([self.df1, self.df2], axis=0)
    # print various info to analyze the data
    def print_head(self):
        print(self.df.head())

    def print_columns(self):
        print(self.df.columns)

    def print_value_counts(self, column):
        print(self.df[column].value_counts())

    def get_minority(self, value): # to get data from minority (or majority) class
        return self.df.loc[self.df['class_label'] == value, 'tweet_text']

    def get_lemm_text(self, texts): # to lemmatize, which I decided not to use
        tokens = word_tokenize(texts)
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens if not word \
                                                            in set(stopwords)]
        tokens = ' '.join(tokens)
        return tokens

    def process_text(self, text): #apply getLemmaText function, or just process (lower, remove stopwords & punctuation)
        self.df[text] = self.df[text].astype(str)
        self.df[text] = self.df[text].str.lower()
        #self.df[text] = self.df[text].str.replace('[#,@,&]', '')
        #self.df[text] = self.df[text].apply(lambda x: x.encode('ascii', 'ignore').decode('ascii'))
        #sentences = list(map(self.get_lemm_text, self.df[text]))
        sentences = self.df[text].apply(lambda x: ' '.join([word for word \
        in x.split() if word not in set(stopwords) and string.punctuation]))
        return sentences

    def bag_of_words(self, sent1, sent2):  # bag_of_word vectorizer
        vectorizer = CountVectorizer()
        sent = self.process_text('tweet_text')
        x = vectorizer.fit_transform(sent).toarray()
        val_x = vectorizer.transform(sent1).toarray()
        test_x = vectorizer.transform(sent2).toarray()
        return x, val_x, test_x

    def tfidf_vectorizer(self, sent1, sent2):  #tf-idf vectorizer
        vectorizer = TfidfVectorizer()
        sent = self.process_text('tweet_text')
        x = vectorizer.fit_transform(sent).toarray()
        val_x = vectorizer.transform(sent1).toarray()
        test_x = vectorizer.transform(sent2).toarray()
        return x, val_x, test_x

    # tokenize and sequence and pad for tensorflow
    def tokenize_padd(self, sentences, vocab_size, oov_tok, max_length,
                      padding_type, trunc_type):
        tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
        tokenizer.fit_on_texts(sentences)
        word_index = tokenizer.word_index
        sequences = tokenizer.texts_to_sequences(sentences)
        padded = pad_sequences(sequences, maxlen=max_length,
                               padding=padding_type,
                               truncating=trunc_type)
        return padded

    def process_labels(self, label):      # put labels into numpy array
        return np.array(self.df[label])

    # to oversample minority class using SMOTE
    def oversample_smote(self, padded, labels):
        oversample = SMOTE()
        oversample_padded, labels_final = oversample.fit_resample(padded,
                                                                  labels)
        counter = Counter(labels_final)
        print(counter)
        return oversample_padded, labels_final

    # to versample minority class using RandomOverSampler
    def oversample_random(self, padded,labels):
        oversample = RandomOverSampler()
        oversample_padded, labels_final = oversample.fit_resample(padded,
                                                                  labels)
        return oversample_padded, labels_final

    def val_data(self):  # to retrieve val data
        return self.df2

# define various sklearn models
MODELS = {'Logistic Regression': LogisticRegression(class_weight='balanced'),
          'SVM': SVC(class_weight='balanced'),
          'Random Forest': RandomForestClassifier(class_weight='balanced'),
          'Decision Tree': DecisionTreeClassifier(class_weight='balanced'),
          'Gradient Boosting':  GradientBoostingClassifier(),
          'Balanced Bagging': BalancedBaggingClassifier(base_estimator=\
                               DecisionTreeClassifier(),
                               sampling_strategy='not majority',
                               replacement=True)}

# to select best model
def best_model(models, train_x, train_y, test_x, test_y, measure):
    results = {}
    #kfold = KFold(n_splits=5) #Kfold
    for name, model in models.items():
        #cv_results = model_selection.cross_val_score(model, x, y, cv=kfold, scoring='f1')
        #results[name] = cv_results
        #cv_mean = np.mean(results.values())
        #print(results)
        model.fit(train_x, train_y)
        pred = model.predict(test_x)
        if measure == 'f1':
            print(f1_score(test_y, pred))
        elif measure == 'f1_weighted':
            print(f1_score(test_y, pred, average='weighted'))
        elif measure == 'accuracy':
            print(accuracy_score(test_y, pred))
        else:
            print('Please select f1, f1_weighted or accuracy as metric.')

def define_hud(dropout_rate, n1, n2, num_class, activation_func):
    hub_layer = hub.KerasLayer\
        ("https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1",
         trainable=True, input_shape=[], dtype=tf.string)
    model = tf.keras.Sequential()
    model.add(hub_layer)
    model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Dense(n1, activation='relu'))
    model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Dense(n2, activation='relu'))
    model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Dense(num_class, activation=activation_func))
    return model

# Global average pooling layer
def define_global_ave(dropout_rate, vocab_size, embedding_dim, max_length,
                      num_class, activation_func):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              input_length=max_length),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.GlobalAveragePooling1D(),

        tf.keras.layers.Dense(embedding_dim, activation='elu',
                              kernel_initializer='he_normal',
                              kernel_regularizer=tf.keras.regularizers.l2(
                                  0.01)),
        #tf.keras.layers.Dense(embedding_dim, activation=activation_func),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(num_class, activation=activation_func)])
    return model

# Conv1D + Global Max Pooling layer
def define_cnn(dropout_rate, vocab_size, embedding_dim, max_length, num_class,
               activation_func):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                  input_length=max_length),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Conv1D(embedding_dim, 5, activation='elu',
        kernel_initializer='he_normal',
        kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(num_class, activation=activation_func)])
    return model

# Bidirectional GRU
def define_gru(dropout_rate, vocab_size, embedding_dim, max_length, num_class,
               activation_func):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                  input_length=max_length),
        tf.keras.layers.BatchNormalization(),  # Batch Normalization
        #tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Bidirectional(tf.keras.layers.GRU(embedding_dim)),
        tf.keras.layers.BatchNormalization(),  # Batch Normalization
        #tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(embedding_dim, activation='elu',
                              kernel_initializer='he_normal',
                              kernel_regularizer=tf.keras.regularizers.l2(
                                  0.01)),
        tf.keras.layers.BatchNormalization(),  # Batch Normalization
        #tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(num_class, activation=activation_func)])
    return model

# Single Bidirectional LSTM
def define_lstm(dropout_rate, vocab_size, embedding_dim, num_class,
                activation_func):
    model = tf.keras.Sequential([
        # Add an Embedding layer
        tf.keras.layers.Embedding(vocab_size, num_class),
        tf.keras.layers.Dropout(rate=dropout_rate),
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(embedding_dim, dropout=dropout_rate)),
        # use ReLU for a hidden dense layer
        # tf.keras.layers.Dense(embedding_dim, activation='relu'),
        # tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Dense(embedding_dim, activation='relu'),
        tf.keras.layers.Dropout(rate=dropout_rate),
        # since we have multiple outputs, use softmax for the output layer.
        tf.keras.layers.Dense(num_class, activation=activation_func)])
    return model

# Multi Bidirectional LSTM
def define_multi_lstm(dropout_rate, vocab_size, embedding_dim, num_class,
                      activation_func):
    model = tf.keras.Sequential([
        # Add an Embedding layer expecting input vocab of size 1000 which outputs set of 128 vectors
        tf.keras.layers.Embedding(vocab_size, num_class),
        tf.keras.layers.Dropout(rate=dropout_rate),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim,
            return_sequences=True)),
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(embedding_dim, dropout=dropout_rate)),
        # use ReLU for a hidden dense layer
        # tf.keras.layers.Dense(embedding_dim, activation='relu'),
        # tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Dense(embedding_dim, activation='relu'),
        tf.keras.layers.Dropout(rate=dropout_rate),
        # since we have multiple outputs, use softmax for the output layer.
        tf.keras.layers.Dense(num_class, activation=activation_func)])
    return model

# graph the performance of model
def graph_plots(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.show()

# compute class weights for imbalanced data
def compute_class_weights(labels):
    n_samples = len(labels)
    n_classes = len(labels.unique())
    class_weights = {}
    class_names = labels.value_counts().index.tolist()
    for i in range(len(labels.value_counts())):
        class_weights[class_names[i]] = round(n_samples/\
                                (n_classes * labels.value_counts()[i]), 2)
    return class_weights

# to grid search best model
def grid_search_model(model, param_grid, x,y, cross_val=None):
    gs = GridSearchCV(estimator=model, param_grid=param_grid, scoring='f1',
                      verbose=1, n_jobs=-1, cv=cross_val)
    gs.fit(x, y)
    print("Best: %f using %s" % (gs.best_score_, gs.best_params_))
    return gs.best_estimator_

# define repeated stratified kfold
def repeated_st_kfold(model, x, y, n_split, n_repeat):
    cv = RepeatedStratifiedKFold(n_splits=n_split, n_repeats=n_repeat)
    scores = cross_val_score(model, x, y, scoring='f1', cv=cv, n_jobs=-1)
    return scores

# to grid search best class weight
def grid_search_class_weights(model, x,y):
    weights = np.linspace(0.0, 0.99, 200)
    param_grid = {'class_weight': [{0: x, 1: 1.0 - x} for x in weights]}
    gs = GridSearchCV(estimator=model,
                              param_grid=param_grid,
                              cv=StratifiedKFold(),
                              n_jobs=-1,
                              scoring='f1',
                              verbose=2).fit(x, y)
    print("Best: %f using %s" % (gs.best_score_, gs.best_params_))
    return gs.best_estimator_

def to_labels(pos_probs, threshold):
    return (pos_probs >= threshold).astype('int')

# check best threshold value
def best_threshold(y, y_pred, average=None):
    thresholds = np.arange(0,1, 0.001)
    scores = [f1_score(y, to_labels(y_pred, t), average=average) for t in thresholds]
    i = np.argmax(scores)
    print('Threshold=%.3f, F1-score=%.5f' % (thresholds[i], scores[i]))

# grid search parameters for SVM
param_grid_svc = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf']}

# grid search parameters for Logistic Regression
param_grid_logistic = {'solver': ['newton-cg', 'lbfgs', 'liblinear'],
                       'penalty': ['none', 'l1', 'l2', 'elasticnet'],
                       'C': np.logspace(-3,3,7)}

# grid search parameters for Random Forest
param_grid_random_tree = {'n_estimators': [int(x) for x in \
                np.linspace(start = 200, stop = 2000, num = 10)],
               'max_features': ['auto', 'sqrt'],
               'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)],
               'min_samples_split': [2, 5, 10],
               'min_samples_leaf': [1, 2, 4],
               'bootstrap': [True, False]}

# grid search parameters for Decision Tree
param_grid_desicion_tree = {
    'criterion': ['gini', 'entropy'],
    'max_depth': range(1,10),
    'min_samples_split': range(1,10),
    'min_samples_leaf': range(1,5)}

# grid search parameters for Gradient Boosting
param_grid_gradient_boost = {'n_estimators': [10, 50, 100, 500],
                'learning_rate': [0.0001, 0.001, 0.01, 0.1, 1.0],
                'subsample': [0.5, 0.7, 1.0],
                'max_depth': [3, 7, 9]}

# grid search parameters for Balanced Bagging Classifier
param_grid_bagging = {"base_estimator__max_depth": [3,5,10,20],
          "base_estimator__max_features": [None, "auto"],
          "base_estimator__min_samples_leaf": [1, 3, 5, 7, 10],
          "base_estimator__min_samples_split": [2, 5, 7],
          'bootstrap_features': [False, True],
          'max_features': [0.5, 0.7, 1.0],
          'max_samples': [0.5, 0.7, 1.0],
          'n_estimators': [2, 5, 10, 20]}

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
