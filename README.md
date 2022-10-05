# CLEF2022 CheckThat Lab Task 1 

## Task

Task 1 focuses on disinformation related to the ongoing COVID-19 infodemic and asks to identify which posts in a Twitter stream are worth fact-checking, contain a verifiable factual claim, are harmful to the society, and why. It consists of the following four subtasks. 

* Subtask 1A - Binary classification to predict whether it is worth fact-checking (yes or no).
* Subtask 1B - Binary classification to predict whether it contains a verifiable factual claim (yes or no). 
* Subtask 1C - Binary classification to predict whether it is harmful to the society and why (yes or no).
* Subtask 1D - Multi-class classification to predict whether it should get the attention of policy makers and why. There are nine class labels: (i) No, not interesting, ((ii) Yes, asks question, (iii) Yes, blame authorities, (iv) Yes, calls for action, (v) Yes, classified as in harmful task, (vi) Yes, contains advice, (vii) Yes, discusses action taken, (viii) Yes, discusses cure, and (ix) Yes, other. 

Subtask 1A and 1C are evaluated based on the F1 measure with respect to the positive class (minority class). Subtask 1B is evaluated based on the accuracy and subtask 1D is evaluated based on the weighted F1. 

## Approach

Since there are four subtasks, I first created the base Python file with the following techniques:

* Bag of Words (BoW)
* TF-IDF
* Preparation with TensorFlow and Keras
* Sklearn models - the models analyzed are Logistic Regression, Support Vector Machines (SVM), Random Forest, Decision Tree, Gradient Boosting and Balanced Bagging Classifier
* Dense layers with transfer learning - the first layer is the TensorFlow Hub layer followed by dropout layer and two dense layers and dropout layers after every dense layer. The last layer is densely connected with a single output node.
* Dense layers with Global Average Pooling layer
* Convolution 1D layer 
* Bidirectional LSTM (Long-Short Term Memory) 
* Multilayer Bidirectional LSTM   
* Bidirectional GRU - Bidirectional GRU (Gated Recurrent Unit) 

I noted all four data from subtasks is imbalanced. Especially, the data for Subtask 1C and 1D is heavily imbalanced as summarized below. 

| Subtask | Class 0 | Class 1 | Class 2 | Class 3 | Class 4 | Class 5 | Class 6 | Class 7 | Class 8 |
| ------- | -------: | -------: | -------: | -------: | -------: | -------: | -------: | -------: | -------: |
| Subtask 1A | 1,826 |   491 |
| Subtask 1B | 2,122 | 1,202 |
| Subtask 1C | 3,307 |   323 |
| Subtask 1D | 2,851 |   173 | 138 | 48 | 42 | 27 | 25 | 12 | 5 |
 
To handle the small imbalanced data, I have considered the following techniques:

1. Oversampling using SMOTE (Synthetic Minority Over-sampling technique) 
2. Oversampling using RandomOverSampling
3. Class weights in the models (cost-sensitive learning)
4. Threshold adjusting
5. Text augmentation using nlpaug library 
6. Lemmatization
7. K-fold
8. Pre-trained models for transfer learning
[Link](https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1)
9. For dying ReLU, Leaky ReLU, use of the ELU with kernel initializer he normal, Batch Normalization (BN), Gradient Clipping

## Results

**Subtask 1A**

The best combination for subtask 1A is Logistic Regression with TF-IDF text representation using the class weight, which resulted in the f1 score of 56.2% for the minority class. 

The f1 scores for the minority class are summarized below:

| Approaches | Cost-Sensitive| Cost-Sensitive | Oversampling | Oversampling |
| ---------- | :-----------: | :------------: | :----------: | :----------: |
| **Text Repr** | **BoW** | **TF-IDF** | **BoW** | **TF-IDF** | 
| Logistic Regression | 52.21% |  **56.18%** | 50.00% | 55.34% |
| SVM | 53.44% |   41.45% | 32.12% | 21.19% |
| Decision Tree | 46.89% | 46.49% | 44.66% | 42.52% |
| Random Forest | 28.05% | 28.75% | 49.82% | 32.93% |
| Gradient Boosting | 28.57% | 31.71% | 44.93% | 46.30% |
| Balanced Bagging | 31.14% | 21.94% | 41.78% | 38.83% |
 
**Subtask 1B**

For this subtask, the transfer learning with TensorFlow Hub helped to improve the performance. With the first and the second dense layers with 128 neurons and 16 neurons, respectively and dropout layers at the dropout rate of 30%, I obtained the accuracy of 78.1% on the validation set. 

The accuracy scores are summarized below:

| Approaches | Cost-Sensitive| Cost-Sensitive | Oversampling | Oversampling |
| ---------- | :-----------: | :------------: | :----------: | :----------: |
| TF Hub     | **78.08%** |          | N/A |  
| **Text Repr** | **BoW** | **TF-IDF** | **BoW** | **TF-IDF** | 
| Logistic Regression | 70.91% |   71.57% | 71.90% | 72.67% |
| SVM | 71.45% |   73.66% | 72.12% | 70.91% |
| Decision Tree | 68.72% | 64.43% | 63.45% | 63.23% |
| Random Forest | 73.33% | 74.09% | 73.00% | 72.67% |
| Gradient Boosting | 66.19% | 67.95% | 68.83% | 68.50% |
| Balanced Bagging | 68.39% | 69.04% | 64.00% | 66.85% |

**Subtask 1C**

The best combination for subtask 1C is Logistic Regression with TF-IDF vectorization and cost-sensitive approach, which resulted in the f1 score of 37.0% for the minority class. This is the heavily imbalanced data with an imbalance ratio of 10.24 (3,307/323), so very difficult to handle. Generally, the higher the degree of imbalance is, the higher the error is. I applied text augmentation as described above but I could not obtain any improved performance. 

**Subtask 1D**

Overall, all the models produced relatively high f1 scores for this subtask. Since this subtask is evaluated based on the weighted average of all per-class f1 scores rather than the f1 score for the minority class, they may have resulted in the relatively high f1 scores. The best combination for subtask 1D seemed Bidirectional GRU with cost-sensitive approach, which showed the f1 score of 81.6%. However, I checked the f1 score for each class and noted that the model predicts only class 0 and 1 (majority classes) and none for the rest of the classes. Although the predictions were heavily biased towards the majority classes, since the majority of the testing data consists of the class 0 and class 1, it appears to have resulted in the high f1 score when averaged. 

SVM with BoW text representation using the class weight produced the f1 score of 81.3% and based on the review of the per-class f1 score, SVM does not seem as biased as Bidirectional GRU. As SVM provides the better predicted distribution of classes, I assumed that SVM is a better model than Bidirectional GRU for this case. 

The weighted f1 scores are summarized below:

| Approaches | Cost-Sensitive| Cost-Sensitive | Oversampling | Oversampling |
| ---------- | :-----------: | :------------: | :----------: | :----------: |
| Bidirectional GRU    | 45.34% | | 81.59% |
| **Text Repr** | **BoW** | **TF-IDF** | **BoW** | **TF-IDF** | 
| SVM | **81.29%** |   80.26% | 80.44% | 79.35% |
| Decision Tree | 77.23% | 76.30% | 75.13% | 76.44% |
| Random Forest | 78.36% | 78.57% | 79.43% | 79.19% |
| Gradient Boosting | 80.24% | 78.32% | 75.94% | 77.45% |
| Balanced Bagging | 78.32% | 80.26% | 77.78% | 78.69% |
