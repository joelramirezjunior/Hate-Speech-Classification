---
jupyter:
  colab:
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.8.5
  nbformat: 4
  nbformat_minor: 0
---

::: {.cell .markdown id="eRgJhlEStGpd"}
# CS224U Final Project

First necessary imports and datasets
:::

::: {.cell .code id="ZR2Ix2qctGpe"}
``` python
from collections import Counter
import numpy as np
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from torch_shallow_neural_classifier import TorchShallowNeuralClassifier
from sklearn.feature_extraction.text import TfidfTransformer
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from torch_rnn_classifier import a
from torch_tree_nn import TorchTreeNN
import sst
import re
import utils
```
:::

::: {.cell .code id="ykrucYqFtGpf"}
``` python
HateSpeech_HOME = os.path.join('CS224U-Final-Project', 'Data')
GLOVE_HOME = os.path.join('data', 'glove.6B')
glove_lookup = utils.glove2dict(
    os.path.join(GLOVE_HOME, 'glove.6B.300d.txt'))
```
:::

::: {.cell .code id="2kTHzpTPtGpf"}
``` python
def preprocess(text_string):
    """
    Accepts a text string and replaces:
    1) urls with URLHERE
    2) lots of whitespace with one instance
    3) mentions with MENTIONHERE

    This allows us to get standardized counts of urls and mentions
    Without caring about specific people mentioned
    """
    space_pattern = '\s+'
    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
        '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+'
    parsed_text = re.sub(space_pattern, ' ', text_string)
    parsed_text = re.sub(giant_url_regex, '', parsed_text)
    parsed_text = re.sub(mention_regex, '', parsed_text)
    #parsed_text = parsed_text.code("utf-8", errors='ignore')
    return parsed_text
```
:::

::: {.cell .code id="IQoAfhkRtGpg"}
``` python
def dataset_reader():
    """
    Iterator for reading in dataset adapted by code provided by the generous CS224U teaching staff

    Parameters
    ----------
    src_filename : str
        Full path to the file to be read.


    dedup : bool
        If True, only one copy of each (text, tabel) pair is included.
        This mainly affects the train set, though there is one repeated
        example in the dev set.

    Yields
    ------
    pd.DataFrame with columns ['example_id', 'text', 'hate_speech', label

    """
    src_filename = os.path.join(HateSpeech_HOME, 'MasterDataSet.csv')
    df = pd.read_csv(src_filename)
    return df
```
:::

::: {.cell .code id="hlwJbi2ZtGpg"}
``` python
hateSpeechDev = dataset_reader()
```
:::

::: {.cell .code id="KymS-HLLtGpg" scrolled="true"}
``` python
hateSpeechDev.sample(3, random_state=1).to_dict(orient='records')
```
:::

::: {.cell .markdown id="fn-0EmWVtGph"}
Here are the label counts
:::

::: {.cell .code id="Xo690i1WtGph" outputId="7be72e27-dc6d-4327-ada4-e3c7757dcfb8"}
``` python
hateSpeechDev.label.value_counts()
```

::: {.output .execute_result execution_count="7"}
    neutral    22476
    hate       11680
    Name: label, dtype: int64
:::
:::

::: {.cell .code id="fe2ySHTmtGph"}
``` python
train, test = train_test_split(hateSpeechDev, test_size=0.7)
test_small, test_large = train_test_split(test, test_size = .9)
```
:::

::: {.cell .code id="qhOdQv3UtGph" outputId="af0d1988-da29-4c42-e28d-956b5dc8b1cf"}
``` python
train.label.value_counts()
```

::: {.output .execute_result execution_count="9"}
    neutral    6789
    hate       3457
    Name: label, dtype: int64
:::
:::

::: {.cell .code id="O8Ga79hxtGph" outputId="9eaac070-e82a-4e82-a1e2-aaa902249470"}
``` python
test.label.value_counts()
```

::: {.output .execute_result execution_count="11"}
    neutral    15687
    hate        8223
    Name: label, dtype: int64
:::
:::

::: {.cell .markdown id="PiSXYG8XtGph"}
Simple Baseline: Linear Softmax Classifier with Unigrams & Bigrams
:::

::: {.cell .markdown id="v-XmZa6RtGpi"}
## Unigrams
:::

::: {.cell .code id="STHVJBzFtGpi"}
``` python
def unigrams_phi(text):
    """
    The basis for a unigrams feature function. Downcases all tokens.

    Parameters
    ----------
    text : str
        The example to represent.

    Returns
    -------
    defaultdict
        A map from strings to their counts in `tree`. (Counter maps a
        list to a dict of counts of the elements in that list.)

    """
    text = preprocess(text)
    return Counter(text.lower().split())
```
:::

::: {.cell .markdown id="tXcyp7f6tGpi"}
## Bigrams
:::

::: {.cell .code id="urZeJKA_tGpi"}
``` python
def bigrams_phi(text):
    """
    The basis for a bigrams feature function. Downcases all tokens.

    Parameters
    ----------
    text : str
        The example to represent.

    Returns
    -------
    defaultdict
        A map from tuples to their counts in `text`.

    """
    text = preprocess(text)
    toks = text.lower().split()
    left = [utils.START_SYMBOL] + toks
    right = toks + [utils.END_SYMBOL]
    grams = list(zip(left, right))
    return Counter(grams)
```
:::

::: {.cell .markdown id="2irfwQArtGpi"}
Joining both Uni and bi grams!
:::

::: {.cell .code id="IUDT_jqwtGpi"}
``` python
def uni_and_bigrams_phi(text):

    return unigrams_phi(text) + bigrams_phi(text)
```
:::

::: {.cell .code id="6seYStBVtGpi"}
``` python
def fit_softmax_with_hyperparameter_search(X, y):
    """
    A MaxEnt model of dataset with hyperparameter cross-validation.

    Some notes:

    * 'fit_intercept': whether to include the class bias feature.
    * 'C': weight for the regularization term (smaller is more regularized).
    * 'penalty': type of regularization -- roughly, 'l1' ecourages small
      sparse models, and 'l2' encourages the weights to conform to a
      gaussian prior distribution.
    * 'class_weight': 'balanced' adjusts the weights to simulate a
      balanced class distribution, whereas None makes no adjustment.

    Other arguments can be cross-validated; see
    http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

    Parameters
    ----------
    X : 2d np.array
        The matrix of features, one example per row.

    y : list
        The list of labels for rows in `X`.

    Returns
    -------
    sklearn.linear_model.LogisticRegression
        A trained model instance, the best model found.

    """
    basemod = LogisticRegression(
        fit_intercept=True,
        solver='liblinear',
        multi_class='auto')
    cv = 5
    param_grid = {
        'C': [0.8, 1.0],
        'penalty': ['l1', 'l2'],
        'class_weight': ['balanced', None]}
    bestmod = utils.fit_classifier_with_hyperparameter_search(
        X, y, basemod, cv, param_grid)
    return bestmod
```
:::

::: {.cell .code id="YXr3gelJtGpi"}
``` python
# softmax_experiment_uni = sst.experiment(
#     train,
#     unigrams_phi,
#     fit_softmax_with_hyperparameter_search,
#     assess_dataframes=[test])
```
:::

::: {.cell .code id="8jvCtKQ9tGpj"}
``` python
# softmax_experiment_bi = sst.experiment(
#     train,
#     bigrams_phi,
#     fit_softmax_with_hyperparameter_search,
#     assess_dataframes=[test])
```
:::

::: {.cell .code id="0zKbwRMjtGpj" outputId="ffecc1c3-ecfc-4c55-f51a-f8e6d5874622"}
``` python
softmax_experiment_uni_bi = sst.experiment(
    train,
    bigrams_phi,
    fit_softmax_with_hyperparameter_search,
    assess_dataframes=[test])
```

::: {.output .stream .stdout}
    Best params: {'C': 1.0, 'class_weight': 'balanced', 'penalty': 'l2'}
    Best score: 0.751
                  precision    recall  f1-score   support

            hate      0.713     0.604     0.654     10463
         neutral      0.811     0.875     0.841     20278

        accuracy                          0.782     30741
       macro avg      0.762     0.739     0.748     30741
    weighted avg      0.777     0.782     0.778     30741
:::
:::

::: {.cell .markdown id="-4oQYssXtGpj"}
# RNN Optimization
:::

::: {.cell .markdown id="9jQJ5qHitGpj"}
### RNN phi
:::

::: {.cell .code id="szJAmuzNtGpj"}
``` python
def rnn_phi(text):
    text = preprocess(text)
    return text.split()
```
:::

::: {.cell .markdown id="IWMMa6SFtGpj"}
### VSM Phi
:::

::: {.cell .code id="9XelWk6ntGpj"}
``` python
def vsm_phi(text, lookup, np_func=np.mean):
    """Represent `tree` as a combination of the vector of its words.

    Parameters
    ----------
    text : str

    lookup : dict
        From words to vectors.

    np_func : function (default: np.sum)
        A numpy matrix operation that can be applied columnwise,
        like `np.mean`, `np.sum`, or `np.prod`. The requirement is that
        the function take `axis=0` as one of its arguments (to ensure
        columnwise combination) and that it return a vector of a
        fixed length, no matter what the size of the tree is.

    Returns
    -------
    np.array, dimension `X.shape[1]`

    """
    text = preprocess(text)
    allvecs = np.array([lookup[w] for w in text.split() if w in lookup])
    if len(allvecs) == 0:
        dim = len(next(iter(lookup.values())))
        feats = np.zeros(dim)
    else:
        feats = np_func(allvecs, axis=0)
    return feats

def glove_phi(text, np_func=np.mean):
    return vsm_phi(text, glove_lookup, np_func=np_func)
```
:::

::: {.cell .code id="TAJTrpKptGpj"}
``` python
def fit_rnn_with_hyperparameter_search(X, y):
    sst_train_vocab = utils.get_vocab(X, mincount=2)
    glove_embedding, sst_glove_vocab = utils.create_pretrained_embedding(glove_lookup, sst_train_vocab)
    basemod = TorchRNNClassifier(
        sst_glove_vocab,
        embedding=glove_embedding,
        batch_size=25,
        bidirectional=True,
        early_stopping=True)

    # There are lots of other parameters and values we could
    # explore, but this is at least a solid start:
    param_grid = {
        'embed_dim': [75, 100],
        'hidden_dim': [75, 100],
        'eta': [0.001, 0.01]}

    bestmod = utils.fit_classifier_with_hyperparameter_search(
        X, y, basemod, cv=3, param_grid=param_grid)

    return bestmod
```
:::

::: {.cell .code id="TnJ5ZTFAtGpj" outputId="f06c3af6-06a4-4577-8ec7-b4cc23b3a3a0"}
``` python
rnn_experiment = sst.experiment(
    train,
    rnn_phi,
    fit_rnn_with_hyperparameter_search,
    vectorize=False,  # For deep learning, use `vectorize=False`.
    assess_dataframes=[test])
```

::: {.output .stream .stderr}
    Stopping after epoch 12. Validation score did not improve by tol=1e-05 for more than 10 epochs. Final error is 0.53592498919124415
:::

::: {.output .stream .stdout}
    Best params: {'embed_dim': 100, 'eta': 0.01, 'hidden_dim': 75}
    Best score: 0.766
                  precision    recall  f1-score   support

            hate      0.721     0.674     0.697     10463
         neutral      0.837     0.865     0.851     20278

        accuracy                          0.800     30741
       macro avg      0.779     0.770     0.774     30741
    weighted avg      0.798     0.800     0.799     30741
:::
:::

::: {.cell .code id="ERwOcos8tGpj"}
``` python
rnn_experiment_hyperparams = rnn_experiment
```
:::

::: {.cell .code id="tSFC2coCtGpk"}
``` python
# rnn_experiment = sst.experiment(
#     train,
#     vsm_phi,
#     fit_rnn_with_hyperparameter_search,
#     vectorize=False,  # For deep learning, use `vectorize=False`.
#     assess_dataframes=[test])
```
:::

::: {.cell .markdown id="3HLBKZh1tGpk"}
# Shallow Neural Network
:::

::: {.cell .code id="0HdSP5-5tGpk"}
``` python
def fit_shallow_neural_classifier_with_hyperparameter_search(X, y):
    basemod = TorchShallowNeuralClassifier(
        early_stopping=True
        )
    cv = 3
    param_grid = {
        'hidden_dim': [50,100,200],
        'hidden_activation': [nn.ReLU(), nn.Tanh()],
        }
    bestmod = utils.fit_classifier_with_hyperparameter_search(X, y, basemod, cv, param_grid)
    return bestmod
```
:::

::: {.cell .code id="I0SV4UW_tGpk" outputId="10527278-1875-459d-d055-eb469db2dabc"}
``` python
torch_shallow_neural_experiment = sst.experiment(
    train,
    bigrams_phi,
    fit_shallow_neural_classifier_with_hyperparameter_search,
    assess_dataframes=[test])
```

::: {.output .stream .stderr}
    Stopping after epoch 21. Validation score did not improve by tol=1e-05 for more than 10 epochs. Final error is 0.127093225717544567
:::

::: {.output .stream .stdout}
    Best params: {'hidden_activation': ReLU(), 'hidden_dim': 100}
    Best score: 0.710
                  precision    recall  f1-score   support

            hate      0.722     0.479     0.576     10463
         neutral      0.771     0.905     0.833     20278

        accuracy                          0.760     30741
       macro avg      0.746     0.692     0.704     30741
    weighted avg      0.754     0.760     0.745     30741
:::
:::

::: {.cell .code id="UM3Bqm_ptGpk" outputId="d6d7697d-a1bd-444a-a4ce-5a03bfd12b38"}
``` python
softmax_linearRegression_glove_phi = sst.experiment(train,
                                                glove_phi,
                                                fit_softmax_with_hyperparameter_search,
                                                assess_dataframes=test,
                                                vectorize=False)
```

::: {.output .stream .stdout}
    Best params: {'C': 0.8, 'class_weight': 'balanced', 'penalty': 'l2'}
    Best score: 0.731
                  precision    recall  f1-score   support

            hate      0.584     0.710     0.641     10463
         neutral      0.832     0.739     0.783     20278

        accuracy                          0.729     30741
       macro avg      0.708     0.725     0.712     30741
    weighted avg      0.747     0.729     0.735     30741
:::
:::

::: {.cell .markdown id="ieXc7OwKtGpk"}
# Uni and Bigram NaiveBayes
:::

::: {.cell .code id="FoaPLrmTtGpk"}
``` python
from sklearn.pipeline import Pipeline
def fit_nb_classifier_with_hyperparameter_search(X, y):
    rescaler = TfidfTransformer()
    mod = MultinomialNB()

    pipeline = Pipeline([('scaler', rescaler), ('model', mod)])

    # Access the alpha and fit_prior parameters of `mod` with
    # `model__alpha` and `model__fit_prior`, where "model" is the
    # name from the Pipeline. Use 'passthrough' to optionally
    # skip TF-IDF.
    param_grid = {
        'model__fit_prior': [True, False],
        'scaler': ['passthrough', rescaler],
        'model__alpha': [0.1, 0.2, 0.4, 0.8, 1.0, 1.2]}

    bestmod = utils.fit_classifier_with_hyperparameter_search(
        X, y, pipeline,
        param_grid=param_grid,
        cv=5)
    return bestmod
```
:::

::: {.cell .code id="beLuCDwdtGpk" outputId="53347d16-6929-4864-e161-e81470bf9b68"}
``` python
unigram_nb_experiment_xval = sst.experiment(
    train,
    bigrams_phi,
    fit_nb_classifier_with_hyperparameter_search,
    assess_dataframes=test)
```

::: {.output .stream .stdout}
    Best params: {'model__alpha': 1.2, 'model__fit_prior': True, 'scaler': 'passthrough'}
    Best score: 0.739
                  precision    recall  f1-score   support

            hate      0.684     0.571     0.623     10463
         neutral      0.796     0.864     0.829     20278

        accuracy                          0.764     30741
       macro avg      0.740     0.718     0.726     30741
    weighted avg      0.758     0.764     0.759     30741
:::
:::

::: {.cell .markdown id="qIG0bxGYtGpx"}
# SVM Results
:::

::: {.cell .code id="Y0NqdfoftGpx"}
``` python
from sklearn.svm import LinearSVC
def fit_svm_classifier_with_hyperparameter_search(X, y):
    rescaler = TfidfTransformer()
    mod = LinearSVC(loss='squared_hinge', penalty='l2')

    pipeline = Pipeline([('scaler', rescaler), ('model', mod)])

    # Access the alpha parameter of `mod` with `mod__alpha`,
    # where "model" is the name from the Pipeline. Use
    # 'passthrough' to optionally skip TF-IDF.
    param_grid = {
        'scaler': ['passthrough', rescaler],
        'model__C': [0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]}

    bestmod = utils.fit_classifier_with_hyperparameter_search(
        X, y, pipeline,
        param_grid=param_grid,
        cv=5)

    return bestmod
```
:::

::: {.cell .code id="sadCQkurtGpx" outputId="3cc70951-c60e-4ed9-b120-9123b58f2b81"}
``` python
svm_experiment_xval = sst.experiment(
    train,
    bigrams_phi,
    fit_svm_classifier_with_hyperparameter_search,
    assess_dataframes=test_small)
```

::: {.output .stream .stdout}
    Best params: {'model__C': 0.1, 'scaler': 'passthrough'}
    Best score: 0.753
                  precision    recall  f1-score   support

            hate      0.732     0.568     0.640      1146
         neutral      0.790     0.887     0.835      2098

        accuracy                          0.774      3244
       macro avg      0.761     0.727     0.738      3244
    weighted avg      0.769     0.774     0.766      3244
:::
:::

::: {.cell .markdown id="ysXuSOMLtGpy"}
# Getting Optimal Models
:::

::: {.cell .code id="Mb9IEN7CtGpy"}
``` python
softmax_experiment_optimal = softmax_experiment_uni_bi['model']
rnn_experiment_optimal = rnn_experiment_hyperparams['model']
shallow_experiment_optimal = torch_shallow_neural_experiment['model']
nb_experiment_optimal = unigram_nb_experiment_xval['model']
svm_experiment_optimal = svm_experiment_xval['model']

softmax_experiment_optimal_data = softmax_experiment_uni_bi['assess_datasets']
rnn_experiment_optimal_data = rnn_experiment_hyperparams['assess_datasets']
shallow_experiment_optimal_data = torch_shallow_neural_experiment['assess_datasets']
nb_experiment_optimal_data = unigram_nb_experiment_xval['assess_datasets']
svm_experiment_optimal_data = svm_experiment_xval['assess_datasets']

del softmax_experiment_uni_bi
del rnn_experiment_hyperparams
del torch_shallow_neural_experiment
del unigram_nb_experiment_xval
del svm_experiment_xval
```
:::

::: {.cell .code id="-G69bTnvtGpy" outputId="3d95419b-1b76-4920-d1f5-ffadfa86fd13"}
``` python
def fit_optimized_softmax(X, y):
    basemod = LogisticRegression(
        C = 1,
        class_weight = 'balanced',
        penalty = "l2",
        fit_intercept=True,
        solver='liblinear',
        multi_class='auto')
    basemod.fit(X, y)
    return basemod

softmax_rerun = sst.experiment(
    train,
    bigrams_phi,
    fit_optimized_softmax,
    assess_dataframes=test_small)
```

::: {.output .stream .stdout}
                  precision    recall  f1-score   support

            hate      0.754     0.625     0.683       861
         neutral      0.807     0.885     0.844      1530

        accuracy                          0.791      2391
       macro avg      0.780     0.755     0.764      2391
    weighted avg      0.788     0.791     0.786      2391
:::
:::

::: {.cell .code id="SflDZAUytGpy" outputId="1e2e363a-a4aa-48fc-e623-dae07eafb19c" scrolled="true"}
``` python
def fit_rnn_optimized(X, y):
    sst_train_vocab = utils.get_vocab(X, mincount=2)
    glove_embedding, sst_glove_vocab = utils.create_pretrained_embedding(glove_lookup, sst_train_vocab)
    basemod = TorchRNNClassifier(
        sst_glove_vocab,
        embedding=glove_embedding,
        embed_dim = 100,
        eta = .01,
        hidden_dim = 75,
        batch_size=25,
        bidirectional=True,
        early_stopping=True)
    basemod.fit(X, y)
    return basemod

rnn_rerun = sst.experiment(
    train,
    rnn_phi,
    fit_rnn_optimized,
    vectorize = False,
    assess_dataframes = test_small
    )
```

::: {.output .stream .stderr}
    Stopping after epoch 12. Validation score did not improve by tol=1e-05 for more than 10 epochs. Final error is 11.27336708268922
:::

::: {.output .stream .stdout}
                  precision    recall  f1-score   support

            hate      0.867     0.619     0.722       861
         neutral      0.815     0.946     0.876      1530

        accuracy                          0.829      2391
       macro avg      0.841     0.783     0.799      2391
    weighted avg      0.834     0.829     0.821      2391
:::
:::

::: {.cell .code id="nWvKLRt1tGpy"}
``` python
train_x, test_x = train_test_split(hateSpeechDev, test_size=0.85)
test_y, test_large_y = train_test_split(test_x, test_size = .9)
```
:::

::: {.cell .code id="emOhjV3GtGpy" outputId="2a2fdff2-7105-4d18-92a7-c5fd30614277"}
``` python
train_x.label.value_counts()
```

::: {.output .execute_result execution_count="95"}
    neutral    3347
    hate       1776
    Name: label, dtype: int64
:::
:::

::: {.cell .code id="V9THc2fqtGpy" outputId="c06586fd-67a3-4e6a-9e5d-a6ae1a96e74d"}
``` python
test_y.label.value_counts()
```

::: {.output .execute_result execution_count="96"}
    neutral    1923
    hate        980
    Name: label, dtype: int64
:::
:::

::: {.cell .code id="zF-bY98LtGpy" outputId="9a672bec-8ee9-45ea-c31e-ac1d96b95f1b"}
``` python
def fit_shallow_optimized(X, y):
    basemod = TorchShallowNeuralClassifier(
        early_stopping=True,
        hidden_activation = nn.ReLU(),
        hidden_dim = 100
        )
    basemod.fit(X, y)
    return basemod

shallow_rerun = sst.experiment(
    train_x,
    bigrams_phi,
    fit_shallow_optimized,
    assess_dataframes=[test_y])
```

::: {.output .stream .stderr}
    Stopping after epoch 22. Validation score did not improve by tol=1e-05 for more than 10 epochs. Final error is 0.08859242871403694
:::

::: {.output .stream .stdout}
                  precision    recall  f1-score   support

            hate      0.765     0.643     0.698       831
         neutral      0.825     0.895     0.858      1560

        accuracy                          0.807      2391
       macro avg      0.795     0.769     0.778      2391
    weighted avg      0.804     0.807     0.803      2391
:::
:::

::: {.cell .code id="tSi3Ucj4tGpy" outputId="af72d73e-0ff8-427f-d911-7e9b9219a17b"}
``` python
def fit_softmax_glove_optimized(X, y):
    basemod = LogisticRegression(
        C = .8,
        class_weight = 'balanced',
        penalty = "l2",
        fit_intercept=True,
        solver='liblinear',
        multi_class='auto')
    basemod.fit(X, y)
    return basemod


softmax_glove_phi_rerun = sst.experiment(train,
                                        glove_phi,
                                        fit_softmax_glove_optimized,
                                        assess_dataframes=test_small,
                                        vectorize=False)
```

::: {.output .stream .stdout}
                  precision    recall  f1-score   support

            hate      0.619     0.676     0.646       861
         neutral      0.808     0.766     0.786      1530

        accuracy                          0.734      2391
       macro avg      0.713     0.721     0.716      2391
    weighted avg      0.740     0.734     0.736      2391
:::
:::

::: {.cell .code id="eW3SIfDKtGpy" outputId="a2423dd1-b894-46ef-e24c-e9fe93e24577"}
``` python
from sklearn.pipeline import Pipeline
def fit_nb_optimized(X, y):
    rescaler = TfidfTransformer()
    mod = MultinomialNB()
    pipeline = Pipeline(
        [('scaler', rescaler), ('model', mod)]
    )
    param_grid = {
        'model__fit_prior': [True],
        'scaler': ['passthrough'],
        'model__alpha': [1.2]}

    bestmod = utils.fit_classifier_with_hyperparameter_search(
        X, y, pipeline,
        param_grid=param_grid,
        cv=5)
    return bestmod


nb_rerun = sst.experiment(
    train_x,
    bigrams_phi,
    fit_nb_optimized,
    assess_dataframes= test_y)
```

::: {.output .stream .stdout}
    Best params: {'model__alpha': 1.2, 'model__fit_prior': True, 'scaler': 'passthrough'}
    Best score: 0.727
                  precision    recall  f1-score   support

            hate      0.785     0.615     0.690       831
         neutral      0.816     0.910     0.861      1560

        accuracy                          0.808      2391
       macro avg      0.801     0.763     0.775      2391
    weighted avg      0.805     0.808     0.801      2391
:::
:::

::: {.cell .code id="tx6vtICttGpy" outputId="e4ac19a6-8541-4678-f2fb-b53f8379fa0b"}
``` python
from sklearn.svm import LinearSVC
def fit_svm_classifier_with_hyperparameter_search(X, y):
    rescaler = TfidfTransformer()
    mod = LinearSVC(loss='squared_hinge', penalty='l2')

    pipeline = Pipeline(
        [('scaler', rescaler), ('model', mod)]
    )

    param_grid = {
        'scaler': ['passthrough'],
        'model__C': [.10]}

    bestmod = utils.fit_classifier_with_hyperparameter_search(
        X, y, pipeline,
        param_grid=param_grid,
        cv=5)
    return bestmod

svm_experiment_xval = sst.experiment(
    train_x,
    bigrams_phi,
    fit_svm_classifier_with_hyperparameter_search,
    assess_dataframes=test_y)
```

::: {.output .stream .stdout}
    Best params: {'model__C': 0.1, 'scaler': 'passthrough'}
    Best score: 0.730
                  precision    recall  f1-score   support

            hate      0.725     0.558     0.631       980
         neutral      0.798     0.892     0.843      1923

        accuracy                          0.779      2903
       macro avg      0.761     0.725     0.737      2903
    weighted avg      0.773     0.779     0.771      2903
:::
:::

::: {.cell .markdown id="5PvP4m-CtGpz"}
### Getting Assess Dataset
:::

::: {.cell .code id="ogq21AAVtGpz"}
``` python
def find_errors(experiment):
    """Find mistaken predictions.

    Parameters
    ----------
    experiment : dict
        As returned by `sst.experiment`.

    Returns
    -------
    pd.DataFrame

    """
    dfs = []
    for i, dataset in enumerate(experiment['assess_datasets']):
        df = pd.DataFrame({
            'raw_examples': dataset['raw_examples'],
            'predicted': experiment['predictions'][i],
            'gold': dataset['y']})
        df['correct'] = df['predicted'] == df['gold']
        df['dataset'] = i
        dfs.append(df)
    return pd.concat(dfs)
```
:::

::: {.cell .code id="9Yjwd7cItGpz"}
``` python
softmax_analysis = find_errors(softmax_rerun)
rnn_analysis = find_errors(rnn_rerun)
shallow_analysis = find_errors(shallow_rerun)
nb_analysis =  find_errors(nb_rerun)
svm_analysis = find_errors(svm_experiment_xval)
```
:::

::: {.cell .code id="a2lpXyU0tGpz"}
``` python
analysis = rnn_analysis.merge(
    shallow_analysis, left_on='raw_examples', right_on='raw_examples')

analysis = analysis.drop('gold_y', axis=1).rename(columns={'gold_x': 'gold'})

# Examples where the rnn model is correct, the SHALLOW is not,
# and the gold label is 'hate'
error_group = analysis[
    (analysis['predicted_x'] != analysis['gold'])
    &
    (analysis['predicted_y'] != analysis['gold'])
    &
    (analysis['gold'] == 'hate')
]
for ex in error_group['raw_examples'].sample(20, random_state=1):
    print("="*70)
    # print(ex)
```
:::

::: {.cell .code id="En2Sm44ctGpz"}
``` python
# Examples where the rnn model is wrong, the SHALLOW is right,
# and the gold label is 'hate'
error_group = analysis[
    (analysis['predicted_x'] != analysis['gold'])
    &
    (analysis['predicted_y'] == analysis['gold'])
    &
    (analysis['gold'] == 'hate')
]
for ex in error_group['raw_examples'].sample(15, random_state=1):
    print("="*70)
    # print(ex)
```
:::

::: {.cell .code id="dNOvvhEytGpz"}
``` python
# Examples where the rnn model is correct, the SHALLOW is not,
# and the gold label is 'hate'
error_group = analysis[
    (analysis['predicted_x'] == analysis['gold'])
    &
    (analysis['predicted_y'] != analysis['gold'])
    &
    (analysis['gold'] == 'hate')
]
for ex in error_group['raw_examples'].sample(10, random_state=69):
    print("="*70)
    print(ex)
```
:::

::: {.cell .code id="znKhflQvtGpz"}
``` python
analysis = rnn_analysis.merge(
    softmax_analysis, left_on='raw_examples', right_on='raw_examples')

analysis = analysis.drop('gold_y', axis=1).rename(columns={'gold_x': 'gold'})

# Examples where the softmax model is incorrent, the SVM is right,
# and the gold label is 'hate'
error_group = analysis[
    (analysis['predicted_x'] == analysis['gold'])
    &
    (analysis['predicted_y'] != analysis['gold'])
    &
    (analysis['gold'] == 'hate')
]
for ex in error_group['raw_examples'].sample(30, random_state=10):
    print("="*70)
    print(ex)
```
:::

::: {.cell .code id="oydvl8qQtGpz"}
``` python
# Examples where the softmax model is incorrent, the SVM is right,
# and the gold label is 'hate'
error_group = analysis[
    (analysis['predicted_x'] != analysis['gold'])
    &
    (analysis['predicted_y'] != analysis['gold'])
    &
    (analysis['gold'] == 'neutral')
]
for ex in error_group['raw_examples'].sample(10, random_state=10):
    print("="*70)
    print(ex)
```
:::

::: {.cell .code id="vVz_1jnhtGpz"}
``` python
analysis = nb_analysis.merge(
    softmax_analysis, left_on='raw_examples', right_on='raw_examples')

analysis = analysis.drop('gold_y', axis=1).rename(columns={'gold_x': 'gold'})

# Examples where the nb model is incorrent, the SVM is right,
# and the gold label is 'hate'
error_group = analysis[
    (analysis['predicted_x'] != analysis['gold'])
    &
    (analysis['predicted_y'] == analysis['gold'])
    &
    (analysis['gold'] == 'hate')
]

for ex in error_group['raw_examples'].sample(10, random_state=10):
    print("="*70)
    print(ex)
```
:::

::: {.cell .code id="mbw_IioLtGp0"}
``` python
analysis = nb_analysis.merge(
    rnn_analysis, left_on='raw_examples', right_on='raw_examples')

analysis = analysis.drop('gold_y', axis=1).rename(columns={'gold_x': 'gold'})

# Examples where the nb model is corect, the RNN is incorrect,
# and the gold label is 'hate'
error_group = analysis[
    (analysis['predicted_x'] == analysis['gold'])
    &
    (analysis['predicted_y'] != analysis['gold'])
    &
    (analysis['gold'] == 'hate')
]

for ex in error_group['raw_examples'].sample(10, random_state=10):
    print("="*70)
    print(ex)
```
:::
