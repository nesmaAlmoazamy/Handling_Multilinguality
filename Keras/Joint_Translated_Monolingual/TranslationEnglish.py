import keras
from keras.layers import GRU,Dropout,BatchNormalization
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer

import pandas as pd
import numpy as np
from keras.utils import np_utils

import matplotlib.pyplot as plt
 
EnglishValidationDS = pd.read_csv("EnglishTestSet.csv")
dataset =  pd.read_csv("TranslationExceptEnglishTest.csv")


def plot_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    
def Remove3Row(inputD):
    mydataset = inputD
    value_counts = mydataset['Label'].value_counts()

    # Select the values where the count is less than 3
    to_remove = value_counts[value_counts < 3].index

    # Keep rows where the city column is not in to_remove
    mydataset = mydataset[~mydataset.Label.isin(to_remove)]
    return mydataset




def myfuncTest(myDataset,shape,lang,testSet):
    myDataset = Remove3Row(myDataset)

    print(myDataset.head())
    X = myDataset.Text
    Y = myDataset.Label

    encoder = LabelEncoder()
    encoder.fit(Y)

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=1,stratify = Y)

#     Xtrain, Xval, Ytrain, Yval = train_test_split(Xtrain, Ytrain, test_size=0.25, random_state=1,stratify = Ytrain)
    Xval = testSet.Text
    Yval = testSet.Label

    Ytrain = encoder.transform(Ytrain)
    Ytest = encoder.transform(Ytest)
    Yval = encoder.transform(Yval)

    Ytrain = np_utils.to_categorical(Ytrain)
    Ytest = np_utils.to_categorical(Ytest)
    Yval = np_utils.to_categorical(Yval)


    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(Xtrain)

    Xtrain = tokenizer.texts_to_sequences(Xtrain)
    Xtest = tokenizer.texts_to_sequences(Xtest)
    Xval = tokenizer.texts_to_sequences(Xval)
    vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index

    maxlen = 100

    Xtrain = pad_sequences(Xtrain, padding='post', maxlen=maxlen)
    Xtest = pad_sequences(Xtest, padding='post', maxlen=maxlen)
    Xval = pad_sequences(Xval, padding='post', maxlen=maxlen)


    from keras.models import Sequential
    from keras import layers

    embedding_dim = 50

    model = Sequential()
    model.add(layers.Embedding(input_dim=vocab_size, 
                               output_dim=embedding_dim, 
                               input_length=maxlen))


    model.add(GRU(256))
    model.add(Dropout(0.1))
    model.add(layers.Dense(shape, activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    history = model.fit(Xtrain, Ytrain,
                        epochs=10,
                        verbose=True,
                        validation_data=(Xval, Yval))
    f= open("report.csv","a")
    loss, accuracy = model.evaluate(Xtrain, Ytrain, verbose=True)
    print("Training Accuracy: {:.4f}".format(accuracy))
    f.write(lang+",{:.4f}".format(accuracy))

    loss, accuracy = model.evaluate(Xval, Yval, verbose=True)
   
    print("Validation Accuracy:  {:.4f}".format(accuracy))
    f.write(",{:.4f} \n".format(accuracy))
    f.close()
    plot_history(history)
    Ypred = model.predict(Xtest)
    from sklearn.metrics import confusion_matrix, classification_report

    matrix = confusion_matrix(Ytest.argmax(axis=1), Ypred.argmax(axis=1))
    classification_Report = classification_report(Ytest.argmax(axis=1), Ypred.argmax(axis=1), output_dict=True)
    df = pd.DataFrame(classification_Report).transpose()
    df.to_csv(lang+"ClassificationReport.csv")

myfuncTest(dataset,45,"TranslationEngTest",EnglishValidationDS)
