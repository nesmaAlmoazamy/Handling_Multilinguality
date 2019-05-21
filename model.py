from keras.layers import GRU,Dropout,BatchNormalization
from sklearn.model_selection import train_test_split
import keras
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer


def myfunc(myDataset,shape):
    print(myDataset.head())
    X = myDataset.text
    Y = myDataset.label
    encoder = LabelEncoder()
    encoder.fit(Y)

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(
       X, Y, test_size=0.25, random_state=1000)

    Ytrain = encoder.transform(Ytrain)
    # convert integers to dummy variables (i.e. one hot encoded)
    Ytrain = np_utils.to_categorical(Ytrain)

    Ytest = encoder.transform(Ytest)
    # convert integers to dummy variables (i.e. one hot encoded)
    Ytest = np_utils.to_categorical(Ytest)


    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(Xtrain)

    Xtrain = tokenizer.texts_to_sequences(Xtrain)
    Xtest = tokenizer.texts_to_sequences(Xtest)

    vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index

    maxlen = 100

    Xtrain = pad_sequences(Xtrain, padding='post', maxlen=maxlen)
    Xtest = pad_sequences(Xtest, padding='post', maxlen=maxlen)

    print(Xtrain[0, :])
    print(Ytrain)


    from keras.models import Sequential
    from keras import layers

    embedding_dim = 50

    model = Sequential()
    model.add(layers.Embedding(input_dim=vocab_size, 
                               output_dim=embedding_dim, 
                               input_length=maxlen))

    model.add(GRU(256))
    model.add(Dropout(0.2))
    model.add(layers.Dense(shape, activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    history = model.fit(Xtrain, Ytrain,
                        epochs=10,
                        verbose=True,
                        validation_data=(Xtest, Ytest))
    loss, accuracy = model.evaluate(Xtrain, Ytrain, verbose=True)
    print("Training Accuracy: {:.4f}".format(accuracy))
    loss, accuracy = model.evaluate(Xtest, Ytest, verbose=True)
    print("Testing Accuracy:  {:.4f}".format(accuracy))
    return model
