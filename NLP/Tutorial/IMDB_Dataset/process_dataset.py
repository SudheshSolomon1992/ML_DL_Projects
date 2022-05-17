import numpy as np
import tensorflow
import tensorflow.keras.datasets as keras_data

# Vectorization
# sequence = words from a sentence converted to its index
def vectorize_sequence(sequences,dimensions):
    results=np.zeros((len(sequences),dimensions))
    for i, sequence in enumerate(sequences):
        results[i,sequence] = 1.
    return results

def process_data():
    # num_words = limits the dataset to 10000 most used words
    (imdb_train_data,imdb_train_labels),(imdb_test_data,imdb_test_labels) = keras_data.imdb.load_data(num_words=10000)

    ############ Exploratory Data Analysis ##################

    # analyse train data
    # print (imdb_train_data[0])
    # print (imdb_train_labels)
    # print(imdb_train_data.shape)
    # print(imdb_test_data.shape)

    # print a record from the numpy array where each word is mapped to a number
    # print(imdb_train_data[0])

    # get_word_index = map the index back to the original words
    # Note: punctuation and spaces are already removed and hence they will not be restored
    # word_index = keras_data.imdb.get_word_index()

    # map index to word mapping into a Python dict
    # reverse_word_index = dict([(value,key) for (key,value) in word_index.items()])

    # print (reverse_word_index)

    # print 10 most common words
    # for key in sorted(reverse_word_index.keys()):
    #     if(key<=10): 
    #         print("%s: %s" % (key, reverse_word_index[key]))

    ############ Exploratory Data Analysis ##################

    x_train=vectorize_sequence(imdb_train_data,10000)
    x_test=vectorize_sequence(imdb_test_data,10000)

    # convert labels to float as keras accepts only float values
    y_train=np.asarray(imdb_train_labels).astype('float32')
    y_test=np.asarray(imdb_test_labels).astype('float32')

    # verify if the vectorization is done successfully
    print(x_train.shape)
    print(x_test.shape)

    return x_train, y_train, x_test, y_test

