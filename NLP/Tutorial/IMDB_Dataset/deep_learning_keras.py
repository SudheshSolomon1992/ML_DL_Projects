from keras import layers
from keras import models
from keras import optimizers
from keras import losses
from keras import metrics
from datetime import datetime
from process_dataset import process_data
from utility import print_scores, print_result

def train_test_model():
    # Build Model
    model=models.Sequential()
    model.add(layers.Dense(16,activation='relu',input_shape=(10000,)))
    model.add(layers.Dense(16,activation='relu'))
    model.add(layers.Dense(1,activation='sigmoid'))
    model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
    start_time = datetime.now()
    model.fit(x_partial_train,y_partial_train,epochs=4,batch_size=512,validation_data=(x_validation,y_validation))
    end_time = datetime.now()
    print_scores(str((end_time - start_time).total_seconds()), 'Score on train', 'Score on test', str(model.evaluate(x_train,y_train)[1]), str(model.evaluate(x_test,y_test)[1]))

def main():
    train_test_model()
    print_result()

if __name__ == "__main__":
    print ("------------------")
    x_train, y_train, x_test, y_test = process_data()
    # split an additional validation dataset
    x_validation=x_train[:100]
    x_partial_train=x_train[100:]
    y_validation=y_train[:100]
    y_partial_train=y_train[100:]
    main()