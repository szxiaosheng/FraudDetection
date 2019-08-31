import pandas as pd
import numpy as np
import edward as ed
import keras 
import matplotlib.pyplot as plt
import seaborn as sns

from edward.models import Normal
from keras.models import Sequential, load_model
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from keras.callbacks import EarlyStopping, ModelCheckpoint

trans = pd.read_csv('transaction.csv')

Features = np.array(trans[[col for col in trans.columns if col!='type']])
Labels = np.array(trans[['type']])

X_train, X_test, y_train, y_test = train_test_split(Features, Labels, test_size = 0.5, random_state = 0)

scalar = StandardScaler().fit(X_train)

X_train = scalar.transform(X_train)
X_test = scalar.transform(X_test)

min_lengh = min(X_train.shape[0], X_test.shape[0])
X_train = X_train[:min_lengh, ]
y_train = y_train[:min_lengh, ]
X_test = X_test[:min_lengh, ]
y_test = y_test[:min_lengh, ]

model = Sequential()
model.add(Dense(13, activation = "relu", input_shape = (13,)))
model.add(Dense(8, activation = "relu"))
model.add(Dense(8, activation = "relu"))
model.add(Dense(1, activation = "sigmoid"))

modelfile = 'detectmodel.h5'
early_stopping_monitor = EarlyStopping(patience = 2)
checkpoint = ModelCheckpoint(modelfile, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

adam = keras.optimizers.Adam(lr=0.0005)
model.compile(loss = "binary_crossentropy",
             optimizer = adam,
             metrics = ['accuracy'])

history = model.fit(X_train, y_train, 
                    validation_split=0.2, 
                    epochs = 20, 
                    batch_size = 4, 
                    callbacks = [early_stopping_monitor, checkpoint],
                    verbose = 1)

print(history.history.keys())

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('./acc_nn', fmt = 'png', dpi = 300)
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('./loss_nn', fmt = 'png', dpi = 300)
plt.show()

model = load_model(modelfile)

# Prediction
y_pred = model.predict(X_test)
y_pred = [np.round(i[0]) for i in y_pred]

score = model.evaluate(X_test, y_test, verbose = 0)
print("Test loss:", score[0])
print("Test acc:", score[1])

conf = confusion_matrix(y_test, y_pred)
print("Test acc from confusion matrix:",float((conf[0][0] + conf[1][1]))/np.sum(conf))
sns.heatmap(conf, annot = True)
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.savefig('./conf_nn', fmt = 'png', dpi = 300)
plt.show()
