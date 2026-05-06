import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

data = pd.read_csv("balanced_40rows_dataset.csv")

X = data.drop("stress", axis=1)
y = data["stress"]

scaler = StandardScaler()
X = scaler.fit_transform(X)

joblib.dump(scaler,"dl_scaler.pkl")

X_train,X_test,y_train,y_test = train_test_split(
X,y,test_size=0.2,random_state=42,stratify=y)

model = Sequential()
model.add(Dense(64,activation='relu',input_shape=(X.shape[1],)))
model.add(Dense(32,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer=Adam(),loss='binary_crossentropy',metrics=['accuracy'])

model.fit(X_train,y_train,epochs=100,batch_size=8,validation_data=(X_test,y_test))

model.save("deep_stress_model.h5")