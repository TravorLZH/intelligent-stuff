#!/usr/bin/env python3
import os
import numpy as np
import keras
import cv2

os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"  # To suppress warnings

mnist=keras.datasets.mnist

(x_train_orig,y_train),(x_test_orig,y_test)=mnist.load_data()

# To make the input between 0 and 1
x_train,x_test=x_train_orig/255.0,x_test_orig/255.0

model=keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(512,activation="relu"),
    keras.layers.Dense(512,activation="relu"),
    keras.layers.Dense(10,activation="softmax")
])
model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",
        metrics=["accuracy"])
model.fit(x_train,y_train,epochs=5)

score=model.evaluate(x_test,y_test,verbose=1)
print("Test result: cost=%f, accuracy=%.f%%" % (score[0],100*score[1]))

exit()

y_predict=model.predict(x_test,batch_size=100)

for i,y in enumerate(y_predict):
    yp=np.argmax(y)
    img=x_test_orig[i].reshape(28,28)
    cvimg=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    cvimg=cv2.resize(cvimg,(28*4,28*4))
    color=(255,0,0)
    wait_time=50
    if yp!=y_test[i]:
        color=(0,0,255) # Use red if wrong
        wait_time=0
    cv2.putText(cvimg,"%d" % np.argmax(y),(5,100),cv2.FONT_HERSHEY_SIMPLEX,
            1,color,2)
    cv2.imshow("MNIST Test",cvimg)
    print("y=%d, yp=%d" % (y_test[i],np.argmax(y)))
    key=cv2.waitKey(wait_time) & 0xFF
    if key==ord('k'):
        cv2.waitKey(0)
    elif key==ord('q'):
        break
