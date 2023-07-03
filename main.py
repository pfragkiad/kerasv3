# import tensorflow as tf
import numpy as np
import cv2

import buildModel
model = buildModel.getModel('numReader.model')

xTest, yTest = buildModel.getTest()

predictions = model.predict(xTest)
#print(predictions)

#print(predictions[0])
print(np.array2string(predictions[0],formatter = {'float_kind':lambda x: f'{x:.0000%}'}))

print(np.argmax(predictions[0]))
cv2.imshow('xTest',xTest[0])
cv2.waitKey(0)
cv2.destroyAllWindows()

# print(xTrain[0])

# import matplotlib.pyplot as plt
# print(plt.cm.binary)
# plt.subplot(221)
# plt.imshow(xTrain[0], cmap=plt.cm.binary)
# plt.subplot(222)
# # colormaps in https://scipy-cookbook.readthedocs.io/items/Matplotlib_Show_colormaps.html
# plt.imshow(xTrain[1], cmap="binary")  #'gray'
# plt.show()

# from functools import reduce; columns = 60; rows=30;
# train = reduce( lambda t1,t2 : np.vstack((t1,t2)),(np.hstack(xTrain[columns*i:columns*(i+1)]) for i in range(0,rows)) )
# cv2.imshow('xTrain',train)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
