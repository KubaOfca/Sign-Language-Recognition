import keras
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import os

id_map = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J', 20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T', 30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z', 36: '_'}
model = tf.keras.models.load_model(r'/home/jakub/Desktop/projekt_AI/augmented_model_20ep.h5')

def classify(image):
    img = cv2.resize(image,(50,50))
    img = np.asarray(img) / 255
    img = img.reshape((1, 50, 50, 3))
    predict_matrix = model.predict(img)
    predicted_label_id = predict_matrix.argmax()
    predicted_label_name = id_map[predicted_label_id]
    confidence_lvl = predict_matrix[0, predicted_label_id]
    return predicted_label_name

cap = cv2.VideoCapture(0)
while 1:
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    top, right, bottom, left = 75, 350, 300, 590
    roi = img[top:bottom, right:left]
    roi=cv2.flip(roi,1)
    alpha=classify(roi)
    cv2.rectangle(img, (left, top), (right, bottom), (0,255,0), 2)
    font=cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img,alpha,(0,130),font,5,(0,0,255),2)
    #cv2.resize(img,(1000,1000))
    cv2.imshow('img',img)
    key = cv2.waitKey(1) & 0xFF
    if key==ord('q'):
        break;
cap.release()
cv2.destroyAllWindows()