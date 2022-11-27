import cv2
import numpy as np
import face_recognition

imgsample = face_recognition.load_image_file("Sample/sample2.jpg")
imgsample = cv2.cvtColor(imgsample,cv2.COLOR_BGR2RGB)
imgtest = face_recognition.load_image_file("Test/test4.jpg")
imgtest = cv2.cvtColor(imgtest,cv2.COLOR_BGR2RGB)

loc_sam = face_recognition.face_locations(imgsample)[0]
en_sam = face_recognition.face_encodings(imgsample)[0]
cv2.rectangle(imgsample,(loc_sam[3],loc_sam[0]),(loc_sam[1],loc_sam[2]),(255,0,255),2)


loc_test = face_recognition.face_locations(imgtest)[0]
en_test= face_recognition.face_encodings(imgtest)[0]
cv2.rectangle(imgtest,(loc_test[3],loc_test[0]),(loc_test[1],loc_test[2]),(255,0,255),2)

results = face_recognition.compare_faces([en_sam],en_test)
print(results)
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(imgsample, 'anushka', (loc_sam[0],loc_sam[0]), font, 1, (255,0,255), 2, cv2.LINE_AA)
cv2.imshow('sample',imgsample)
cv2.imshow('test',imgtest)
cv2.waitKey(0)