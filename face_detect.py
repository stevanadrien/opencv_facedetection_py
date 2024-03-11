import cv2 as cv

path = 'ucl.jpg'
img = cv.imread(path)
# cv.imshow("son", img)

grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow("grey",grey)

haar_cascade = cv.CascadeClassifier('haar_face.xml')

faces_rect = haar_cascade.detectMultiScale(grey, scaleFactor=1.1,minNeighbors=8)

print(f'Number of faces found ={len(faces_rect)}')
for (x,y,w,h) in faces_rect:
    cv.rectangle(img,(x,y), (x+w,y+h), (0,255,0), thickness=2)
    
cv.imshow('Detected', img)
cv.waitKey(0)