import cv2
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#img = cv2.imread('image4.jpg')
webcam = cv2.VideoCapture(0)

while True:

    successful_frame_read, frame = webcam.read()
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)

    cv2.imshow('text', frame)
    cv2.waitKey(1)

##
#face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)


#for (x, y, w, h) in face_coordinates:
#    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)

#print(face_coordinates)

#cv2.imshow('text', img)
#                       cv2.waitKey()
#print("Code Completed")