import numpy as np
import face_recognition
import cv2
import os
from datetime import datetime

path = './KnownFaces'
if not os.path.exists(path):
    print(f"Error: The directory '{path}' does not exist.")
    exit()

images = []
classNames = []
myList = os.listdir(path)
print("Known Faces:", myList)

for cls in myList:
    curImg = cv2.imread(f'{path}/{cls}')
    if curImg is None:
        print(f"Error: Unable to load image {cls}. Skipping...")
        continue
    images.append(curImg)
    classNames.append(os.path.splitext(cls)[0])

print("Class Names:", classNames)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodes = face_recognition.face_encodings(img)
        if encodes:
            encodeList.append(encodes[0])
        else:
            print("Warning: No face detected in an image. Skipping...")
    return encodeList

def markAttendance(name):
    with open("Attendance.csv", "r+") as f:
        myDataList = f.readlines()
        nameList = [line.split(',')[0] for line in myDataList]

        #DO FUCKING ATTEDANCE NORMAL CHECK DOLBAEB
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime("%D , %H:%M:%S")
            f.writelines(f'\n{name}, {dtString}')

encodeListKnown = findEncodings(images)
print("Encodings Complete")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Unable to access the webcam.")
    exit()

while True:
    success, img = cap.read()
    if not success:
        print("Error: Failed to read frame from webcam.")
        break

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodeCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

#DO PLACEHOLDER TO UNKNOWN FACES


        if matches[matchIndex]:
            name = classNames[matchIndex]
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)

    cv2.imshow("Recognite", img)
    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        print("Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
