import asyncio
import cv2
import face_recognition
import numpy as np
import os
from datetime import datetime

# TODO: Разделить на функции \/
# TODO: Добавить улучшенную систему записи последнего появления человека \/
# TODO: Не обводить лица, если нет сходства более 70% \/
# TODO: Сделать код асинхронным \/

async def load_known_faces(path):
    if not os.path.exists(path):
        print(f"Error: The directory '{path}' does not exist.")
        return [], []

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
    return images, classNames

async def find_encodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodes = face_recognition.face_encodings(img)
        if encodes:
            encodeList.append(encodes[0])
        else:
            print("Warning: No face detected in an image. Skipping...")
    return encodeList

async def mark_attendance(name):
    try:
        with open("Attendance.csv", "r+") as f:
            myDataList = f.readlines()
            nameList = [line.split(',')[0] for line in myDataList]

        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime("%D , %H:%M:%S")
            with open("Attendance.csv", "a") as f:
                f.write(f'{name}, {dtString}\n')
    except FileNotFoundError:
        print("Attendance file not found. Creating a new one.")
        now = datetime.now()
        dtString = now.strftime("%D , %H:%M:%S")
        with open("Attendance.csv", "w") as f:
            f.write(f'{name}, {dtString}\n')

async def process_video_feed(encodeListKnown, classNames):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access the webcam.")
        return

    try:
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

                if matches[matchIndex] and faceDis[matchIndex] < 0.4:  # Условие схожести более 70%
                    name = classNames[matchIndex]
                    y1, x2, y2, x1 = faceLoc
                    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                    cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                    await mark_attendance(name)

            cv2.imshow("Recognite", img)
            if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
                print("Exiting...")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

async def main():
    path = './KnownFaces'
    images, classNames = await load_known_faces(path)
    encodeListKnown = await find_encodings(images)
    print("Encodings Complete")
    await process_video_feed(encodeListKnown, classNames)

if __name__ == "__main__":
    asyncio.run(main())
