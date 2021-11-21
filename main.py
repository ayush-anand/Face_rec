import cv2
import numpy as np
import os
from datetime import datetime
import face_recognition
from PIL import ImageGrab
import streamlit as st
from PIL import Image
import numpy as np

path = 'images'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findEncodings(images):
    encodeList = []


    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown = findEncodings(images)
print('Encoding Complete')

def findimg(imga):
    img = np.array(imga)
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        print(faceDis)
        # print('hi')
        # print(match)
        # matchIndex = np.argmin(faceDis)
        print(matches)
        name1 = 'unknown'
        curr = 110
        matchIndex = np.argmax(matches)
        for i in range(len(matches)):
            if faceDis[i] < 45:
                if curr > faceDis[i]:
                    name1 = classNames[i].upper()
        # if matches[matchIndex]:
        #     name = classNames[matchIndex].upper()
# print(name)
        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (255, 255, 0), cv2.FILLED)
        cv2.putText(img, name1, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)
            # markAttendance(name)

    # cv2.imshow('Webcam', img)
    # cv2.waitKey(1)
    return img
def main():
    """Face Recognition App"""

    st.title("Streamlit Tutorial")

    html_temp = """
    <body style="background-color:red;">
    <div style="background-color:teal ;padding:10px">
    <h2 style="color:white;text-align:center;">Face Recognition WebApp</h2>
    </div>
    </body>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    image_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
    if image_file is not None:
        our_image = Image.open(image_file)
        st.text("Original Image")
        st.image(our_image)

    if st.button("Recognise"):
        result_img= findimg(our_image)
        st.image(result_img)



if __name__ == '__main__':
    main()
