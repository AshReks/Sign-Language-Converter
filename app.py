import streamlit as st
import cv2
import numpy as np
import math
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier

def main():
    st.title("Live Hand Detection and Classification with Streamlit")

    # Create a HandDetector object
    cap = cv2.VideoCapture(-1)
    detector = HandDetector(maxHands=1)

    classifier = Classifier("model/keras_model.h5", "model/labels.txt")
    labels = ["A", "B", "C"]

    offset = 20
    imgSize = 300

    stframe = st.empty() 

    while cap.isOpened():
        ret, img = cap.read()
        imgOutput = img.copy()

        if not ret:
            st.warning("Unable to capture video.")
            break

        # Detect hands in the frame
        hands, img = detector.findHands(img)

        if hands:
                hand = hands[0]
                x, y, w, h = hand["bbox"]

                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                imgCrop = img[y-20: y+h + 20, x-20: x+w+ 20]

                imgCropShape = imgCrop.shape

                aspectRatio = h/w
                
                if aspectRatio > 1:
                    k = imgSize/h 
                    wcal = math.ceil(k*w)

                    imgResize = cv2.resize(imgCrop, (wcal, imgSize))
                    imgResizeShape = imgResize.shape

                    wGap = math.ceil((imgSize - wcal)/2)

                    imgWhite[:, wGap:wcal+wGap] = imgResize
                    prediction, index = classifier.getPrediction(imgWhite, draw=False)
                    print(prediction, index)
                
                else:
                    k = imgSize/w 
                    hCal = math.ceil(k*h)

                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    imgResizeShape = imgResize.shape

                    hGap = math.ceil((imgSize - hCal)/2)

                    imgWhite[hGap:hCal+hGap, :] = imgResize

                    prediction, index = classifier.getPrediction(imgWhite, draw=False)
                    print(prediction, index)

                cv2.rectangle(imgOutput, (x-offset, y-offset -50), (x-offset + 50, y-offset), (255, 0, 255), cv2.FILLED)
                cv2.putText(imgOutput, labels[index], (x - offset + 8, y - offset - 5), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255,255,255), 2)
                
                cv2.rectangle(imgOutput, (x - offset,y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 4)

        
        # Display the processed frame in the Streamlit app
        stframe.image(imgOutput, channels="BGR", use_column_width=True)

    cap.release()

if __name__ == "__main__":
    main()
