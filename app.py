import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
import numpy as np

def main():
    st.sidebar.title("Hand Detection Options")
    show_hands = st.sidebar.checkbox("Show Hands", True)

    webrtc_ctx = webrtc_streamer(key="example")

    if webrtc_ctx.video_receiver:
        while True:
            _, frame = webrtc_ctx.video_receiver.read()

            if frame is not None:
                # Convert the frame to grayscale.
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Apply Gaussian blur to reduce noise.
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)

                # Apply thresholding to create a binary image.
                _, thresh = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY)

                # Find contours in the binary image.
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Draw rectangles around detected hands on the original frame.
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 2000:  # Adjust the area threshold as needed.
                        x, y, w, h = cv2.boundingRect(contour)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Display the frame with detected hands.
                if show_hands:
                    st.image(frame, channels="BGR")
            else:
                st.warning("No frame captured.")
                break

if __name__ == "__main__":
    main()
