import streamlit as st
import cv2
import numpy as np
from PIL import Image

def main():
    st.title('Face Recognition App')

    picture = st.camera_input("Take a picture")
    if picture:
        st.image(picture, caption='Captured Image', use_container_width=True)
        

        # detect faces in the pages using openCV
        image = Image.open(picture)
        image_np = np.array(image)
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        faces_img = []
        st.header(f"Detected Faces: {len(faces)}")

        for (x, y, w, h) in faces:
            #cv2.rectangle(image_np, (x, y), (x+w, y+h), (255, 0, 0), 2)
            f = image_np[y: y+h, x:x+w]
            #resize face image to 250 x 250
            f = cv2.resize(f, (250, 250))
            faces_img.append(f)
            st.image(f)

if __name__ == "__main__":
    main()
