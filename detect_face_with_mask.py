import cv2
import tensorflow as tf
import numpy as np
import pyautogui
import imutils

model = tf.keras.models.load_model('keras_model.h5')

vid = cv2.VideoCapture(1)

while True:
    ret, frame = vid.read()
    
    if not ret:
        break
    
    img = cv2.resize(frame, (224, 224))
    text_img = np.array(img, dtype=np.float32)
    normalised_image = text_img / 255.0
    prediction = model.predict(np.expand_dims(normalised_image, axis=0))
    print("Prediction:", prediction)
    
    # Capture the screen and save it to memory
    image_memory = pyautogui.screenshot()
    image_memory = cv2.cvtColor(np.array(image_memory), cv2.COLOR_RGB2BGR)
    cv2.imwrite("in_memory_to_disk.png", image_memory)
    
    # Capture the screen and save it directly to disk
    pyautogui.screenshot("straight_to_disk.png")
    
    # Read the image from disk and display it
    image_disk = cv2.imread("straight_to_disk.png")
    cv2.imshow("Screenshot", imutils.resize(image_disk, width=600))
    
    # Display the video frame
    cv2.imshow('frame', frame)
    
    # Quit window with spacebar
    key = cv2.waitKey(1)
    if key == 32:
        break

# Release the video capture object and close all windows
vid.release()
cv2.destroyAllWindows()
