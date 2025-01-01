import cv2
import mediapipe as mp

# Initialize mediapipe hands and drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


# Set up webcam
cap = cv2.VideoCapture(0)
hands = mp_hands.Hands()

while True:
    ret, image = cap.read()
    if not ret:
        break
    
    # Convert the image color space from BGR to RGB
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    
    # Convert the image color space from RGB to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Draw hand landmarks on the image
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    # Display the resulting frame
    cv2.imshow('Handtracker', image)
    
    # Exit loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

