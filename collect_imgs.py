import os
import cv2
import mediapipe as mp

DATA_DIR = './data'

number_of_classes = 3
dataset_size = 100

# Camera
cap = cv2.VideoCapture(0)

# MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5
)

for j in range(number_of_classes):

    class_dir = os.path.join(DATA_DIR, str(j))

    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print('Collecting data for class {}'.format(j))

    counter = 0

    while counter < dataset_size:

        ret, frame = cap.read()

        if not ret:
            print("Could not read camera frame")
            break

        # Convert BGR → RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect hand
        results = hands.process(frame_rgb)

        # Display status
        if results.multi_hand_landmarks:
            cv2.putText(
                frame,
                f'Hand detected: {counter}/{dataset_size}',
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            # Save only when hand is detected
            cv2.imwrite(
                os.path.join(class_dir, '{}.jpg'.format(counter)),
                frame
            )

            counter += 1

        else:
            cv2.putText(
                frame,
                'Show your hand',
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

        cv2.imshow('frame', frame)

        if cv2.waitKey(25) == ord('q'):
            break

    if counter < dataset_size:
        print('Collection stopped for class {}'.format(j))
        break

cap.release()
hands.close()
cv2.destroyAllWindows()