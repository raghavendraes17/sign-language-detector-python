import os
import pickle

import mediapipe as mp
import cv2


mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.3
)

DATA_DIR = './data'

data = []
labels = []

for dir_ in os.listdir(DATA_DIR):

    if not os.path.isdir(os.path.join(DATA_DIR, dir_)):
        continue

    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):

        data_aux = []

        x_ = []
        y_ = []

        img_path_full = os.path.join(DATA_DIR, dir_, img_path)

        img = cv2.imread(img_path_full)

        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:

            hand_landmarks = results.multi_hand_landmarks[0]

            for landmark in hand_landmarks.landmark:

                x_.append(landmark.x)
                y_.append(landmark.y)

            for landmark in hand_landmarks.landmark:

                data_aux.append(landmark.x - min(x_))
                data_aux.append(landmark.y - min(y_))

            # 21 landmarks × 2 coordinates = 42 features
            if len(data_aux) == 42:
                data.append(data_aux)
                labels.append(dir_)

hands.close()

f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()

print("Dataset creation completed!")
print("Total samples:", len(data))
print("Total labels:", len(labels))