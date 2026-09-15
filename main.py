import cv2
import mediapipe as mp
import ast
import numpy as np
import math
import subprocess

UMBRAL_VALUE = 1

command_number = 0

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),           # Pulgar
    (0, 5), (5, 6), (6, 7), (7, 8),           # Índice
    (5, 9), (9, 10), (10, 11), (11, 12),      # Medio
    (9, 13), (13, 14), (14, 15), (15, 16),    # Anular
    (13, 17), (17, 18), (18, 19), (19, 20),   # Meñique
    (0, 17)                                   # Palma base
]

def draw_landmarks_on_image(rgb_image: np.ndarray, detection_result) -> np.ndarray:
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)
    height, width, _ = annotated_image.shape

    for idx in range(len(hand_landmarks_list)): # For each hand in handlandmarks
        hand_landmarks = hand_landmarks_list[idx] # Get the landmarks coords

        # Normalized coords into Pixels
        points = []
        for lm in hand_landmarks: # Gets the x and y from the hand_landmarks data structure
            px = int(lm.x * width)
            py = int(lm.y * height)
            points.append((px, py))

        # 1. Draw the lines
        for start_idx, end_idx in HAND_CONNECTIONS:
            cv2.line(annotated_image, points[start_idx], points[end_idx], (255, 255, 255), 2)

        # 2. Draw the dots
        for px, py in points:
            cv2.circle(annotated_image, (px, py), 5, (0, 0, 255), -1)

    return annotated_image

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    print('hand landmarker result: {}'.format(result))

def truncate(num):
    decim = 10 ** UMBRAL_VALUE
    return math.trunc(num * decim)/decim

def write_position_on_file(hand_pos, positions_set, positions_file, command_file):
    pos_list = []
    reference_point = ((hand_pos[0].x),(hand_pos[0].y))
    for i in hand_pos:
        point = (i.x,i.x)
        distance = math.dist(reference_point, point)
        pos_list.append(truncate(distance))
    pos_tupple = tuple(pos_list)
    print(pos_tupple)
    command = command_file.readline().strip()
    positions_set[pos_tupple] = command
    positions_file.write(str(pos_tupple)+'\n')
    return positions_set

def compare_positions(hand_pos, positions_set):
    pos_list = []
    reference_point = ((hand_pos[0].x),(hand_pos[0].y))
    for i in hand_pos:
        point = (i.x,i.x)
        distance = math.dist(reference_point, point)
        pos_list.append(truncate(distance))
    pos_tupple = tuple(pos_list)
    if pos_tupple in positions_set:
        execute_command(positions_set[pos_tupple])

def execute_command(command):
    subprocess.run(command, shell=True)

def main():
    # We open the read only commands file
    command_file = open('commands.txt', 'r', encoding="utf-8")

    # We read the registered positions
    positions_file = open('positions.txt', 'r', encoding="utf-8") 
    # Save them on a dictionary as keys
    positions_set = {} 
    all_positions = positions_file.readlines()
    if len(all_positions) >= 1:
        for i in all_positions:
            command = command_file.readline().strip() 
            positions_set[ast.literal_eval(i)] = command # Commands as values
    positions_file.close()
    # And we open the file as a append file
    positions_file = open('positions.txt', 'a', encoding="utf-8")


    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
        running_mode=VisionRunningMode.VIDEO
    )

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        exit()
    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            # ret = Boolean that tells if there is an error when reading camera
            # frame = The actual frame captured
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Can't receive frame.")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Turning the frame (NumPy array) into something the Mediapipe can work with 
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            result = landmarker.detect_for_video(mp_image, cv2.getTickCount())

            # Display the frame
            annotated_image = draw_landmarks_on_image(rgb_frame, result)
            cv2.imshow("camera",cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))        

            if len(result.hand_landmarks) >= 1:
                compare_positions(result.hand_landmarks[0], positions_set)

            pressed_key = cv2.waitKey(1)
            # Press 'p' to save current position on a file
            if pressed_key == ord('p') and len(result.hand_landmarks) >= 1:
                positions_set = write_position_on_file(result.hand_landmarks[0], positions_set, positions_file, command_file)
            # Press 'q' on the keyboard to exit the loop
            elif pressed_key == ord('q'):
                positions_file.close()
                command_file.close()
                break

        cap.release()
        cv2.destroyAllWindows()

main()