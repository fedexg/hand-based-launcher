import cv2
import mediapipe as mp
import numpy as np

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),           # Pulgar
    (0, 5), (5, 6), (6, 7), (7, 8),           # Índice
    (5, 9), (9, 10), (10, 11), (11, 12),      # Medio
    (9, 13), (13, 14), (14, 15), (15, 16),    # Anular
    (13, 17), (17, 18), (18, 19), (19, 20),   # Meñique
    (0, 17)                                   # Palma base
]

MARGIN = 10
FONT_SIZE = 0.8
FONT_THICKNESS = 2
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # Verde

def draw_landmarks_on_image(rgb_image: np.ndarray, detection_result) -> np.ndarray:
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)
    height, width, _ = annotated_image.shape

    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx] 

        # Convertir coordenadas normalizadas (0.0 a 1.0) a píxeles exactos
        points = []
        for lm in hand_landmarks:
            px = int(lm.x * width)
            py = int(lm.y * height)
            points.append((px, py))

        # 1. Dibujar las líneas de conexión entre puntos
        for start_idx, end_idx in HAND_CONNECTIONS:
            cv2.line(annotated_image, points[start_idx], points[end_idx], (255, 255, 255), 2)

        # 2. Dibujar los puntos
        for px, py in points:
            cv2.circle(annotated_image, (px, py), 5, (0, 0, 255), -1)

        # Right or Left and place text, UNNECESSARY
        #handedness = handedness_list[idx] 
        """
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        text_x = max(0, min(x_coords))
        text_y = max(20, min(y_coords) - MARGIN)
        cv2.putText(
            annotated_image, 
            f"{handedness[0].category_name}",
            (text_x, text_y), 
            cv2.FONT_HERSHEY_DUPLEX,
            FONT_SIZE, 
            HANDEDNESS_TEXT_COLOR, 
            FONT_THICKNESS, 
            cv2.LINE_AA
        )
        """
    return annotated_image

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    print('hand landmarker result: {}'.format(result))

def main():
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

            # Press 'q' on the keyboard to exit the loop
            if cv2.waitKey(1) == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

main()