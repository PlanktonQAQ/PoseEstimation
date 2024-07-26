import cv2
import mediapipe as mp
from Pose_Estimate import landmarks_list_to_array, add_extra_points, Save_Json
 
if __name__ == '__main__':

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_holistic = mp.solutions.holistic
    mp_hands = mp.solutions.hands

    json_path = "JSON/"
    video_path = 'Data/Pose_test.MP4'
    cap = cv2.VideoCapture(video_path)

    # save the video
    video_name = video_path.split('/')[-1].split('.')[0]
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    size = (frame_width, frame_height)
    result = cv2.VideoWriter('./test_esti.avi',
                                cv2.VideoWriter_fourcc(*'MJPG'),
                                fps, size)
    
    # cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.8,
        model_complexity=2) as holistic:
      while cap.isOpened():
        success, image = cap.read()
        # current_frame
        frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        if not success:
          print("Some probelm with video, or at the end of the video!!")
          break

        # To improve performance, optionally mark the image as not writeable to
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)

        # ---- Body pose ----
        rows, cols, _ = image.shape
        try:
            pose_landmarks = landmarks_list_to_array(results.pose_landmarks)  
            # also can use results.pose_landmarks
            # world_pose_landmarks = world_landmarks_list_to_array(results.pose_world_landmarks, image.shape)
            add_extra_points(pose_landmarks)
            # add_extra_points(world_pose_landmarks)
            body_pose = {
                'predictions': pose_landmarks,
                'frame': frame,
                'height': rows,
                'width': cols
            }
        except:
            body_pose = {
                'predictions': [],
                'frame': frame,
                'height': rows,
                'width': cols
            }
            # print("Estimate pose error, there may be no corresponding human target in the video!")
            continue
        # ---- Hands ----
        hands_array_R = []
        hands_array_L = []
        if results.left_hand_landmarks:
            hands_array_L = landmarks_list_to_array(results.left_hand_landmarks)
        if results.right_hand_landmarks:
            hands_array_R = landmarks_list_to_array(results.right_hand_landmarks)
        hands_pose = {
            'handsR': hands_array_R,
            'handsL': hands_array_L,
            'frame': frame
        }

        ## ---- Face ---- 
        ## Need to assist face detection algorithm to further implement
        # facial_expression = mocap.get_frame_facial_mocap(image,frame)
        # if facial_expression is None:
        #     facial_expression = {
        #     'leftEyeWid': -1,
        #     'rightEyeWid': -1,
        #     'mouthWid': -1,
        #     'mouthLen': -1,
        #     'frame': frame
        #     }

        json_data = {
            'bodyPose': body_pose,
            'handsPose': hands_pose,
            'frame': frame
        }
        # print(json_data)
        # yield json_data
        json_name = video_name+ '_' + str(int(frame))
        Save_Json(json_path, json_name, json_data)

        # Draw landmark annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw face landmarks
        if results.face_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.face_landmarks,
                mp_holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_contours_style())
        
        # Draw body landmarks
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles
                .get_default_pose_landmarks_style())
        
        # Draw hand landmarks
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.right_hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.left_hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
        result.write(image)
    cap.release()