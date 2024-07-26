import json

import cv2
import mediapipe as mp
import numpy as np

def Save_Json(path, index,dump_data):
    json_path = path + "" + str(index) + ".json"
    with open(json_path, 'w') as fl:
        # np.around(pose_landmarks, 4).tolist()
        fl.write(json.dumps(dump_data, indent=2, separators=(',', ': ')))
        fl.close()

# For adding new landmarks based on default predicted landmarks
def add_extra_points(landmark_list):
    left_shoulder = landmark_list[11]
    right_shoulder = landmark_list[12]
    left_hip = landmark_list[23]
    right_hip = landmark_list[24]

    # Calculating hip position and visibility
    hip = {
          'x': (left_hip['x'] + right_hip['x']) / 2.0,
          'y': (left_hip['y'] + right_hip['y']) / 2.0,
          'z': (left_hip['z'] + right_hip['z']) / 2.0,
          'visibility': (left_hip['visibility'] + right_hip['visibility']) / 2.0
        }
    landmark_list.append(hip)

    # Calculating spine position and visibility
    spine = {
          'x': (left_hip['x'] + right_hip['x'] + right_shoulder['x'] + left_shoulder['x']) / 4.0,
          'y': (left_hip['y'] + right_hip['y'] + right_shoulder['y'] + left_shoulder['y']) / 4.0,
          'z': (left_hip['z'] + right_hip['z'] + right_shoulder['z'] + left_shoulder['z']) / 4.0,
          'visibility': (left_hip['visibility'] + right_hip['visibility'] + right_shoulder['visibility'] + left_shoulder['visibility']) / 4.0
        }
    landmark_list.append(spine)

    left_mouth = landmark_list[9]
    right_mouth = landmark_list[10]
    nose = landmark_list[0]
    left_ear = landmark_list[7]
    right_ear = landmark_list[8]
    # Calculating neck position and visibility
    neck = {
          'x': (left_mouth['x'] + right_mouth['x'] + right_shoulder['x'] + left_shoulder['x']) / 4.0,
          'y': (left_mouth['y'] + right_mouth['y'] + right_shoulder['y'] + left_shoulder['y']) / 4.0,
          'z': (left_mouth['z'] + right_mouth['z'] + right_shoulder['z'] + left_shoulder['z']) / 4.0,
          'visibility': (left_mouth['visibility'] + right_mouth['visibility'] + right_shoulder['visibility'] + left_shoulder['visibility']) / 4.0
        }
    landmark_list.append(neck)

    # Calculating head position and visibility
    head = {
          'x': (nose['x'] + left_ear['x'] + right_ear['x']) / 3.0,
          'y': (nose['y'] + left_ear['y'] + right_ear['y']) / 3.0,
          'z': (nose['z'] + left_ear['z'] + right_ear['z']) / 3.0,
          'visibility': (nose['visibility'] + left_ear['visibility'] + right_ear['visibility']) / 3.0,
        }
    landmark_list.append(head)

def landmarks_list_to_array(landmark_list):

    array = []
    for lmk in landmark_list.landmark:
        new_row = {
          'x': lmk.x,
          'y': lmk.y,
          'z': lmk.z,
          'visibility': lmk.visibility
        }
        array.append(new_row)
    return array
    # return np.asarray([(lmk.x, lmk.y, lmk.z, lmk.visibility)
    #                    for lmk in landmark_list.landmark])


if __name__ == '__main__':
    print("pose estimator started...")
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose

    videp_path = 'Data/Pose_test.MP4'
    video_name = videp_path.split('/')[-1].split('.')[0]
    cap = cv2.VideoCapture(videp_path)
    frame = 0
    Total_result = []

    json_path = "JSON/"
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    size = (frame_width, frame_height)
    result = cv2.VideoWriter('./test.avi',
                                cv2.VideoWriter_fourcc(*'MJPG'),fps, size)

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.8) as pose:
        while cap.isOpened():
            success, image = cap.read()
            # current_frame
            frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            if not success:
                print("Some probelm with video!")
                break

            # To improve performance, optionally mark the image as not writeable
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            try:
                pose_landmarks = landmarks_list_to_array(results.pose_world_landmarks) 
                #also can use results.pose_landmarks
                # world_pose_landmarks = world_landmarks_list_to_array(results.pose_world_landmarks, image.shape)
                rows, cols, _ = image.shape
                add_extra_points(pose_landmarks)
                # add_extra_points(world_pose_landmarks)

                json_data = {
                    'predictions': pose_landmarks,
                    'frame': frame,
                    'height': rows,
                    'width': cols
                    }
                # print(json_data)
                # yield json_data
                json_name = video_name+ '_' + str(int(frame))
                Save_Json(json_path, json_name, json_data)

            except:
                # print("Estimate pose error, there may be no corresponding human target in the video!")
                continue
            # Draw the pose annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(image,
                                        results.pose_landmarks,
                                        mp_pose.POSE_CONNECTIONS,
                                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            result.write(image)

    cap.release()