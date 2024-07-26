import mediapipe as mp
import cv2
import json
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt

def landmarks_list_to_array(landmark_list):

    array = []
    for lmk in landmark_list:
        new_row = {
          'x': lmk.x,
          'y': lmk.y,
          'z': lmk.z,
          'visibility': lmk.visibility
        }
        array.append(new_row)
    return array

def Save_Json(path, index,dump_data):
    json_path = path + "" + str(index) + ".json"
    with open(json_path, 'w') as fl:
        # np.around(pose_landmarks, 4).tolist()
        fl.write(json.dumps(dump_data, indent=2, separators=(',', ': ')))
        fl.close()

def draw_landmarks_on_image(rgb_image, detection_result):
  face_landmarks_list = detection_result.face_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected faces to visualize.
  for idx in range(len(face_landmarks_list)):
    face_landmarks = face_landmarks_list[idx]

    # Draw the face landmarks.
    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    face_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
    ])

    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_tesselation_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_contours_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_IRISES,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp.solutions.drawing_styles
          .get_default_face_mesh_iris_connections_style())

  return annotated_image

def plot_face_blendshapes_bar_graph(face_blendshapes):
  # Extract the face blendshapes category names and scores.
  face_blendshapes_names = [face_blendshapes_category.category_name for face_blendshapes_category in face_blendshapes]
  face_blendshapes_scores = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]
  # The blendshapes are ordered in decreasing score value.
  face_blendshapes_ranks = range(len(face_blendshapes_names))

  fig, ax = plt.subplots(figsize=(12, 12))
  bar = ax.barh(face_blendshapes_ranks, face_blendshapes_scores, label=[str(x) for x in face_blendshapes_ranks])
  ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
  ax.invert_yaxis()

if __name__ == '__main__':

    json_path = "JSON/"
    video_path = 'Data/face_test.mov'
    cap = cv2.VideoCapture(video_path)
    
    # Save the Video result
    video_name = video_path.split('/')[-1].split('.')[0]
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    size = (frame_width, frame_height)
    result_save = cv2.VideoWriter('./test_esti_face.avi',
                                cv2.VideoWriter_fourcc(*'MJPG'),
                                fps, size)

    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='Task/face_landmarker.task'),
        running_mode=VisionRunningMode.IMAGE)

    with FaceLandmarker.create_from_options(options) as landmarker:
    # The landmarker is initialized. Use it here.
        while cap.isOpened():
            ret, frame = cap.read()
            # current_frame
            frame_count = cap.get(cv2.CAP_PROP_POS_FRAMES)
            if not ret:
                print("Some probelm with video, or at the end of the video!")
                break

            # Convert the frame to RGB as Mediapipe uses RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rows, cols, _ = rgb_frame.shape
            rgb_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Process the frame to detect face landmarks
            results = landmarker.detect(rgb_frame)

            # get the points
            try:
                face_landmarks = landmarks_list_to_array(results.face_landmarks[0])
                face_pose = {
                    'predictions': face_landmarks,
                    'frame': frame_count,
                    'height': rows,
                    'width': cols
                    }

                json_data = {
                    'facePose': face_pose,
                    'frame': frame_count
                    }
                # print(json_data)
                # yield json_data
                json_name = video_name+ '_' + str(int(frame_count))
                Save_Json(json_path, json_name, json_data)

            except:
                # print("Estimate pose error, there may be no corresponding target in the video!")
                continue

            # Draw face landmarks
            if results.face_landmarks:
                save_frame = draw_landmarks_on_image(frame, results)
                result_save.write(save_frame)
    cap.release()
    