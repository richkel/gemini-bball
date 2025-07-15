<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" class="logo" width="120"/>

# Open Source Repositories and Tools Using MediaPipe for Object Tracking and Identification

MediaPipe is a widely adopted open-source framework developed by Google for building cross-platform machine learning pipelines, especially in computer vision. Below is a curated list of repositories and open-source tools that leverage MediaPipe for object tracking, people body part identification, and object identification.

## 1. Official MediaPipe Repositories

### **MediaPipe Core Repository**

- **Description:** The core repository includes solutions for object detection, object tracking, pose estimation, hand and face landmark detection, and more.
- **Features:** Cross-platform support (Android, iOS, Python, Web), customizable pipelines, and real-time performance.
- **Repo:** [google-ai-edge/mediapipe][^1]


### **MediaPipe Box Tracking**

- **Description:** Provides a real-time box tracking solution, which can be paired with object detection for efficient tracking pipelines. Used in applications like YouTube’s privacy blur and Google Lens.
- **Repo/Docs:** [Box Tracking Documentation][^2]


### **MediaPipe Objectron**

- **Description:** Real-time 3D object detection and pose estimation for everyday objects. Detects objects in 2D images and estimates their 3D poses.
- **Repo/Docs:** [Objectron Documentation][^3][^4]


## 2. Community and Example Repositories

### **MediaPipe Pose, Face, and Hand Detection**

- **Description:** Real-time detection and tracking of human body poses, faces, and hands, including 2D and 3D joint detection.
- **Features:** 33 body landmarks for pose, 468 facial landmarks, 21 hand keypoints, gesture recognition, and cross-platform support.
- **Repo:** [AISoltani/MediaPipe_Pose_Face_Hand_Detection][^5]


### **Full-Body Estimation Using MediaPipe Holistic**

- **Description:** Example code for full-body, face, and hand pose detection using Python and MediaPipe Holistic.
- **Features:** Real-time video feed, landmark visualization, gesture and pose recognition.
- **Repo:** [nicknochnack/Full-Body-Estimation-using-Media-Pipe-Holistic][^6]


### **Custom Object Detection with MediaPipe**

- **Description:** Tutorials and sample notebooks for training custom object detection models using MediaPipe Model Maker.
- **Features:** Custom dataset support, model training, and deployment for object detection tasks.
- **Repos:**
    - [mediapipe-samples/examples/customization/object_detector.ipynb][^7]
    - [Train a custom object detection model with MediaPipe Model Maker][^8]


## 3. Integration and Deployment Examples

### **MediaPipe Object Detection Demo with OpenVINO**

- **Description:** Demonstrates how to implement MediaPipe object detection graphs using OpenVINO Model Server for efficient inference.
- **Features:** Real-time stream analysis, Docker deployment, and bounding box visualization.
- **Repo/Docs:** [OpenVINO Model Server Demo][^9]


## 4. Tutorials and Guides

- **Real-time Human Pose Estimation using MediaPipe:** Step-by-step guide for pose estimation in Python, including code and requirements for real-time body landmark detection[^10].
- **Custom Object Detection on the Web with MediaPipe:** Guide for training and deploying custom object detectors in JavaScript using MediaPipe Model Maker[^11].
- **Real-Time 3D Pose Detection \& Classification:** Tutorial for 3D pose detection and pose classification using MediaPipe in Python[^12].


## 5. Useful Documentation

- **MediaPipe Solutions Guide:** Overview of available solutions, customization options, and platform-specific guides[^13].
- **Object Detection Task Guide:** Platform-specific implementation guides for object detection using MediaPipe[^14].
- **Pose Landmark Detection Guide:** Detailed instructions for detecting human body landmarks in images and video[^15].


## Summary Table

| Tool/Repo Name | Functionality | Platform(s) | Reference |
| :-- | :-- | :-- | :-- |
| MediaPipe Core | Object tracking, pose, hand, face detection | Android, iOS, Web, Python | [^1] |
| Box Tracking | Real-time box/object tracking | Cross-platform | [^2] |
| Objectron | 3D object detection and pose estimation | Mobile, Desktop | [^3][^4] |
| Pose, Face, Hand Detection (AISoltani) | Body, face, hand landmark detection | Python | [^5] |
| Full-Body Estimation (nicknochnack) | Holistic body, face, hand tracking | Python | [^6] |
| Custom Object Detection (MediaPipe Model Maker) | Train custom object detectors | Python, Web | [^7][^8] |
| OpenVINO MediaPipe Object Detection Demo | Object detection with efficient inference | Python, Docker | [^9] |

These repositories and tools provide robust starting points for building applications involving object tracking, body part detection, and object identification using MediaPipe. Most are actively maintained and widely adopted in both research and industry.

<div style="text-align: center">⁂</div>

[^1]: https://github.com/google-ai-edge/mediapipe

[^2]: https://github.com/google/mediapipe/blob/master/docs/solutions/box_tracking.md

[^3]: https://github.com/google/mediapipe/blob/master/docs/solutions/objectron.md

[^4]: https://mediapipe.readthedocs.io/en/latest/solutions/objectron.html

[^5]: https://github.com/AISoltani/MediaPipe_Pose_Face_Hand_Detection

[^6]: https://www.youtube.com/watch?v=pG4sUNDOZFg

[^7]: https://github.com/googlesamples/mediapipe/blob/main/examples/customization/object_detector.ipynb

[^8]: https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/tutorials/object_detection/Object_Detection_for_3_dogs.ipynb

[^9]: https://docs.openvino.ai/2025/model-server/ovms_docs_demo_mediapipe_object_detection.html

[^10]: https://sigmoidal.ai/en/real-time-human-pose-estimation-using-mediapipe/

[^11]: https://javascript.plainenglish.io/custom-object-detection-on-the-web-with-mediapipe-900e7b9030fd

[^12]: https://bleedaiacademy.com/introduction-to-pose-detection-and-basic-pose-classification/

[^13]: https://ai.google.dev/edge/mediapipe/solutions/guide

[^14]: https://ai.google.dev/edge/mediapipe/solutions/vision/object_detector

[^15]: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker

[^16]: https://developers.googleblog.com/object-detection-and-tracking-using-mediapipe/

[^17]: https://viso.ai/computer-vision/mediapipe/

[^18]: https://www.reddit.com/r/madeinpython/comments/124w0b8/custom_object_detection_using_python_mediapipe/

[^19]: https://chuoling.github.io/mediapipe/

[^20]: http://forum.vvvv.org/t/mediapipe-object-tracking-question/22761

