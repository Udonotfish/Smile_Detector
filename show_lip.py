# 显示嘴部特征点
# Draw the positions of someone's lip

import dlib         # 人脸识别的库 Dlib
import cv2          # 图像处理的库 OpenCv
from get_features import get_features   # return the positions of feature points


def faceDetectFunc(s):
    path_test_img = s

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('data/data_dlib_model/shape_predictor_68_face_landmarks.dat')

    # Get lip's positions of features points
    positions_lip = get_features(path_test_img)

    img_rd = cv2.imread(path_test_img)

    faces = detector(img_rd,1)
    print("Number of faces detected :".format(len(faces)))
    for index, face in enumerate(faces):
        left = face.left()
        right = face.right()
        top = face.top()
        bottom = face.bottom()
        cv2.rectangle(img_rd,(left,top),(right,bottom),(0,255,0),3)

    # Draw on the lip points
    for i in range(0, len(positions_lip), 2):
        print(positions_lip[i], positions_lip[i+1])
        cv2.circle(img_rd, tuple([positions_lip[i], positions_lip[i+1]]), radius=1, color=(0, 255, 0))

    return img_rd
    # cv2.namedWindow("img_read", 2)
    # cv2.imshow("img_read", img_rd)
    # cv2.waitKey(0)