# use the saved model
from sklearn.externals import joblib
import dlib
from get_features import get_features
import ML_ways_sklearn

import cv2


def smileDetectFunc(s):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('data/data_dlib_model/shape_predictor_68_face_landmarks.dat')

    # path of test img
    path_test_img = s

    # 提取单张40维度特征
    positions_lip_test = get_features(path_test_img)

    # path of models
    path_models = "data/data_models/"

    print("The result of"+path_test_img+":")
    print('\n')

    # #########  LR  ###########
    LR = joblib.load(path_models+"model_LR.m")
    ss_LR = ML_ways_sklearn.model_LR()
    X_test_LR = ss_LR.transform([positions_lip_test])
    y_predict_LR = str(LR.predict(X_test_LR)[0]).replace('0', "no smile").replace('1', "with smile")
    print("LR:", y_predict_LR)

    # #########  LSVC  ###########
    LSVC = joblib.load(path_models+"model_LSVC.m")
    ss_LSVC = ML_ways_sklearn.model_LSVC()
    X_test_LSVC = ss_LSVC.transform([positions_lip_test])
    y_predict_LSVC = str(LSVC.predict(X_test_LSVC)[0]).replace('0', "no smile").replace('1', "with smile")
    print("LSVC:", y_predict_LSVC)

    # #########  MLPC  ###########
    MLPC = joblib.load(path_models+"model_MLPC.m")
    ss_MLPC = ML_ways_sklearn.model_MLPC()
    X_test_MLPC = ss_MLPC.transform([positions_lip_test])
    y_predict_MLPC = str(MLPC.predict(X_test_MLPC)[0]).replace('0', "no smile").replace('1', "with smile")
    print("MLPC:", y_predict_MLPC)

    # #########  SGDC  ###########
    SGDC = joblib.load(path_models+"model_SGDC.m")
    ss_SGDC = ML_ways_sklearn.model_SGDC()
    X_test_SGDC = ss_SGDC.transform([positions_lip_test])
    y_predict_SGDC = str(SGDC.predict(X_test_SGDC)[0]).replace('0', "no smile").replace('1', "with smile")
    print("SGDC:", y_predict_SGDC)

    img_test = cv2.imread(path_test_img)

    faces = detector(img_test, 1)
    print("Number of faces detected :", len(faces))
    for index, face in enumerate(faces):
        left = face.left()
        right = face.right()
        top = face.top()
        bottom = face.bottom()
        cv2.rectangle(img_test,(left,top),(right,bottom),(0,255,0),3)

    img_height = int(img_test.shape[0])
    img_width = int(img_test.shape[1])

    # show the results on the image
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img_test, "LR:    "+y_predict_LR,   (int(img_height/10), int(img_width/10)), font, 0.8, (84, 255, 159), 1, cv2.LINE_AA)
    cv2.putText(img_test, "LSVC:  "+y_predict_LSVC, (int(img_height/10), int(img_width/10*2)), font, 0.8, (84, 255, 159), 1, cv2.LINE_AA)
    cv2.putText(img_test, "MLPC:  "+y_predict_MLPC, (int(img_height/10), int(img_width/10)*3), font, 0.8, (84, 255, 159), 1, cv2.LINE_AA)
    cv2.putText(img_test, "SGDC:  "+y_predict_SGDC, (int(img_height/10), int(img_width/10)*4), font, 0.8, (84, 255, 159), 1, cv2.LINE_AA)

    # cv2.namedWindow("img", 2)
    # cv2.imshow("img", img_test)
    # cv2.waitKey(0)
    return img_test