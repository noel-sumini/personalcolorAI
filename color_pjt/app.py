# -*- coding: utf-8 -*-

import cv2
import imutils
import matplotlib.pyplot as plt
import dlib
import numpy as np
import pandas as pd
import os

from imutils import paths
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from werkzeug.utils import secure_filename

import requests
from flask import Flask, render_template, request

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__,
            template_folder = "templates",
            static_folder = "static")
# app.config['ENV'] = 'development'
# app.config['DEBUG'] = True
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



eye_cascade = cv2.CascadeClassifier('./cv2_env/haarcascade_xml/haarcascade_lefteye_2splits.xml')
predictor = dlib.shape_predictor("./cv2_env/shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()


def pcolor_analysis(imagePath):
    image = cv2.imread(imagePath)

    faces = detector(image)
    face = faces[0]
    ## 얼굴 이미지 추출
    
    x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
    face_img = np.array(image[y1:y2, x1:x2, :])

    ## 눈 이미지 추출
    eye = eye_cascade.detectMultiScale(face_img, 1.01, 10)

    eye_x1 = eye[0,0]
    eye_y1 = eye[0,1]
    eye_x2 = eye[0,0] + eye[0,2]
    eye_y2 = eye[0,1] + eye[0,3]

    eye_img = face_img[eye_y1:eye_y2, eye_x1:eye_x2, :]

    ## 이미지 전처리
    face_img_ycrcb = cv2.cvtColor(face_img, cv2.COLOR_BGR2YCrCb)
    eye_img_gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
    eye_img_ycrcb = cv2.cvtColor(eye_img, cv2.COLOR_BGR2YCrCb)

    ## 얼굴 이미지에서 피부 추출
    lower = np.array([0,133,77], dtype = np.uint8)
    upper = np.array([255,173,127], dtype = np.uint8)

    skin_msk = cv2.inRange(face_img_ycrcb, lower, upper)
    skin = cv2.bitwise_and(face_img, face_img, mask = skin_msk)


    ## 눈동자, 모발 추출
    eye_gray_np = eye_img_gray.reshape(-1)
    eye_msk = cv2.inRange(eye_img_gray, 0, np.quantile(eye_gray_np, 0.5))
    eye = cv2.bitwise_and(eye_img, eye_img, mask = eye_msk)

    skin_lab = cv2.cvtColor(skin, cv2.COLOR_BGR2Lab)
    skin_hsv = cv2.cvtColor(skin, cv2.COLOR_BGR2HSV)
    eye_lab = cv2.cvtColor(eye, cv2.COLOR_BGR2Lab)

    ## 피부/눈동자에서 LAB값 추출(눈동자/눈썹/모발 -> L, 피부 -> ab)
    L = np.mean(eye_lab[:,:,0]) *1.0
    a = np.mean(skin_lab[:,:,1])-128
    b = np.mean(skin_lab[:,:,2])-128
    
    S = np.mean(skin_hsv[:,:,1])
    V = np.mean(skin_hsv[:,:,2])

    # ## 퍼스널컬러 분류
    
    return L, a, b, S, V



## Scikit-Learn coding 전처리


train_data = []
label = []
file_paths = os.listdir('./train_dataset')

for idx, filePath in enumerate(file_paths):
    train_imagePaths = list(paths.list_images(os.path.join('./train_dataset', filePath)))
    
    for imagePath in train_imagePaths:
        print(imagePath)
        print('='*20)
        image = cv2.imread(imagePath)

        train_data.append(image)
        label.append(filePath)


result_list = []
value_data = []

for idx, l in enumerate(label):
    _, season, detail = l.split('_')

    temp = season + detail
    result_list.append(temp)


print("train data 특징 추출 중")
print("=" * 20)

for idx, img in enumerate(train_data):

    ## train_data 얼굴 raw 이미지 추출
    faces = detector(img)

    for face in faces:
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        face_img = np.array(img[y1:y2, x1:x2, :])
    
    # plt.imshow(face_img[:,:,::-1])
    ## train_data 눈 raw 이미지 추출
    eye = eye_cascade.detectMultiScale(face_img, 1.01, 10)

    eye_x1 = eye[0,0]
    eye_y1 = eye[0,1]
    eye_x2 = eye[0,0] + eye[0,2]
    eye_y2 = eye[0,1] + eye[0,3]

    eye_img = face_img[eye_y1:eye_y2, eye_x1:eye_x2, :]
    # print(idx, label[idx])
    # plt.imshow(eye_img[:,:,::-1])

    ## 얼굴, 눈 raw이미지 전처리

    face_img_ycrcb = cv2.cvtColor(face_img, cv2.COLOR_BGR2YCrCb)
    eye_img_gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
    eye_img_ycrcb = cv2.cvtColor(eye_img, cv2.COLOR_BGR2YCrCb)

    ## 얼굴 raw 이미지에서 피부 추출
    lower = np.array([0,133,77], dtype = np.uint8)
    upper = np.array([255,173,127], dtype = np.uint8)

    skin_msk = cv2.inRange(face_img_ycrcb, lower, upper)
    skin = cv2.bitwise_and(face_img, face_img, mask = skin_msk)

    ## 눈동자, 모발 추출
    eye_gray_np = eye_img_gray.reshape(-1)
    eye_msk = cv2.inRange(eye_img_gray, 0, np.quantile(eye_gray_np, 0.5))
    eye = cv2.bitwise_and(eye_img, eye_img, mask = eye_msk)

    skin_lab = cv2.cvtColor(skin, cv2.COLOR_BGR2Lab)
    skin_hsv = cv2.cvtColor(skin, cv2.COLOR_BGR2HSV)
    eye_lab = cv2.cvtColor(eye, cv2.COLOR_BGR2Lab)  

    ## 피부/눈동자에서 LAB값 추출(눈동자/눈썹/모발 -> L, 피부 -> ab)
    L = np.mean(eye_lab[:,:,0]) *1.0
    a = np.mean(skin_lab[:,:,1])-128
    b = np.mean(skin_lab[:,:,2])-128

    S = np.mean(skin_hsv[:,:,1])
    V = np.mean(skin_hsv[:,:,2])

    temp = [L, a, b, S, V]
    value_data.append(temp)  


len(result_list)

print("AI분석중")
print("=" * 20)

rfc = RandomForestClassifier()
rfc.fit( np.array(value_data), np.array(result_list))


@app.route('/', methods =['GET', 'POST'])
def result():
    fin_result = ""
    if request.method == 'GET':
        return render_template('index.html', result = fin_result)

    if request.method == 'POST':
        f = request.files['file']
        name = f.filename
        

        if f is None:
            fin_result = 'error'
    
        else:
            f.save(f'uploads/{secure_filename(name)}')
            file_path = os.path.join('./uploads', name )

            try:
                L, a, b, S, V = pcolor_analysis( file_path )
                x = np.array([L, a, b, S, V]).reshape((1,5))
                fin_result = rfc.predict(x)

                if fin_result == "봄라이트":
                    src = 'https://raw.githubusercontent.com/noel-sumini/personalcolorAI/master/color_pjt/color_palate/spring_light.png'
                    fin_result = '봄 웜 라이트'

                elif fin_result == '봄브라이트':
                    src = 'https://raw.githubusercontent.com/noel-sumini/personalcolorAI/master/color_pjt/color_palate/spring_bright.png'
                    fin_result = '봄 웜 브라이트'

                elif fin_result == '여름라이트':
                    src = 'https://raw.githubusercontent.com/noel-sumini/personalcolorAI/master/color_pjt/color_palate/summer_light.png'
                    fin_result = '여름 쿨 라이트'

                elif fin_result == '여름뮤트':
                    src = 'https://raw.githubusercontent.com/noel-sumini/personalcolorAI/master/color_pjt/color_palate/summer_mute.png'
                    fin_result = '여름 쿨 뮤트'

                elif fin_result == '가을뮤트':
                    src = 'https://raw.githubusercontent.com/noel-sumini/personalcolorAI/master/color_pjt/color_palate/fall_mute.png'
                    fin_result = '가을 웜 뮤트'

                elif fin_result == '가을딥':
                    src = 'https://raw.githubusercontent.com/noel-sumini/personalcolorAI/master/color_pjt/color_palate/fall_deep.png'
                    fin_result = '가을 웜 딥'

                elif fin_result == '겨울브라이트':
                    src = 'https://raw.githubusercontent.com/noel-sumini/personalcolorAI/master/color_pjt/color_palate/winter_bright.png'
                    fin_result = '겨울 쿨 브라이트'

                elif fin_result == '겨울딥':
                    src = 'https://raw.githubusercontent.com/noel-sumini/personalcolorAI/master/color_pjt/color_palate/winter_deep.png'
                    fin_result = '겨울 쿨 딥'
                    


            except:
                fin_result = "얼굴/눈 인식에 실패하였습니다. 얼굴/눈이 또렷히 보이는 사진을 다시 준비해주세요!"
        
    return render_template('result.html', 
                            result = fin_result, 
                            src = src)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port = '80')
    
