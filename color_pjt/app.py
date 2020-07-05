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
    h, w, d = image.shape

    if w > 400:
        r = 400.0 / w
        dim = (400, int(h * r))
        image = cv2.resize(image,dim)

    faces = detector(image)

    ## 얼굴 이미지 추출
    for face in faces:
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


result_list = []
value_data = []

for idx, filePath in enumerate(file_paths):
    train_imagePaths = list(paths.list_images(os.path.join('./train_dataset', filePath)))
    
    for imagePath in train_imagePaths:
        print(imagePath)

        try :
            L, a, b, S, V = pcolor_analysis(imagePath)
        except :
            os.remove(imagePath)

        temp = [L, a, b, S, V]
        value_data.append(temp)  

        label.append(filePath)
        _, season, detail = filePath.split('_')
        result_temp = season + detail
        result_list.append(result_temp)

## Scikit-Learn 모델 학습

print("AI분석중")
print("=" * 20)

rfc = RandomForestClassifier()
rfc.fit( np.array(value_data), np.array(result_list))

## Flask 

@app.route('/', methods =['GET', 'POST'])
def result():
    fin_result = ""
    final_result = ""
    color_result_src_default_url = "https://raw.githubusercontent.com/noel-sumini/personalcolorAI/master/color_pjt/color_palate/"
    color_result_src_1 =""
    color_result_src_2 =""
    color_result_src_3 =""
    color_result_src_4 =""
    color_result_src_5 =""
    celeb_name1 = "" 
    celeb_name2 = ""
    celeb_name3 =  ""
    celeb_data1 = ""
    celeb_data2 = ""
    celeb_img_default_url = 'https://raw.githubusercontent.com/noel-sumini/personalcolorAI/master/color_pjt/celeb_image/'
    celeb_img_1 = ""
    celeb_img_2 = ""
    celeb_img_3 = ""
    result1 = ""
    result2 = ""

    if request.method == 'GET':
        intro1 = "파일 선택 후 제출 버튼을 눌러주세요"
        intro2 = "AI가 당신을 위한 퍼스널컬러 분석을 시작합니다"

        return render_template('index.html', 
                            intro1 = intro1, intro2 = intro2,
                            celeb_img_1 = celeb_img_1,
                            celeb_img_2 = celeb_img_2,
                            celeb_img_3 = celeb_img_3)


    if request.method == 'POST':
        intro1 = "재 분석을 원하시면 다시 파일 선택 후 제출 버튼을 눌러주세요"
        
        f = request.files['file']
        name = f.filename

        f.save(f'uploads/{secure_filename(name)}')
        file_path = os.path.join('./uploads', name )
        
        print(file_path)

        try:
            L, a, b, S, V = pcolor_analysis( file_path )
            x = np.array([L, a, b, S, V]).reshape((1,5))
            fin_result = rfc.predict(x)

            print(fin_result)

            

            if fin_result == '봄라이트':
                color_result_src_1 = color_result_src_default_url + 'spring_light/1.jpg'
                color_result_src_2 = color_result_src_default_url + 'spring_light/2.jpg'
                color_result_src_3 = color_result_src_default_url + 'spring_light/3.jpg'
                color_result_src_4 = color_result_src_default_url + 'spring_light/4.jpg'
                color_result_src_5 = color_result_src_default_url + 'spring_light/5.jpg'
                # celeb_data = f'당신과 같은 {fin_result} 톤 연예인은?'
                celeb_name1, celeb_name2, celeb_name3 = '윤아 정소민 박보영'.split(" ")
                celeb_img_1 = celeb_img_default_url + 'spring_light/1.jpg'
                celeb_img_2 = celeb_img_default_url + 'spring_light/2.jpg'
                celeb_img_3 = celeb_img_default_url + 'spring_light/3.jpg'
                
            elif fin_result == '봄브라이트':
                color_result_src_1 = color_result_src_default_url + 'spring_bright/1.jpg'
                color_result_src_2 = color_result_src_default_url + 'spring_bright/2.jpg'
                color_result_src_3 = color_result_src_default_url + 'spring_bright/3.jpg'
                color_result_src_4 = color_result_src_default_url + 'spring_bright/4.jpg'
                color_result_src_5 = color_result_src_default_url + 'spring_bright/5.jpg'
                
                # celeb_data = f'당신과 같은 {fin_result} 톤 연예인은?'
                celeb_name1, celeb_name2, celeb_name3 = '조이 아이유 송혜교'.split(" ")
                celeb_img_1 = celeb_img_default_url + 'spring_bright/1.jpg'
                celeb_img_2 = celeb_img_default_url + 'spring_bright/2.jpg'
                celeb_img_3 = celeb_img_default_url + 'spring_bright/3.jpg'

            elif fin_result == '여름라이트':
                color_result_src_1 = color_result_src_default_url + 'summer_light/1.jpg'
                color_result_src_2 = color_result_src_default_url + 'summer_light/2.jpg'
                color_result_src_3 = color_result_src_default_url + 'summer_light/3.jpg'
                color_result_src_4 = color_result_src_default_url + 'summer_light/4.jpg'
                color_result_src_5 = color_result_src_default_url + 'summer_light/5.jpg'

                
                # celeb_data = f'당신과 같은 {fin_result} 톤 연예인은?'
                celeb_name1, celeb_name2, celeb_name3 = '아이린 정채연 다현'.split(" ")
                celeb_img_1 = celeb_img_default_url + 'summer_light/1.jpg'
                celeb_img_2 = celeb_img_default_url + 'summer_light/2.jpg'
                celeb_img_3 = celeb_img_default_url + 'summer_light/3.jpg'

            elif fin_result == '여름뮤트':
                color_result_src_1 = color_result_src_default_url + 'summer_mute/1.jpg'
                color_result_src_2 = color_result_src_default_url + 'summer_mute/2.jpg'
                color_result_src_3 = color_result_src_default_url + 'summer_mute/3.jpg'
                color_result_src_4 = color_result_src_default_url + 'summer_mute/4.jpg'
                color_result_src_5 = color_result_src_default_url + 'summer_mute/5.jpg'

                # celeb_data = f'당신과 같은 {fin_result} 톤 연예인은?'
                celeb_name1, celeb_name2, celeb_name3 = '김태리 김연아 김고은'.split(" ")
                celeb_img_1 = celeb_img_default_url + 'summer_mute/1.jpg'
                celeb_img_2 = celeb_img_default_url + 'summer_mute/2.jpg'
                celeb_img_3 = celeb_img_default_url + 'summer_mute/3.jpg'

            elif fin_result == '가을뮤트':
                color_result_src_1 = color_result_src_default_url + 'fall_mute/1.jpg'
                color_result_src_2 = color_result_src_default_url + 'fall_mute/2.jpg'
                color_result_src_3 = color_result_src_default_url + 'fall_mute/3.jpg'
                color_result_src_4 = color_result_src_default_url + 'fall_mute/4.jpg'
                color_result_src_5 = color_result_src_default_url + 'fall_mute/5.jpg'

                # celeb_data = f'당신과 같은 {fin_result} 톤 연예인은?'
                celeb_name1, celeb_name2, celeb_name3 = '제니 이효리 박서준'.split(" ")
                celeb_img_1 = celeb_img_default_url + 'fall_mute/1.jpg'
                celeb_img_2 = celeb_img_default_url + 'fall_mute/2.jpg'
                celeb_img_3 = celeb_img_default_url + 'fall_mute/3.jpg'

            elif fin_result == '가을딥':
                color_result_src_1 = color_result_src_default_url + 'fall_deep/1.jpg'
                color_result_src_2 = color_result_src_default_url + 'fall_deep/2.jpg'
                color_result_src_3 = color_result_src_default_url + 'fall_deep/3.jpg'
                color_result_src_4 = color_result_src_default_url + 'fall_deep/4.jpg'
                color_result_src_5 = color_result_src_default_url + 'fall_deep/5.jpg'
                
                # celeb_data = f'당신과 같은 {fin_result} 톤 연예인은?'
                celeb_name1, celeb_name2, celeb_name3 = '케이 세정 쯔위'.split(" ")
                celeb_img_1 = celeb_img_default_url + 'fall_deep/1.jpg'
                celeb_img_2 = celeb_img_default_url + 'fall_deep/2.jpg'
                celeb_img_3 = celeb_img_default_url + 'fall_deep/3.jpg'

            elif fin_result == '겨울브라이트':
                color_result_src_1 = color_result_src_default_url + 'winter_bright/1.jpg'
                color_result_src_2 = color_result_src_default_url + 'winter_bright/2.jpg'
                color_result_src_3 = color_result_src_default_url + 'winter_bright/3.jpg'
                color_result_src_4 = color_result_src_default_url + 'winter_bright/4.jpg'
                color_result_src_5 = color_result_src_default_url + 'winter_bright/5.jpg'

                # celeb_data = f'당신과 같은 {fin_result} 톤 연예인은?'
                celeb_name1, celeb_name2, celeb_name3 = '지수 채영 트와이스미나'.split(" ")
                celeb_img_1 = celeb_img_default_url + 'winter_bright/1.jpg'
                celeb_img_2 = celeb_img_default_url + 'winter_bright/2.jpg'
                celeb_img_3 = celeb_img_default_url + 'winter_bright/3.jpg'

            elif fin_result == '겨울딥':
                color_result_src_1 = color_result_src_default_url + 'winter_deep/1.jpg'
                color_result_src_2 = color_result_src_default_url + 'winter_deep/2.jpg'
                color_result_src_3 = color_result_src_default_url + 'winter_deep/3.jpg'
                color_result_src_4 = color_result_src_default_url + 'winter_deep/4.jpg'
                color_result_src_5 = color_result_src_default_url + 'winter_deep/5.jpg'


                # celeb_data = f'당신과 같은 {fin_result} 톤 연예인은?'
                celeb_name1, celeb_name2, celeb_name3 = '선미 권은비 찬미'.split(" ")
                celeb_img_1 = celeb_img_default_url + 'winter_deep/1.jpg'
                celeb_img_2 = celeb_img_default_url + 'winter_deep/2.jpg'
                celeb_img_3 = celeb_img_default_url + 'winter_deep/3.jpg'

            # fin_result = fin_result.replace("봄", "봄 웜 ")
            # fin_result = fin_result.replace("여름", "여름 쿨 ")
            # fin_result = fin_result.replace("가을", "가을 웜 ")
            # fin_result = fin_result.replace("겨울", "겨울 쿨 ")

            result1 = '당신의 퍼스널 컬러는'
            result2 =  '톤 입니다.' 
            celeb_data1 = '당신과 같은 '
            celeb_data2 = '톤 연예인은? '
            fin_result = str(fin_result)
            fin_result = fin_result.replace("'","")
            fin_result = fin_result.replace('[','"')
            fin_result = fin_result.replace(']','"')



        except:
            fin_result = "얼굴/눈 인식에 실패하였습니다. \n 얼굴/눈이 또렷히 보이는 사진을 다시 준비해주세요!"
        
        if os.path.isfile(file_path):
            os.remove(file_path)
    
   

    return render_template('result.html', 
                            intro1 = intro1,
                            result = fin_result, 
                            result1 = result1,
                            result2 = result2,
                            celeb_data1 = celeb_data1,
                            celeb_data2 = celeb_data2,
                            celeb_name1 = celeb_name1,
                            celeb_name2 = celeb_name2,
                            celeb_name3 = celeb_name3,
                            celeb_img_1 = celeb_img_1,
                            celeb_img_2 = celeb_img_2,
                            celeb_img_3 = celeb_img_3,
                            color_result_src_1 = color_result_src_1,
                            color_result_src_2 = color_result_src_2,
                            color_result_src_3 = color_result_src_3,
                            color_result_src_4 = color_result_src_4,
                            color_result_src_5 = color_result_src_5)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port = '80', threaded=True)
    
