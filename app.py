from flask import Flask, request, jsonify
import cv2
import os
import numpy as np
from tqdm import tqdm
import pandas as pd
import time
import json
from PIL import Image


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from deepface import DeepFace
from deepface.extendedmodels import Age
from deepface.commons import functions, realtime, distance as dst
from deepface.detectors import FaceDetector

model_name = 'Facenet512'
# model_name = 'VGG-Face'
detector_backend = 'retinaface'
distance_metric = 'cosine'
db_path = 'database/'


def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    if request.method == 'OPTIONS':
        response.headers['Access-Control-Allow-Methods'] = 'DELETE, GET, POST, PUT'
        headers = request.headers.get('Access-Control-Request-Headers')
        if headers:
            response.headers['Access-Control-Allow-Headers'] = headers
    return response

face_detector = FaceDetector.build_model(detector_backend)
print("Detector backend is ", detector_backend)

#------------------------

input_shape = (224, 224); input_shape_x = input_shape[0]; input_shape_y = input_shape[1]

text_color = (255,255,255)

employees = []
#check passed db folder exists
if os.path.isdir(db_path) == True:
    for r, d, f in os.walk(db_path): # r=root, d=directories, f = files
        for file in f:
            if ('.jpg' in file):
                #exact_path = os.path.join(r, file)
                exact_path = r + "/" + file
                #print(exact_path)
                employees.append(exact_path)

if len(employees) == 0:
    print("WARNING: There is no image in this path ( ", db_path,") . Face recognition will not be performed.")

#------------------------

if len(employees) > 0:

    model = DeepFace.build_model(model_name)
    print(model_name," is built")

    #------------------------

    input_shape = functions.find_input_shape(model)
    input_shape_x = input_shape[0]; input_shape_y = input_shape[1]

    #tuned thresholds for model and metric pair
    threshold = dst.findThreshold(model_name, distance_metric)

#------------------------
#facial attribute analysis models



tic = time.time()

emotion_model = DeepFace.build_model('Emotion')
print("Emotion model loaded")

age_model = DeepFace.build_model('Age')
print("Age model loaded")

gender_model = DeepFace.build_model('Gender')
print("Gender model loaded")

toc = time.time()

print("Facial attibute analysis models loaded in ",toc-tic," seconds")

#------------------------

#find embeddings for employee list

tic = time.time()

#-----------------------

pbar = tqdm(range(0, len(employees)), desc='Finding embeddings')

#TODO: why don't you store those embeddings in a pickle file similar to find function?

embeddings = []
#for employee in employees:
for index in pbar:
    employee = employees[index]
    pbar.set_description("Finding embedding for %s" % (employee.split("/")[-1]))
    embedding = []

    #preprocess_face returns single face. this is expected for source images in db.
    img = functions.preprocess_face(img = employee, target_size = (input_shape_y, input_shape_x), enforce_detection = False, detector_backend = detector_backend)
    img_representation = model.predict(img)[0,:]

    embedding.append(employee)
    embedding.append(img_representation)
    embeddings.append(embedding)

dfr = pd.DataFrame(embeddings, columns = ['employee', 'embedding'])
dfr['distance_metric'] = distance_metric

toc = time.time()

print("Embeddings found for given data set in ", toc-tic," seconds")

#-----------------------

pivot_img_size = 112 #face recognition result image

#-----------------------


app = Flask(__name__)
app.after_request(add_cors_headers)


@app.route("/")
def home():
    return "Ok", 200


@app.route("/recognize", methods=["POST"])
def recognize():
    # data = request.get_json()[23:]
    # cv2.imwrite("output.jpg",(string_to_image(data)))
    data = request.files['image']
    img = Image.open(data)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    faces = FaceDetector.detect_faces(face_detector, detector_backend, img, align = False)
    # print ("Found {0} faces!".format(len(faces[1])))
    detected_faces = 0
    face_dict={}
    
    

    for face, (x, y, w, h) in faces:
        # face_image = image[int(y):int(y+h), int(x):int(x+w)]
        # detected_faces.append((x,y,w,h))
        # cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # cv2.imwrite(res, face_image)
        # obj = DeepFace.analyze(img_path = res, actions = ['age', 'gender', 'emotion'], enforce_detection=False)
        # df = DeepFace.find(img_path = res, db_path = "database/", enforce_detection=False)
        # print(obj)
        # print(df)
    # for detected_face in detected_faces:
        # print('--------------------------------')
        # x = detected_face[0]; y = detected_face[1]
        # w = detected_face[2]; h = detected_face[3]
        face_image = img[y:y+h, x:x+w]
        #emtion prediction
        gray_img = functions.preprocess_face(img = face_image, target_size = (48, 48), grayscale = True, enforce_detection = False, detector_backend = 'opencv')
        emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        emotion_predictions = emotion_model.predict(gray_img)[0,:]
        sum_of_predictions = emotion_predictions.sum()

        mood_items = []
        for i in range(0, len(emotion_labels)):
            mood_item = []
            emotion_label = emotion_labels[i]
            emotion_prediction = 100 * emotion_predictions[i] / sum_of_predictions
            mood_item.append(emotion_label)
            mood_item.append(emotion_prediction)
            mood_items.append(mood_item)

        emotion_df = pd.DataFrame(mood_items, columns = ["emotion", "score"])
        emotion_df = emotion_df.sort_values(by = ["score"], ascending=False).reset_index(drop=True)

        #age prediction
        face_224 = functions.preprocess_face(img = face_image, target_size = (224, 224), grayscale = False, enforce_detection = False, detector_backend = 'opencv')

        age_predictions = age_model.predict(face_224)[0,:]
        apparent_age = Age.findApparentAge(age_predictions)

        #gender prediction
        gender_prediction = gender_model.predict(face_224)[0,:]

        if np.argmax(gender_prediction) == 0:
            gender = "Woman"
        elif np.argmax(gender_prediction) == 1:
            gender = "Man"

        # print(str(int(apparent_age))+" "+gender+" "+emotion_df["emotion"][0])

        #face recognition
        face_image = functions.preprocess_face(img = face_image, target_size = (input_shape_y, input_shape_x), enforce_detection = False, detector_backend = 'opencv')
        if face_image.shape[1:3] == input_shape:
            if dfr.shape[0] > 0: #if there are images to verify, apply face recognition
                img1_representation = model.predict(face_image)[0,:]

                #print(freezed_frame," - ",img1_representation[0:5])

                def findDistance(row):
                    distance_metric = row['distance_metric']
                    img2_representation = row['embedding']

                    distance = 1000 #initialize very large value
                    if distance_metric == 'cosine':
                        distance = dst.findCosineDistance(img1_representation, img2_representation)
                    elif distance_metric == 'euclidean':
                        distance = dst.findEuclideanDistance(img1_representation, img2_representation)
                    elif distance_metric == 'euclidean_l2':
                        distance = dst.findEuclideanDistance(dst.l2_normalize(img1_representation), dst.l2_normalize(img2_representation))

                    return distance

                dfr['distance'] = dfr.apply(findDistance, axis = 1)

                df = dfr.sort_values(by = ["distance"])

                candidate = df.iloc[0]
                employee_name = candidate['employee']
                best_distance = candidate['distance']

                if employee_name.find("Tran Duc Hung") != -1:
                    employee_name = 'Tran Duc Hung'
                elif employee_name.find("Tran Trung Hau") != -1:
                    employee_name = 'Tran Trung Hau'
                elif employee_name.find("Do Thai Son") != -1:
                    employee_name = 'Do Thai Son'
                elif employee_name.find("Nguyen Manh Tien Anh") != -1:
                    employee_name = 'Nguyen Manh Tien Anh'
                elif employee_name.find("Dao Vinh Linh") != -1:
                    employee_name = 'Dao Vinh Linh'
                elif employee_name.find("Dang Tuan Linh") != -1:
                    employee_name = 'Dr. Dang Tuan Linh'
                else:
                    employee_name = 'Bui Tung Anh'


                #print(candidate[['employee', 'distance']].values)

                if best_distance > threshold:
                    employee_name = 'Unknown'

        face_inf = {
            'region': str((x, y, w, h)),
            'name': employee_name,
            'age': str(int(apparent_age)),
            'gender': gender,
            'emotion': emotion_df["emotion"][0]
        }
        face_dict[detected_faces] = face_inf
        detected_faces+=1

    # print('Found ', detected_faces, ' faces')

    # print(face_dict)
    # with open("sample.json", "w") as outfile:
    #     json.dump(face_dict, outfile)
        
    return jsonify(face_dict)


if __name__ == '__main__':
    app.run(debug = True)
