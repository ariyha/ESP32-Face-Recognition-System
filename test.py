import cv2
import mediapipe as mp
import face_recognition
import numpy as np
import os
import firebase_admin
from firebase_admin import credentials, messaging, firestore,storage
from datetime import datetime, timedelta
import time
import threading

# Initialize Firebase app with credentials
cred = credentials.Certificate("PROJECT/esp32-9e776-firebase-adminsdk-24lzj-b0eb618d5a.json")
app = firebase_admin.initialize_app(cred,{
    'storageBucket': 'esp32-9e776.appspot.com'  # Replace with your actual bucket name
})

#send message via firebase for push noti
def sendPush(title, msg, registration_token, dataObject=None):
    message = messaging.MulticastMessage(
        notification=messaging.Notification(
            title=title,
            body=msg
        ),
        data=dataObject,
        tokens=registration_token,
    )
    response = messaging.send_multicast(message)
    print('Successfully sent message:', response)

#get FCM token from DB
def get_Token():
    db = firestore.client()
    doc = db.collection('tokens').get()
    tokens = [i.to_dict()['fcmToken'] for i in doc]
    return tokens

door_unlocked = False
camera_running = threading.Event()

#load faces for recognition
def load_known_faces(known_faces_dir):
    known_face_encodings = []
    known_face_names = []
    for name in os.listdir(known_faces_dir):
        if os.path.isdir(os.path.join(known_faces_dir, name)):
            for image_file in os.listdir(os.path.join(known_faces_dir, name)):
                image_path = os.path.join(known_faces_dir, name, image_file)
                image = face_recognition.load_image_file(image_path)
                encoding = face_recognition.face_encodings(image)[0]
                known_face_encodings.append(encoding)
                known_face_names.append(name)
    return known_face_encodings, known_face_names

#upload image to firebase (face data)
def upload_image_to_firebase(face_image, name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_name = f"{name}_{timestamp}.jpg"
    bucket = storage.bucket()
    blob = bucket.blob(image_name)
    _, image_encoded = cv2.imencode('.jpg', cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR))
    blob.upload_from_string(image_encoded.tobytes(), content_type='image/jpeg')
    image_url = blob.generate_signed_url(timedelta(days=365), method='GET')

    db = firestore.client()
    doc_ref = db.collection('recognized_faces').add({
        'name': name,
        'image_url': image_url,
        'timestamp': datetime.now()
    })
    print(f"Uploaded {image_name} to Firebase Storage with URL: {image_url}")

#save recognized face
def save_recognized_face(face_image, name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"C:/Users/nithi/Documents/IoT/{name}_{timestamp}.jpg"
    print(save_path)
    cv2.imwrite(save_path, cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR))

#total camera system takes video and does work
def camera_system():
    global door_unlocked, camera_running

    known_faces_dir = 'PROJECT/p1'
    known_face_encodings, known_face_names = load_known_faces(known_faces_dir)
    mp_face_detection = mp.solutions.face_detection
    cap = cv2.VideoCapture("http://192.168.6.195:81/stream")
    last_recognized_time = {name: datetime.min for name in known_face_names}

    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        while camera_running.is_set() and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb_frame)
            face_detected = False

            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                                 int(bboxC.width * iw), int(bboxC.height * ih)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    face_roi = np.ascontiguousarray(rgb_frame[y:y + h, x:x + w])
                    face_encodings = face_recognition.face_encodings(face_roi)
                    if face_encodings:
                        face_encoding = face_encodings[0]
                        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                        best_match_index = np.argmin(face_distances)
                        current_time = datetime.now()
                        if matches[best_match_index]:
                            name = known_face_names[best_match_index]
                            last_recognized_time[name] = current_time
                            tokens = get_Token()
                            sendPush("Known Person Detected", f"{name} Detected", tokens)
                            upload_image_to_firebase(face_roi, name)
                            db = firestore.client()
                            db.collection('flags').document('lock').set({'yes': False})
                            door_unlocked = True
                            print("Face detected. Door unlocked.")
                            camera_running.clear()
                        else:
                            name = "Unknown"
                            tokens = get_Token()
                            sendPush("Unknown Person Detected", "Unknown Person Detected", tokens)
                            upload_image_to_firebase(face_roi, name)
                            print("Unknown face detected. Pausing for one minute.")
                            time.sleep(60)
                        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            cv2.imshow('MediaPipe Face Recognition', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

#lock system
def door_lock_change():
    global door_unlocked, camera_running, camera_thread
    db = firestore.client()
    while True:
        doc = db.collection('flags').document('lock').get()
        lock_status = not doc.to_dict().get('yes', False)
        if lock_status != door_unlocked:
            door_unlocked = lock_status
            if door_unlocked and camera_running.is_set():
                print("Door unlocked. Stopping camera system.")
                camera_running.clear()
                camera_thread.join()
            elif not door_unlocked and not camera_running.is_set():
                print("Door locked. Starting camera system.")
                camera_running.set()
                camera_thread = threading.Thread(target=camera_system)
                camera_thread.start()
        time.sleep(1)

# Start the door lock change monitoring thread
camera_running.set()
camera_thread = threading.Thread(target=camera_system)
camera_thread.start()

door_lock_thread = threading.Thread(target=door_lock_change)
door_lock_thread.start()