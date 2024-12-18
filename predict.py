from deepface import DeepFace
import cv2

#img1 = cv2.imread('photo.jpg')

def analyze_emotion(img1):
    result = DeepFace.analyze(img1,actions=['emotion'])
    #print(result)
    emotion_data = result[0]['emotion']
    max_emotion_key = max(emotion_data, key=emotion_data.get)
    max_emotion_value = emotion_data[max_emotion_key]
    #print(f"The emotion predicted is {max_emotion}")
    return max_emotion_key,max_emotion_value

