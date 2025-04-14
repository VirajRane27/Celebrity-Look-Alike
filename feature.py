# # import os
# # from tensorflow.keras.applications import VGG16
# # from tensorflow.keras.applications import InceptionResNetV2
# # from keras_vggface.vggface import VGGFace
# # from tensorflow.keras.preprocessing import image
# # from tensorflow.keras.applications.vgg16 import preprocess_input
# # from facenet_pytorch import InceptionResnetV1, MTCNN
# # import torch
# import os
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np
# import pickle
# import cv2
# from PIL import Image
# from mtcnn import MTCNN
# from tensorflow.keras.applications import InceptionResNetV2
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
# import streamlit as st


# detect = MTCNN()
# celeb = os.listdir('Celeb Faces')

# # print(celeb)

# path = []

# for i in celeb:
#     for j in os.listdir(os.path.join('Celeb Faces', i)):
#         path.append(os.path.join('Celeb Faces', i, j))

# # print(len(path))

# # model = InceptionResnetV1(pretrained='vggface2').eval()
# model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='avg')
# # model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
# # model = VGGFace(model='vggface', include_top=False, input_shape=(224, 224, 3))

# # features = []
# # def feature_path(path, model):
# #     img = image.load_img(path, target_size=(224, 224))
# #     img = image.img_to_array(img)
# #     img = np.expand_dims(img, axis=0)
# #     img = preprocess_input(img)
# #     features = model.predict(img).flatten()
# #     return features

# # features3 = []
# def feature_img(img, model):
#   img = np.array(img)
#   img = cv2.resize(img, (224, 224))
#   img = image.img_to_array(img)
#   img = np.expand_dims(img, axis=0)
#   img = preprocess_input(img)
#   features = model.predict(img).flatten()
#   return features

# # def feature_img(img, model):
# #     img = mtcnn(img)
# #     if img is not None:
# #         with torch.no_grad():
# #             res = model(img).cpu().numpy()
# #             return res

# def valid(img):
#     try:
#         face = detect.detect_faces(np.array(img))
#         confi = face[0]['confidence']
#         if face and confi > 0.9:
#             return 1
#         else:
#             return -1
#     except Exception as e:
#         st.error("The uploaded image is not valid for face detection. Please try again with a clear image.")
#         return -2

# def crop(img):
#     faces = detect.detect_faces(np.array(img))
#     if faces:
#         x, y, width, height = faces[0]['box']
#         img = np.array(img)
#         img = img[y:y+height, x:x+width]
#         return img
#     return None

# def recommend(feature, a):
#     similarty = []
#     for i in range(len(a)):
#      similarty.append(cosine_similarity(feature.reshape(1,-1), a[i].reshape(1,-1))[0][0])
#     index = sorted(list(enumerate(similarty)), reverse=True, key=lambda x:x[1])[0][0]
#     return index

# # features3 = []
# # for i in range(len(path)):
# #   img = Image.open(path[i])
# #   features3.append(feature_img(img, model))
# #   print(len(features3))
# # pickle.dump(features3, open('features3.pkl', 'wb'))

# a = pickle.load(open('features3.pkl', 'rb'))


# st.title('Which Celebrity are You ?')

# your_image = st.file_uploader('Upload your Image', type=['jpg', 'jpeg', 'png'])

# if your_image is not None:
#     img = Image.open(your_image)
#     if valid(img) == 1:
#         img = crop(img)
#         if img is not None:
#             feature = feature_img(img, model)
#             index = recommend(feature, a)

#             col1, col2 = st.columns(2)

#             with col1:
#                 st.header('Your Uploaded Photo')
#                 your_image = Image.open(your_image)
#                 your_image = your_image.resize((250, 300))
#                 st.image(your_image)

#             with col2:
#                 name = " ".join(path[index].split('\\')[1].split('_'))
#                 st.header('You look like ' + name)
#                 celeb = Image.open(path[index])
#                 celeb = celeb.resize((250, 300))
#                 st.image(celeb)
#         else:
#             st.header('Image is not clear or face is not recognized')
#             st.header('Try with other Image')
#     else:
#         st.header('Image is not clear or face is not recognized')
#         st.header('Try with other Image')















import os
import pickle
import cv2
import numpy as np
from PIL import Image
from mtcnn import MTCNN
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
import streamlit as st

# Streamlit page config
st.set_page_config(page_title="Celebrity Look-Alike", layout="centered")

# Custom CSS for styling
st.markdown("""
    <style>
    body {
        background: linear-gradient(to right top, #ff6ec4, #7873f5);
        background-attachment: fixed;
    }
    .stApp {
        background-color: transparent;
    }
    .header {
        text-align: center;
        color: white;
        margin-top: 30px;
    }
    .header h1 {
        font-size: 48px;
        margin-bottom: 10px;
    }
    .header p {
        font-size: 20px;
        color: #e0e0e0;
    }
    .label-above {
        text-align: center;
        font-size: 24px;
        font-weight: 800;
        margin-bottom: 10px;
        background: linear-gradient(90deg, #ffffff, #ffd700, #ffffff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 1px 1px 6px rgba(0, 0, 0, 0.5);
    }
    </style>
""", unsafe_allow_html=True)

# App Header
st.markdown('<div class="header"><h1>✨ Which Celebrity Do You Look Like?</h1><p>Upload your photo and find your celebrity twin!</p></div>', unsafe_allow_html=True)

# Load model and data
model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='avg')
detector = MTCNN()

# Load celebrity face paths and precomputed features
celeb_folders = os.listdir('Celeb Faces')
paths = [os.path.join('Celeb Faces', name, img) for name in celeb_folders for img in os.listdir(os.path.join('Celeb Faces', name))]
features = pickle.load(open('features3.pkl', 'rb'))

# Functions
def detect_face(img):
    try:
        result = detector.detect_faces(np.array(img))
        return result[0]['confidence'] > 0.9 if result else False
    except:
        return False

def crop_face(img):
    try:
        face = detector.detect_faces(np.array(img))[0]['box']
        x, y, w, h = face
        return np.array(img)[y:y+h, x:x+w]
    except:
        return None

def get_feature(img, model):
    img = cv2.resize(img, (224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return model.predict(x).flatten()

def find_best_match(user_feature, db_features):
    sims = [cosine_similarity(user_feature.reshape(1, -1), db.reshape(1, -1))[0][0] for db in db_features]
    return np.argmax(sims)

# Upload section
uploaded = st.file_uploader("📤 Upload your image", type=["jpg", "jpeg", "png"])

if uploaded:
    user_img = Image.open(uploaded)

    if detect_face(user_img):
        cropped_face = crop_face(user_img)

        if cropped_face is not None:
            user_feature = get_feature(cropped_face, model)
            match_idx = find_best_match(user_feature, features)

            match_path = paths[match_idx]
            celeb_name = " ".join(os.path.basename(os.path.dirname(match_path)).split("_"))
            celeb_img = Image.open(match_path)

            # Display both images side by side
            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f'<div class="label-above">YOU</div>', unsafe_allow_html=True)
                st.image(user_img.resize((300, 350)))

            with col2:
                st.markdown(f'<div class="label-above">You look like <strong>{celeb_name}</strong></div>', unsafe_allow_html=True)
                st.image(celeb_img.resize((300, 350)))
        else:
            st.error("😔 Face crop failed. Please upload a clearer image.")
    else:
        st.error("😔 No clear face detected. Please try another photo.")
