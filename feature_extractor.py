import os
import pickle
import cv2, numpy as np
from keras.models import Model
from keras.models import load_model
import keras

CNN_weights_file_name = r'D:/Office/MS/finding_sign_on_form/keras_model/trained_models/model_v3.h5'
model = load_model(CNN_weights_file_name)

def get_image_model(model):
	# for layer in model.layers:
	#     print(layer.name)
	model_new = Model(inputs=model.input,
	                  outputs=model.get_layer('dense_1').output)

	return model_new


fe_model = get_image_model(model)

def get_image_features(model_new, image):
	# image_features = np.zeros((512))
	test_img = cv2.resize(image, (100, 32))
	test_img = cv2.threshold(test_img, 0, 255, cv2.THRESH_OTSU)[1]
	data = test_img.reshape(-1, 100, 32, 1)
	data = data.astype('float')
	data /= 255
	image_features = model_new.predict(data)
	return image_features


image_features_list=[]

PATH = 'D:/Office/MS/finding_sign_on_form/keras_model/new_train_data/'
images = [img for img in os.listdir(PATH) if img.endswith('.jpg') or img.endswith('.JPG')]


for image in images:
	image = cv2.imread(os.path.join(PATH, image), 0)
	image_features=get_image_features(fe_model, image)
	image_features_list.append(image_features)

image_features_arr=np.asarray(image_features_list,dtype='f')
# image_features_arr = image_features_arr.reshape(-1,3200)
print(image_features_arr[0].shape)
print(image_features_arr.shape)
image_features_arr = np.rollaxis(image_features_arr,1,0)
image_features_arr = image_features_arr[0,:,:]
print(image_features_arr.shape)

np.savetxt('./feature_files/feature_vectors_4000_samples.txt',image_features_arr)
#feature_vectors = np.loadtxt('feature_vectors.txt')
pickle.dump(image_features_arr, open('./feature_files/feature_vectors_400_samples.pkl', 'wb'))





