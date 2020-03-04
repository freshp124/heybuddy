from fastai.vision import *
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFront
from keras import models
print('Loaded')

path = Path('.')
#Load the saved model
learn = load_learner(path)
print('Model loaded')
video = cv2.VideoCapture(0)
font_type = ImageFont.truetype('Arial.ttf', 28)

defaults.device = torch.device('cuda')

while True:
	_, frame = video.read()

	#Convert the captured frame into RGB
#	im = Image.fromarray(frame)
	im = Image.fromarray(frame, 'RGB')
	    
	#Resizing into 128x128 because we trained the model with this image size.
	im = im.resize((128,128))
	t = torch.tensor(np.ascontiguousarray(np.flip(im, 2)).transpose(2,0,1)).float()/255
	cleanimg = vision.Image(t)

#	img_array = np.array(im)

#	#Our keras model used a 4D tensor, (images x height x width x channel)
#	#So changing dimension 128x128x3 into 1x128x128x3 
#	img_array = np.expand_dims(img_array, axis=0)
	
	#Calling the predict method on model to predict 'me' on the image
	pred_class,pred_idx,outputs = learn.predict(cleanimg)
	
	#if prediction is 0, which mean I am missing on the image, then show the frame in gray color.
	
	#label frame with prediction
	draw = ImageDraw.Draw(frame)
	draw.text(xy=(50,50), text=str(pred_class), fill=(256,0,0), font=font_type)
	
	cv2.imshow("Capturing", frame)
	key=cv2.waitKey(1)
	if key == ord('q'):
		break
video.release()
cv2.destroyAllWindows()
