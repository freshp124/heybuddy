#!/usr/bin/env python
# coding: utf-8

# # Clone Rep

# In[ ]:


get_ipython().system('git clone https://github.com/fastai/fastai.git ../fastai')


# # Import

# In[ ]:


#%%capture
#conda install pillow=6.1
from fastai.vision import *
import cv2
import numpy as np
import time
from PIL import Image, ImageDraw, ImageFont
from keras import models
print('Loaded')


# # Load the model

# In[ ]:


#Load the saved model
path = Path('.')
learn = load_learner(path)
defaults.device = torch.device('cuda')
print('Model loaded')


# # Initialize Variables and Font

# In[ ]:


font_type = ImageFont.truetype('arial.ttf', 28)
leaveupfor = 0


# In[ ]:


fontscale = 1.5
# (B, G, R)
color = (0, 0, 255)
# select font
fontface = cv2.FONT_HERSHEY_COMPLEX
thickness = 5


# # Test Model

# video = cv2.VideoCapture(0)
# _, frame = video.read()
# im = Image.fromarray(frame, 'RGB')
# im = im.resize((128,128))
# t = torch.tensor(np.ascontiguousarray(np.flip(im, 2)).transpose(2,0,1)).float()/255
# cleanimg = vision.Image(t)
# 
# pred_class,pred_idx,outputs = learn.predict(cleanimg)
# video.release()

# per = outputs[0]
# #per = outputs.numpy()
# #per = np.squeeze(outputs)
# per.item()
# str(pred_class)
# greatest = outputs.max()
# greatest.item()

# # Launch camera capture and detector

# In[ ]:


video = cv2.VideoCapture(0)

while True:
        _, frame = video.read()
        #Convert the captured frame into RGB
        #im = Image.fromarray(frame)
        im = Image.fromarray(frame, 'RGB')

        #Resizing into 128x128 because we trained the model with this image size.
        im = im.resize((128,128))
        t = torch.tensor(np.ascontiguousarray(np.flip(im, 2)).transpose(2,0,1)).float()/255
        cleanimg = vision.Image(t)
    #img_array = np.array(im)
    ##Our keras model used a 4D tensor, (images x height x width x channel)
    ##So changing dimension 128x128x3 into 1x128x128x3 
    #img_array = np.expand_dims(img_array, axis=0)

        #Calling the predict method on model to predict 'me' on the image
        pred_class,pred_idx,outputs = learn.predict(cleanimg)
        greatest = outputs.max()
        if str(pred_class) != 'nodog':
            if greatest.item() > .6:
            #add text
                commandlabel = str(pred_class) + " " + str(greatest.item())
                #cv2.putText(frame, commandlabel, (25, 40), fontface, fontscale, color, thickness)
                #cv2.imshow("Capturing", frame)
                #time.sleep(1)
                leaveupfor = 60
        if leaveupfor > 0:
            cv2.putText(frame, commandlabel, (25, 40), fontface, fontscale, color, thickness)
        cv2.imshow("Capturing", frame)
        leaveupfor = leaveupfor - 1
        key=cv2.waitKey(1)
        if key == ord('q'):
            break
video.release()
cv2.destroyAllWindows()


# # Re-release webcam
# For troubleshooting purposes.

# In[ ]:


video.release()
cv2.destroyAllWindows()


# # Convert notebook to script

# In[1]:


import sys
get_ipython().system('{sys.executable} -m jupyter nbconvert --to python detect_dog.ipynb')


# In[ ]:




