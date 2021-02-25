# Object Detection using SSD512(Single Shot Multibox Detector)

# Libraries
from keras import backend as K
from keras.preprocessing import image
from keras.optimizers import Adam
import imageio
import numpy as np
import os
import cv2
import threading
import time
# Dependencies
from models.keras_ssd512 import ssd_512
from keras_loss_function.keras_ssd_loss import SSDLoss
from tkinter import *    # Carga módulo tk (widgets estándar)
from tkinter import ttk  # Carga ttk (para widgets nuevos 8.5+)
from tkinter import filedialog
from tkinter import messagebox as MessageBox


# Defining the width and height of the image
height = 512
width = 512

# Defining confidence threshold
confidence_threshold = 0.5

# Different Classes of objects in VOC dataset
classes = ['background',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat',
           'chair', 'cow', 'diningtable', 'dog',
           'horse', 'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor']


K.clear_session() # Clear previous models from memory.

# creating model and loading pretrained weights
model = ssd_512(image_size=(height, width, 3),	# dimensions of the input images (fixed for SSD512)
                n_classes=20,	# Number of classes in VOC 2007 & 2012 dataset
                mode='inference',
                l2_regularization=0.0005,
                scales=[0.07, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05], 
                aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5],
                                         [1.0, 2.0, 0.5]],
               two_boxes_for_ar1=True,
               steps=[8, 16, 32, 64, 128, 256, 512],
               offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
               clip_boxes=False,
               variances=[0.1, 0.1, 0.2, 0.2],
               normalize_coords=True,
               subtract_mean=[123, 117, 104],
               swap_channels=[2, 1, 0],
               confidence_thresh=0.5,
               iou_threshold=0.45,
               top_k=200,
               nms_max_output_size=400)

# path of the pre trained model weights 
weights_path = 'weights/VGG_VOC0712Plus_SSD_512x512_ft_iter_160000.h5'
model.load_weights(weights_path, by_name=True)

# Compiling the model 
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
model.compile(optimizer=adam, loss=ssd_loss.compute_loss)


# Transforming image size
def transform(input_image):
	return cv2.resize(input_image, (512, 512), interpolation = cv2.INTER_CUBIC)

# Function to detect objects in image
def detect_object(original_image):
	original_image_height, original_image_width = original_image.shape[:2]
	input_image = transform(original_image)
	input_image = np.reshape(input_image, (1, 512, 512, 3))
	y_pred = model.predict(input_image)
	actual_prediction = [y_pred[k][y_pred[k,:,1] > confidence_threshold] for k in range(y_pred.shape[0])]
	for box in actual_prediction[0]:
		# Coordinates of diagonal points of bounding box
		x0 = box[-4] * original_image_width / width
		y0 = box[-3] * original_image_height / height
		x1 = box[-2] * original_image_width / width
		y1 = box[-1] * original_image_height / height
		label_text = '{}: {:.2f}'.format(classes[int(box[0])], box[1])	# label text
		cv2.rectangle(original_image, (int(x0), int(y0)), (int(x1), int(y1)), (255, 0, 0), 2)	# drwaing rectangle
		cv2.putText(original_image, label_text, (int(x0), int(y0)), cv2.FONT_HERSHEY_DUPLEX, 1, (231, 237, 243), 2, cv2.LINE_AA) # putting lable
	return original_image



def camera():
    video_capture = cv2.VideoCapture(0) #Put path in bracket here	
    while video_capture.isOpened():
        _, frame = video_capture.read() 
        canvas = detect_object(frame)
        cv2.imshow('Video', canvas) 
        if cv2.waitKey(1) & 0xFF == ord('x'): # To stop the loop.
            break 
        if cv2.waitKey(1) & 0xFF == ord('c'): # To stop the loop.
            MessageBox.showinfo('Capture','Frame captured.\nClick Accept to continue')
            path_output = filedialog.asksaveasfilename()
            if os.path.split(path_output)[1]!="":
                imageio.imwrite(path_output+".png", canvas[:, :, :])	# savinng back images
            else:
                MessageBox.showerror('ERROR','ERROR:Name no valid')

    video_capture.release() # We turn the webcam/video off.
    cv2.destroyAllWindows() # We destroy all the windows inside which the images were displayed.


def image():
    path_input = filedialog.askopenfilename()
    name=os.path.split(path_input)[1]
    typeImage=os.path.splitext(name)[1]
    MessageBox.showinfo('Reading','Reading ' + name+'.\nClick Accept to continue')
    original_image = imageio.imread(path_input)	# Reading image
    if original_image is not None:
        timeInit=time.time()
        output_image = detect_object(original_image)	# detecting objects
        timeFinish=time.time()
        print(timeFinish-timeInit)
        path_output = filedialog.asksaveasfilename()
        if os.path.split(path_output)[1]!="":
            imageio.imwrite(path_output+typeImage, output_image[:, :, :])	# savinng back images
        else:
            MessageBox.showerror('ERROR','ERROR:Name no valid')





def video():
    path_input = filedialog.askopenfilename()
    name=os.path.split(path_input)[1]
    typeVideo=os.path.splitext(name)[1]
    print('Reading', name)
    video_reader = imageio.get_reader(path_input)	# Reading video
    fps = video_reader.get_meta_data()['fps']	# gettinf fps of the image
    MessageBox.showinfo('Reading','Reading ' + name+'.\nClick Accept to continue')
    path_output = filedialog.asksaveasfilename()
    if os.path.split(path_output)[1]!="":
        video_writer = imageio.get_writer(path_output+typeVideo, fps = fps)	# Writing back output image
        for i, frame in enumerate(video_reader):
            output_frame = detect_object(frame)	# detecting objects frame by frame
            video_writer.append_data(output_frame)	# appending frame to vidoe
            print('frame ', i, 'done')
        video_writer.close()
    else:
        MessageBox.showerror('ERROR','ERROR:Name no valid')
    
    


root = Tk()
root.geometry('300x200') # anchura x altura
root.configure(bg = 'grey')
root.title('SSD512')
ttk.Button(root, text='Camera', command=camera).pack(side=TOP)
ttk.Button(root, text='Image', command=image).pack(side=TOP)
ttk.Button(root, text='Video', command=video).pack(side=TOP)
ttk.Button(root, text='Exit', command=quit).pack(side=BOTTOM)
root.mainloop()





