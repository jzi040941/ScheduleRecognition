# show images inline
#%matplotlib inline

# automatically reload modules when they have changed
#%load_ext autoreload
#%autoreload 2

# import keras
import keras

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

# use this environment flag to change which GPU to use
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())

# adjust this to point to your downloaded/trained model
# models can be downloaded here: https://github.com/fizyr/keras-retinanet/releases
model_path = os.path.join('..', 'DaytimetbSegmentation', 'snapshots','Inference', 'resnet50_pascal_03no_nms_no_csf.h5')

# load retinanet model
model = models.load_model(model_path, backbone_name='resnet50')

# if the model is not converted to an inference model, use the line below
# see: https://github.com/fizyr/keras-retinanet#converting-a-training-model-to-inference-model
#model = models.load_model(model_path, backbone_name='resnet50', convert=True)

#print(model.summary())

# load label to names mapping for visualization purposes
labels_to_names = {
        0: 'table',
        1: 'scheduled',
        2: 'day',
        3: 'time'
}

write_path = os.path.join('..','DaytimetbSegmentation','predict','pascal_03nonms_nocsf')
image_path = os.path.join('..','Data_Zoo','Daytimetbseg_Pascal','Images')
	
	#image = read_image_bgr('000000008021.jpg')

if not os.path.exists(write_path):
	os.makedirs(write_path)
dirs = os.listdir(image_path)
for imgname in dirs:
	# copy to draw on
	image = read_image_bgr(os.path.join(image_path,imgname))
	draw = image.copy()
	draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

	# preprocess image for network
	image = preprocess_image(image)
	image, scale = resize_image(image)

	# process image
	start = time.time()
	boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
	print("processing time: ", time.time() - start)

	# correct for image scale
	boxes /= scale

	# visualize detections
	for box, score, label in zip(boxes[0], scores[0], labels[0]):
		# scores are sorted so we can break
		if score < 0.5:
			break

		color = label_color(label)

		b = box.astype(int)
		draw_box(draw, b, color=color)

		caption = "{} {:.3f}".format(labels_to_names[label], score)
		draw_caption(draw, b, caption)


	cv2.imwrite(os.path.join(write_path,imgname), draw, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])

'''
plt.figure(figsize=(15, 15))
plt.axis('off')
plt.imshow(draw)
plt.show()
'''
