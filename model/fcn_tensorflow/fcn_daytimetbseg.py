# Run prediction and genertae pixelwise annotation for every pixels in the image using fully coonvolutional neural net
# Output saved as label images, and label image overlay on the original image
# 1) Make sure you you have trained model in logs_dir (See Train.py for creating trained model)
# 2) Set the Image_Dir to the folder where the input image for prediction are located
# 3) Set number of classes number in NUM_CLASSES
# 4) Set Pred_Dir the folder where you want the output annotated images to be save
# 5) Run script
#--------------------------------------------------------------------------------------------------------------------
import tensorflow as tf
import numpy as np
import scipy.misc as misc
import sys
from . import BuildNetVgg16
from . import TensorflowUtils
import os
from . import Data_Reader
from . import OverrlayLabelOnImage as Overlay
from . import CheckVGG16Model
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
logs_dir= "zoo/model_zoo/fcn_tensorflow/logs/daytimetbseg"# "path to logs directory where trained model and information will be stored"
Image_Dir="zoo/data_zoo/fcn_daytimetbseg"# Test image folder
w=0.6# weight of overlay on image
Pred_Dir="output/Output_Prediction/fcn_daytimetbseg/" # Library where the output prediction will be written
model_path="zoo/model_zoo/fcn_tensorflow/vgg16.npy"# "Path to pretrained vgg16 model for encoder"
NameEnd="" # Add this string to the ending of the file name optional
NUM_CLASSES = 4 # Number of classes
#-------------------------------------------------------------------------------------------------------------------------
CheckVGG16Model.CheckVGG16(model_path)# Check if pretrained vgg16 model avialable and if not try to download it

################################################################################################################################################################################
def Inference(image_nparray):
      # .........................Placeholders for input image and labels........................................................................
    keep_prob = tf.placeholder(tf.float32, name="keep_probabilty")  # Dropout probability
    image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name="input_image")  # Input image batch first dimension image number second dimension width third dimension height 4 dimension RGB
    

    sess = tf.Session() #Start Tensorflow session
# -------------------------Build Net----------------------------------------------------------------------------------------------
    with tf.variable_scope("fcn_daytimetbseg"):
        NewNet = BuildNetVgg16.BUILD_NET_VGG16(vgg16_npy_path=model_path)  # Create class instance for the net
        NewNet.build(image, NUM_CLASSES, keep_prob)  # Build net and load intial weights (weights before training)
        # -------------------------Data reader for validation/testing images-----------------------------------------------------------------------------------------------------------------------------
        #-------------------------Load Trained model if you dont have trained model see: Train.py-----------------------------------------------------------------------------------------------------------------------------

        
        print("Setting up Saver...")
        saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='fcn_daytimetbseg'))

        sess.run(tf.global_variables_initializer())
        
        ckpt = tf.train.get_checkpoint_state(logs_dir)
        if ckpt and ckpt.model_checkpoint_path: # if train model exist restore it
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Model restored...")
        else:
            print("ERROR NO TRAINED MODEL IN: "+ckpt.model_checkpoint_path+" See Train.py for creating train network ")
            sys.exit()

    #--------------------Create output directories for predicted label, one folder for each granulairy of label prediciton---------------------------------------------------------------------------------------------------------------------------------------------

        if not os.path.exists(Pred_Dir): os.makedirs(Pred_Dir)
        if not os.path.exists(Pred_Dir+"/OverLay"): os.makedirs(Pred_Dir+"/OverLay")
        if not os.path.exists(Pred_Dir + "/Label"): os.makedirs(Pred_Dir + "/Label")

        
        print("Running Predictions:")
        print("Saving output to:" + Pred_Dir)
     #----------------------Go over all images and predict semantic segmentation in various of classes-------------------------------------------------------------
        fim = 0
            # ..................................Load image.......................................................................................
        #FileName=ValidReader.OrderedFiles[ValidReader.itr] #Get input image name
        #Images = ValidReader.ReadNextBatchClean()  # load testing image
        Sy = 380
        Sx = 250
        batch_size=1
        Images = np.zeros([batch_size,Sy,Sx,3], dtype=np.float)
        #image size is set as [Sy,Sx] ->  [380, 250] by network if you want to change you need to retrain network
        Images[0] = misc.imresize(image_nparray, [Sy,Sx], interp='bilinear')
        # Predict annotation using net
        LabelPred = sess.run(NewNet.Pred, feed_dict={image: Images, keep_prob: 1.0})
                 #------------------------Save predicted labels overlay on images---------------------------------------------------------------------------------------------
        misc.imsave(Pred_Dir + "/OverLay/"+ "daytimetbsegtest.jpg"  , Overlay.OverLayLabelOnImage(Images[0],LabelPred[0], w)) #Overlay label on image
        misc.imsave(Pred_Dir + "/Label/" + "daytimetbsegtest.png" + NameEnd, LabelPred[0].astype(np.uint8))
    
    
    return LabelPred[0].astype(np.uint8)
    ##################################################################################################################################################
print("Finished")
