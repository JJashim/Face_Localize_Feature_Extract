from glob import glob
import sys #can be used to perform sys.exit()
import cv2
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # being tried to avoid unnecessary/warning prints of Tensorfliow
import tensorflow as tf
tf.get_logger().setLevel('INFO') # being tried to avoid unnecessary/warning prints of Tensorfliow
tf.logging.set_verbosity(tf.logging.ERROR) # being tried to avoid unnecessary/warning prints of Tensorfliow
import pandas as pd
from facenet.src import facenet
from facenet.src.align import detect_face

def create_Dir(folder_path,folder_name=''):
    if folder_name is not None:
        folder_path = os.path.join(folder_path,folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_path
    
# Method to perform cropping the images using bounding box info.
def crop_image_by_bbox(image,bbox,img_size):
    bb = np.zeros(4, dtype=np.int32)
    bb[0] = np.maximum(bbox[0] - margin / 2, 0)
    bb[1] = np.maximum(bbox[1] - margin / 2, 0)
    bb[2] = np.minimum(bbox[2] + margin / 2, img_size[1])
    bb[3] = np.minimum(bbox[3] + margin / 2, img_size[0])
    cropped = image[bb[1]:bb[3], bb[0]:bb[2], :]
    return cropped,bb
    
def load_image_align_data(dest_path,image_paths,image_size, margin, pnet, rnet, onet, discarded_folder_path = '',  bbox_thresh = 0.95):
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor
    image_list,image_names = [], []
    discared_image_cnt = 0
    img_files = glob(image_paths+'*.png') #Incase glob doesn't work in Windows environment replace it with 'os' library module.
    img_files.extend(glob(image_paths+'*.jpg')) #add more image extensions else restrict user to share images in specific format
    img_files = sorted(img_files) #chk length -- report error if its empty and exit
    print('Total images read are:', len(img_files))
    for files in img_files:
        img = cv2.imread(files,1)
        if img is not None:
            image_list.append(img)
    image_files = img_files.copy()
    recog_image_paths,img_list = [],[]
    fnames_list = []
    cropped_big_face = dest_path+'_cropped_face/' #Dir to store cropped faces
    cropped_big_face = create_Dir(cropped_big_face)
    print('path of cropped_big_face:',cropped_big_face)
        
    for img_path,x in zip(image_files,range(len(image_list))):
        fname = os.path.basename(img_path)
        dest_fname = os.path.join(dest_path,fname)
        img_size = np.asarray(image_list[x].shape)[0:2]
        img = cv2.imread(img_path,cv2.IMREAD_COLOR) #chk img.shape and log filename if its empty
        bounding_boxes, _ = detect_face.detect_face(
            image_list[x], minsize, pnet, rnet, onet, threshold, factor)
        nrof_samples = len(bounding_boxes)
        r_cnt = 0
        img_temp = image_list[x].copy()
        while(nrof_samples == 0 and r_cnt < 3):
           image_list[x] = cv2.rotate(image_list[x], cv2.ROTATE_90_CLOCKWISE)
           bounding_boxes, _ = detect_face.detect_face(
               image_list[x], minsize, pnet, rnet, onet, threshold, factor)
           nrof_samples = len(bounding_boxes)
           r_cnt += 1
        if nrof_samples > 0:
            if r_cnt == 0:
                #cv2.imwrite(os.path.join(dest_path,fname),img)
                pass
            #perform image rotation of degrees: [90,180,270] iff faces aren't recognized
            elif r_cnt == 1:
                rot_angle = cv2.ROTATE_90_CLOCKWISE
            elif r_cnt == 2:
                rot_angle = cv2.ROTATE_180
            elif r_cnt == 3:
                rot_angle = cv2.ROTATE_90_COUNTERCLOCKWISE
                
            if r_cnt > 0:
                image_list[x] = cv2.rotate(img_temp, rot_angle)
            else:
                image_list[x] = img_temp
            big_area = -1;big_face_no = -1 #param used for finding the bigger face within the image
            img_size = np.asarray(image_list[x].shape)[0:2]
            for i in range(nrof_samples):
                if bounding_boxes[i][4] > bbox_thresh:
                    img_name = fname#img_path
                    det = np.squeeze(bounding_boxes[i, 0:4])
                    cropped,bb = crop_image_by_bbox(image_list[x],det,img_size)
                    x1,y1,x2,y2 = bb
                    area_ratio = (x2-x1)*(y2-y1)/(np.prod(img_size))
                    if area_ratio > big_area:
                        big_area = area_ratio
                        big_face_no = i

                    #cv2.rectangle(image_list[x], (x1, y1), (x2, y2), (0,0,255), 3) #comment -- to remove drawing bounding box on all faces detected
                    
            #cv2.imwrite(dest_fname,image_list[x]) #comment -- to remove drawing bounding box on all faces detected
            if big_face_no < 0:
                continue
            else: #indirectly checks bounding_boxes[i][4] > 0.95
                det = np.squeeze(bounding_boxes[big_face_no, 0:4])
                print('conf. score of ',img_name,' is:',str(round(bounding_boxes[big_face_no][4],3))) #print in log: confidence score of big face detected and localized.
                cropped,bb = crop_image_by_bbox(image_list[x],det,img_size)
                cv2.imwrite(os.path.join(cropped_big_face,img_name),cropped)
                x1,y1,x2,y2 = bb
                cv2.rectangle(image_list[x], (x1, y1), (x2, y2), (0,0,255), 3)#draw bounding box only on big face
                cv2.imwrite(dest_fname,image_list[x])
                aligned = cv2.resize(
                        cropped, (image_size, image_size), cv2.INTER_LINEAR)
                prewhitened = facenet.prewhiten(aligned)
                img_list.append(prewhitened)
                fnames_list.append(img_name)
        else:
            discared_image_cnt += 1
            discard_fname = os.path.join(discarded_folder_path,fname)
            cv2.imwrite(discard_fname,img_temp)
    print('Total number of Discarded images are:',discared_image_cnt)
            
    if len(img_list) > 0:
        images = np.stack(img_list)
        print('Total number of Localized images:',len(images)) #No. of images been able to be localized
        return images, fnames_list
    else:
        #Perform exit & mention no faces recognized..check input folder
        print("load_image_align_data returned None !")
        return None

# Create model to perform localization -- MTCNN
def create_network_face_detection(gpu_memory_fraction):
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
    return pnet, rnet, onet

if __name__ == '__main__':

    FACE_FEATURE_REQUIRED = True #should be set by the user -- True/False. True/1 means Face Localization + Feature Extraction and False/0 means only Face Localization is performed
    margin = 44 #add to config -- developer
    image_size = 160 #add to config -- developer --image size used to resize faces which will be passed to Facenet for face feature extraction
    BBox_Thresh = 0.95 #add to config -- developer
    image_paths = '/Users/jowherali.syed/Projects/DL_Comp/face_comp_codes/input/Images/' #Input path
    dest_path = '/Users/jowherali.syed/Projects/DL_Comp/face_comp_codes/output/' #Output Folder
    dest_path = create_Dir(dest_path) #create output DIR
    img_dest_path = create_Dir(dest_path,'Localized_Images') #create DIR to store localized images within output/Localized_Images
    discard_folder_path = create_Dir(dest_path,'Discarded_Images') #create DIR to store discarded images

    if FACE_FEATURE_REQUIRED:
        model_path = '/Users/jowherali.syed/Projects/DL_Comp/face_comp_codes/models/20180402-114759/' #add to config --model_path: "Required for face extraction alone"
        csv_name = 'face_fingerprint.csv' #Output CSV file name
        csv_dest_path = create_Dir(dest_path,'csv_output') #Create csv folder within output folder
        csv_dest_file_path = os.path.join(csv_dest_path,csv_name)

    # To perform face localize
    pnet, rnet, onet  = create_network_face_detection(gpu_memory_fraction=1.0)
    train_images,image_paths = load_image_align_data(img_dest_path,image_paths,image_size, margin, pnet, rnet, onet,discarded_folder_path = discard_folder_path, bbox_thresh = BBox_Thresh)

    # To perform Facial feature extraction
    if FACE_FEATURE_REQUIRED:
        with tf.Graph().as_default():
            with tf.Session() as sess:
                facenet.load_model(model_path)
                images_placeholder = sess.graph.get_tensor_by_name("input:0")
                embeddings = sess.graph.get_tensor_by_name("embeddings:0")
                phase_train_placeholder = sess.graph.get_tensor_by_name("phase_train:0")
                feed_dict = {images_placeholder: train_images,
                             phase_train_placeholder: False} #currently passing entire images as input to the model..pass it in batches and keep the batch_size at config param -- default it to 32
                train_embs = sess.run(embeddings, feed_dict=feed_dict)
                train_imgs_dict = dict(zip(image_paths,train_embs))
                df_train = pd.DataFrame.from_dict(train_imgs_dict,orient='index')
                print('Face Embedded: No. of images:', len(image_paths),'within',len(train_images),'Localized Images')
                df_train.to_csv(csv_dest_file_path) #output CSV files -- {img_names,features}

#At the end of execution -- print or inform users about the Output folders
