import os
import cv2
from datetime import datetime

class IO:

    def __init__(self,dir_name):
        self.dir_name=dir_name
        
        if not (os.path.isdir(dir_name)):
            os.mkdir(dir_name)


    @staticmethod
    def get_img_filename():
        ''' Returns filename using timestamp. '''
 
        time_stamp = str(datetime.now())
        file_name='img-{}.jpg'.format(time_stamp)

        return file_name


    def save_image(self,images):
        ''' Saves face-crop images using detected bounding boxes. '''

        for img in images:
            file_name = self.get_img_filename()
            cv2.imwrite(os.path.join(self.dir_name, file_name), img) 


    def relay_face_crop_to_server(self,img):
        # TODO:send the detected face crops to server for classification
        pass
