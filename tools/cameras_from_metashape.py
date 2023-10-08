import numpy as np
import xml.etree.ElementTree as ET
import copy
import os

# change it to your calibration xml file from Metashape
xml_file = 'cameras.xml'   

# the folder where the NHR "inf" files to be saved.
target_path = '.'

class Cameras:
    def __init__(self):
        self.K = np.zeros((3,3))
        self.T = np.zeros((4,4))
        self.T[3,3] = 1.0
        self.id = -1
        
    def set_resolution(self, w, h):
        w = int(w)
        h = int(h)
        self.width = w
        self.height = h
    
    def set_intrinsic(self, f, cx, cy, b1, b2):
        f, cx, cy, b1, b2 = float(f), float(cx), float(cy), float(b1), float(b2)
        cx = self.width/2  + cx
        cy = self.height/2 + cy
        fy = f
        fx = fy+b1
        self.K= np.array([ [fx, b2, cx],[0,fy,cy],[0,0,1] ])
        self.K = self.K.astype(np.float32)
        
    def set_distort(self, dic):
        
        if 'k1' in dic:
            k1 = float(dic['k1'])
        else:
            k1 = 0
        if 'k2' in dic:
            k2 = float(dic['k2'])
        else:
            k2 = 0
        if 'p1' in dic:
            p1 = float(dic['p1'])
        else:
            p1 = 0
        if 'p2' in dic:
            p2 = float(dic['p2'])
        else:
            p2 = 0
        if 'k3' in dic:
            k3 = float(dic['k3'])
        else:
            k3 = 0
        
        self.distort = np.array([k1,k2,p1,p2,k3])
        self.distort = [k1,k2,p1,p2,k3]


tree = ET.ElementTree(file=xml_file)
root = tree.getroot()
cameras_temp = [None]*900

for elem in tree.iter(tag='sensor'):
    
    cam = Cameras()
    cali = elem.find('calibration')
    
    if cali.attrib['class'] == 'initial':
        continue
    dic = {}
    for i in cali.iter():
        dic[i.tag]=i.text
        
    if 'cx' in dic:
        cx = float(dic['cx'])
    else:
        cx = 0
        
    if 'cy' in dic:
        cy = float(dic['cy'])
    else:
        cy = 0
        
    if 'b1' in dic:
        b1 = float(dic['b1'])
    else:
        b1 = 0
        
    if 'b2' in dic:
        b2 = float(dic['b2'])
    else:
        b2 = 0
        
    cam.set_resolution(cali[0].attrib['width'],cali[0].attrib['height'])
    cam.set_intrinsic(dic['f'],cx,cy,b1,b2)
    
    

    cam.set_distort(dic)
    cameras_temp[int(elem.attrib['id'])] = cam
    print(int(elem.attrib['id']))
    
cameras = []
for elem in tree.iter(tag='cameras'):
    cameras = [None]*int(elem.attrib['next_id'])
    for cam in elem.iter(tag='camera'):
        sensor_id = int(cam.attrib['sensor_id'])
        cam_id =  int(cam.attrib['id'])
        cameras[cam_id] = copy.deepcopy(cameras_temp[sensor_id])
        T = np.array([ float(i) for i in cam[0].text.split(' ')])
        T = T.reshape(4,4)
        cameras[cam_id].T = T
        cameras[cam_id].id = cam_id


with open(os.path.join(target_path,'Intrinsic.inf'), 'w') as f:
    for i,cam in enumerate(cameras):
        f.write('%d\n'%i)
        f.write('%f %f %f\n %f %f %f\n %f %f %f\n' % tuple(cam.K.reshape(9).tolist()))
        f.write('\n')
        
        
with open(os.path.join(target_path,'CamPose.inf'), 'w') as f:
    for i,cam in enumerate(cameras):
        A = cam.T[0:3,:]
        tmp = np.concatenate( [A[0:3,2].T, A[0:3,0].T,A[0:3,1].T,A[0:3,3].T])
        f.write('%f %f %f %f %f %f %f %f %f %f %f %f\n' % tuple(tmp.tolist()))
