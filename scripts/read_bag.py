import rosbag
from geometry_msgs.msg import TransformStamped
import numpy as np
import tf
from tf.transformations import *
import scipy.io as scio
import time,sys,os,traceback

rot_off = quaternion_about_axis(2.094,(1,1,-1)) # robot rotation from Vicon body frame
pos_off = [0.0165,0.07531,-0.04] # coords of the robot origin in the Vicon body frame

# Pre-processing
off_mat = quaternion_matrix(rot_off)
mis_mat = euler_matrix(-0.03, 0.04, -0.02, 'rxyz')
off_mat = np.dot(off_mat,mis_mat)
off_mat[0:3,3] = pos_off

name = sys.argv[1]
outName = name[0:19]
poseFile = open(outName + ".txt", 'w')
cmdFile = open(outName + "_ctrl.txt", 'w')

for topic, msg, t in rosbag.Bag(name).read_messages():
    if topic == '/vicon/jumper/body':
        data = msg
        
        # Extract transform from message
        rot = data.transform.rotation
        tr = data.transform.translation
        q = np.array([rot.x, rot.y, rot.z, rot.w])
        pos = np.array([tr.x, tr.y, tr.z])

        # Convert to homogeneous coordinates
        HV = quaternion_matrix(q)
        HV[0:3,3] = pos # Vicon to markers
        HR = HV.dot(off_mat) # Vicon to robot

        # Extract relevant robot coordinates
        pos = HR[0:3,3]
        euler_temp = euler_from_matrix(HR, axes='rzxy')
        euler = [euler_temp[0], euler_temp[1], euler_temp[2]]
        
        #print t, pos[0], pos[1], pos[2], euler[0], euler[1], euler[2]
        poseFile.write(str(t) + ", " + str(pos[0]) + ", " + str(pos[1]) + ", " + str(pos[2]) + ", " + str(euler[0]) + ", " + str(euler[1]) + ", " + str(euler[2]) + "\n")

    elif topic == '/control/yaw':
        cmdFile.write(str(t) + ", 1, " + str(msg.data) + "\n")
    elif topic == '/control/rol':
        cmdFile.write(str(t) + ", 2, " + str(msg.data) + "\n")
    elif topic == '/control/pit':
        cmdFile.write(str(t) + ", 3, " + str(msg.data) + "\n")
    elif topic == '/control/ret':
        cmdFile.write(str(t) + ", 4, " + str(msg.data) + "\n")
    elif topic == '/control/ext':
        cmdFile.write(str(t) + ", 5, " + str(msg.data) + "\n")

