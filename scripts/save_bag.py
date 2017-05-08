import rospy
from geometry_msgs.msg import TransformStamped

import numpy as np
import tf
from tf.transformations import *
import scipy.io as sio
import time,sys,os,traceback

rot_off = quaternion_about_axis(2.094,(1,1,-1)) # robot rotation from Vicon body frame
pos_off = [0.0165,0.07531,-0.04] # coords of the robot origin in the Vicon body frame

# Pre-processing
off_mat = quaternion_matrix(rot_off)
rot_mis = quaternion_about_axis(0.05,(0,0,1)) # Miscalibration offsets
mis_mat = quaternion_matrix(rot_mis)
off_mat = np.dot(off_mat,mis_mat)
off_mat[0:3,3] = pos_off


class VRI:
    def __init__(self):

        self.unheard_flag = 0
        self.start_time = 0

        rospy.init_node('VRI')
        rospy.Subscriber('vicon/jumper/body', TransformStamped, self.callback)

        rospy.init_node('VRI')
        rospy.Subscriber('vicon/jumper/body', TransformStamped, self.callback)
        while not rospy.is_shutdown():
            rospy.sleep(0.1)

    def callback(self, data):
        if self.unheard_flag == 0:
            self.unheard_flag = 1
            self.start_time = time.time()
    
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
        
        print time.time()-self.start_time, pos[0], pos[1], pos[2], euler[0], euler[1], euler[2]

        '''
        # CONTROLLERS
        pit = euler[2]

        self.pit = pit
        if self.unheard_flag == 0: # first message
            self.unheard_flag = 1
            self.pos = pos
            #rospy.loginfo(rospy.get_caller_id() + ' FIRST CONTACT ')
        else: # subsequent messages after the first
            vel = (pos - self.pos)/dt
            acc = (vel - self.vel)/dt
            self.pos = pos
            self.vel = alpha_v*vel + (1-alpha_v)*self.vel
            self.acc = alpha_a*acc + (1-alpha_a)*self.acc
            #rospy.loginfo(rospy.get_caller_id() + ' ' + np.array_str(vel))
        
        #print(np.hstack((self.vel, euler[0], euler[1], euler[2])))

        # Sequence steps
        #   simple test: acceleration magnitude threshold for ground contact
        if (self.acc[2] > 2*9.81 or abs(self.acc[0]) > 5 or abs(self.acc[1]) > 5): 
            if (time.time() - self.last_step) > 0.3:
                self.step_ind += 1
            self.last_step = time.time()

        # Calculate desired takeoff velocity
        #   foot planning
        indNext = min(self.step_ind, n_steps-1)
        indAim = min(self.step_ind+1, n_steps-1)
        ptNext = step_list[indNext,:]
        ptAim = step_list[indAim,:]
        GLnext = ptNext[1]
        vzNext = ptNext[2]
        
        g = 9.81
        l = 0.2

        pz = self.pos[2] - l - GLnext
        px = self.pos[0]
        vz = self.vel[2]
        vx = self.vel[0]
        
        tTD = (vz + (vz**2 + 2*g*pz)**0.5)/g
        xTD = px + vx*tTD
        launch = [xTD,GLnext]
        Dstep = ptAim[0:2] - launch

        tP = (vz + (vzNext**2 - 2*g*(Dstep[1]-l)))/g
        vxNext = Dstep[0]/(tP + l/vzNext)

        vTD = [vx, vz - g*tTD]

        #   simple test: stay near 0
        vi = vTD
        #vo = [-0.3*self.pos[0],vzNext]
        vo = [vxNext, vzNext]

        # Calculate touchdown angle, leg length, and leg current
        x_vect = np.vstack((1, np.matrix(vi).T, np.matrix(vo).T, 
            vi[0]*vi[0], vo[0]*vo[0], vi[0]*vo[0],
            vi[1]*vi[1], vo[1]*vo[1], vi[1]*vo[1],
            vi[0]*vi[1], vi[0]*vo[1], vi[1]*vo[0], vo[0]*vo[1]))

        ctrl = k.dot(x_vect)

        # initial override
        if time.time()-self.startTime < 0.3:
            ctrl[0] = 0
            ctrl[1] = -50
            ctrl[2] = -70

        # Raibert-inspired simple hopping
        # Forwards-backwards
        #FBvel = 1.0/3.0
        #startDwell = 3.0
        #if (time.time()-self.startTime) < startDwell:
            #self.despos = 0.0
            #self.desvel = 0.0
        #elif (time.time()-self.startTime-startDwell) % (4.0/FBvel) > (2.0/FBvel):
            #self.despos = 2.0 - FBvel*((time.time()-self.startTime-startDwell)%(2.0/FBvel))
            #self.desvel = -FBvel
        #else:
            #self.despos = FBvel*((time.time()-self.startTime-startDwell)%(2.0/FBvel))
            #self.desvel = FBvel

        #ctrl = [-self.params.leftFreq[0]*(self.vel[0]-self.desvel) - self.params.leftFreq[1]*max(min(self.pos[0]-self.despos, self.params.leftFreq[2]),-self.params.leftFreq[2]), self.params.phase[0], self.params.phase[1]] # Simple controller

        AngleScaling = 7334;#7334; # radians to 16b 2000deg/s integrated 1000Hz
            # 180(deg)/pi(rad) * 2**16(ticks)/2000(deg/s) * 1000(Hz) = 1877468
            # 1877467 / 2**8 = 7334
        LengthScaling = 256; # radians to 23.8 fixed pt radians
        CurrentScaling = 256; # radians to 23.8 fixed pt radians
        ES = [int(euler[0]*AngleScaling),int(euler[1]*AngleScaling),int(euler[2]*AngleScaling)]

        if np.isnan(ctrl[0]):
            ctrl[0] = 0
        if np.isnan(ctrl[1]):
            ctrl[1] = -50
        if np.isnan(ctrl[2]):
            ctrl[2] = -50

        CS = [int(-ctrl[0]*AngleScaling),int(-ctrl[1]*LengthScaling),int(-ctrl[2]*CurrentScaling)]
        yaw = int(0);
        roll = int(AngleScaling*(self.params.leftFreq[3]*self.vel[1] + self.params.leftFreq[4]*max(min(self.pos[1],self.params.leftFreq[5]),-self.params.leftFreq[5])));
        '''


if __name__ == '__main__':
    VRI()
