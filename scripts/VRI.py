#!/usr/bin/env python
# Software License Agreement (BSD License)
#
# Copyright (c) 2008, Willow Garage, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of Willow Garage, Inc. nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Revision $Id$

import rospy
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Float32
from ros_ip25.srv import *

import math
import numpy as np
import tf
from tf.transformations import *
import scipy.io as sio
import time,sys,os,traceback
import serial

sys.path.append('/home/justin/Documents/Studio1458/ImageProc/roach/python')
sys.path.append('/home/justin/Documents/Studio1458/ImageProc/roach/python/lib')
sys.path.append('/home/justin/Documents/Studio1458/ImageProc/imageproc-settings')
from lib import command
import shared_multi as shared # note this is local to this machine
from velociroach import *

from hall_helpers import *

EXIT_WAIT = False

# Parameters
alpha_v = 0.5 # velocity first-order low-pass
alpha_a = 0.1 # acceleration first-order low-pass
dt = 0.01 # Vicon frame time step
rot_off = quaternion_about_axis(2.094,(1,1,-1)) # robot rotation from Vicon body frame
pos_off = [0.0165,0.07531,-0.04] # coords of the robot origin in the Vicon body frame
#[0.00587, 0.0165, -0.07531]
#[0.0165,0.07531,-0.00587]

step_list = np.array([[1.0,0.0,4.0]])

# Pre-processing
off_mat = quaternion_matrix(rot_off)
off_mat[0:3,3] = pos_off
k_file = sio.loadmat('/home/justin/Berkeley/FearingLab/Jumper/jumper/8_Bars/salto1p_v_poly_ctrler4b.mat')#salto1p_poly_ctrler1.mat')
k = k_file['a_nl'].T
k = np.delete(k, (2), axis = 0)

n_steps = len(step_list)

class VRI:
    def __init__(self):
        self.pit = 0
        self.pos = np.array([0,0,0])
        self.vel = np.array([0,0,0])
        self.acc = np.array([0,0,0])
        self.step_ind = 0
        self.last_step = time.time()
        self.yawCmd = 0.05
        self.telemetry_read = 0

        self.unheard_flag = 0
        self.xbee_sending = 1
        self.MJ_state = 0 # 0: run, 1: stand, 2: stop

        self.tf_pub = tf.TransformBroadcaster()

        self.ctrl_pub_rol = rospy.Publisher('control/rol',Float32)
        self.ctrl_pub_pit = rospy.Publisher('control/pit',Float32)
        self.ctrl_pub_yaw = rospy.Publisher('control/yaw',Float32)
        self.ctrl_pub_ret = rospy.Publisher('control/ret',Float32)
        self.ctrl_pub_ext = rospy.Publisher('control/ext',Float32)
        self.ctrl_pub_flag = rospy.Publisher('control/flag',Float32)

        setupSerial()
        queryRobot()
        
        # In-place
        self.despos = 0.5
        self.desvel = 0.0
        self.startTime = time.time()

        # Motor gains format:
        #  [ Kp , Ki , Kd , Kaw , Kff     ,  Kp , Ki , Kd , Kaw , Kff ]
        #    ----------LEFT----------        ---------_RIGHT----------
        motorgains = [100,0,40,0,0,0,0,0,0,0]
        thrustgains = [70,0,100,70,0,130]
        
        #motorgains = [0,0,0,0,0,0,0,0,0,0]
        #thrustgains = [0,0,0,0,0,0]

        duration = 4000
        rightFreq = thrustgains # thruster gains
        leftFreq = [0.16, 0.2, 0.5, .16, 0.12, 0.25] # Raibert-like gains
        #           xv xp xsat yv yp ysat
        phase =  [67, 80] # Raibert leg extension
        #       retract extend
        telemetry = True
        repeat = False

        # Gains for actual Raibert controller
        #leftFreq = [0.1, 0.015, 0.5, 0.15, 0.12, 0.1] 

        self.manParams = manueverParams(0,0,0,0,0,0)
        self.params = hallParams(motorgains, duration, rightFreq, leftFreq, phase, telemetry, repeat)
        xb_send(0, command.SET_THRUST_OPEN_LOOP, pack('6h',*thrustgains))
        setMotorGains(motorgains)

        rospy.init_node('VRI')
        rospy.Subscriber('vicon/jumper/body', TransformStamped, self.callback)

        s = rospy.Service('MJ_state_server',MJstate,self.handle_MJ_state)

        # Initiate telemetry recording; the robot will begin recording immediately when cmd is received.
        path     = '/home/justin/Data/'
        name     = 'trial'
        datetime = time.localtime()
        dt_str   = time.strftime('%Y.%m.%d_%H.%M.%S', datetime)
        root     = path + dt_str + '_' + name
        shared.dataFileName = root + '_imudata.txt'
        print "Data file:  ", shared.dataFileName
        print os.curdir

        self.numSamples = int(ceil(1000 * (self.params.duration + shared.leadinTime + shared.leadoutTime) / 1000.0))
        eraseFlashMem(self.numSamples)
        
        # START EXPERIMENT
        raw_input("Press enter to start run ...")
        startTelemetrySave(self.numSamples)
        exp = [2]
        stopSignal = [0]
        viconTest = [0,0,0,0,0,0,0,0]
        xb_send(0, command.INTEGRATED_VICON, pack('8h', *viconTest))
        xb_send(0, command.START_EXPERIMENT, pack('h', *exp))
        
        self.startTime = time.time()
        self.step_ind = 0
        self.last_step = self.startTime


        self.xbee_sending = 0
        print "Done"
        #'''

        while not rospy.is_shutdown():
            rospy.sleep(0.1)
        #rospy.spin()

    def handle_MJ_state(self, data):
        print "RECEIVED " + str(data.a)

        if data.a == 2:
            while self.xbee_sending == 1:
                rospy.sleep(0.001)
            stopSignal = [0]
            xb_send(0, command.STOP_EXPERIMENT, pack('h', *stopSignal))
            self.MJ_state = 2
        elif data.a == 3:
            if self.MJ_state != 2:
                print "ROBOT NOT STOPPED: NOT READING TELEMETRY"
                return 0
            while self.xbee_sending == 1:
                rospy.sleep(0.001)
            if self.telemetry_read == 0:
                flashReadback(self.numSamples, self.params, self.manParams)
                self.telemetry_read = 1
            self.MJ_state = 3

        return 0

    def callback(self, data):
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
        '''
        euler[2] = euler[2] - 3.14159265/2
        if euler[2] < - 3.14159265:
            euler[2] = euler[2] + 3.14159265*2
        '''
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

        # bottom-out preventing shift
        ctrl[1] = ctrl[1] - 10
        ctrl[2] = ctrl[2] - 10

        # bottom-out protection
        ctrl[1] = min(ctrl[1], -65)
        ctrl[1] = max(ctrl[1], -80)
        ctrl[2] = max(ctrl[2], -80)

        # initial override
        if time.time()-self.startTime < 0.3:
            ctrl[0] = 0
            ctrl[1] = -75
            ctrl[2] = -75

        # Raibert-inspired simple hopping
        # Forwards-backwards
        #'''
        FBvel = 1.0/3.0
        FBslowdown = 0.1 # factor to reduce commanded velocity
        startDwell = 3.0
        offset = -0.5
        endPt = 3.0
        if (time.time()-self.startTime) < startDwell:
            self.despos = offset
            self.desvel = 0.0
        elif (time.time()-self.startTime-startDwell) % (2*endPt/FBvel) > (endPt/FBvel):
            self.despos = endPt - FBvel*((time.time()-self.startTime-startDwell)%(endPt/FBvel)) + offset
            self.desvel = -FBvel*FBslowdown
        else:
            self.despos = FBvel*((time.time()-self.startTime-startDwell)%(endPt/FBvel)) + offset
            self.desvel = FBvel*FBslowdown
        #'''
        
        # Vertical variation
        '''
        if (time.time()-self.startTime) % 12.0 > 9.0:
            self.params.phase[1] = 70
        elif (time.time()-self.startTime) % 12.0 > 6.0:
            self.params.phase[1] = 80
        elif (time.time()-self.startTime) % 12.0 > 3.0:
            self.params.phase[1] = 75
        else:
            self.params.phase[1] = 80
        '''

        # Raibert controller
        KP = self.params.leftFreq[0]
        K = self.params.leftFreq[1] #Raibert velocity control gain
        Vmax = self.params.leftFreq[2]
        Ts = 0.06   # stance time in seconds
        L = 0.225   # leg length in meters
        KV = 0      # not used for position control     
        Rvdes = -KP*(self.pos[0]-self.despos) - KV*self.vel[0]
        Rvdes = max(min(Rvdes,Vmax),-Vmax)

        Rxf = self.vel[0]*Ts/2 + K*(self.vel[0] - Rvdes)
        Rxf = max(min(Rxf,L),-L)
        Rth = math.asin(Rxf/L)

        ctrl = [-Rth,self.params.phase[0], self.params.phase[1]]

        # Raibert-inspired controller
        ctrl = [-self.params.leftFreq[0]*(self.vel[0]-self.desvel) - self.params.leftFreq[1]*max(min(self.pos[0]-self.despos, self.params.leftFreq[2]),-self.params.leftFreq[2]), self.params.phase[0], self.params.phase[1]] # Simple controller

        AngleScaling = 7334;#7334; # rad to 16b 2000deg/s integrated 1000Hz
            # 180(deg)/pi(rad) * 2**16(ticks)/2000(deg/s) * 1000(Hz) = 1877468
            # 1877467 / 2**8 = 7334
        LengthScaling = 256; # radians to 23.8 fixed pt radians
        CurrentScaling = 256; # radians to 23.8 fixed pt radians
        ES = [int(euler[0]*AngleScaling),int(euler[1]*AngleScaling),int(euler[2]*AngleScaling)]

        if np.isnan(ctrl[0]): # NaN check
            ctrl[0] = 0
        if np.isnan(ctrl[1]):
            ctrl[1] = 65
        if np.isnan(ctrl[2]):
            ctrl[2] = 65

        CS = [int(ctrl[0]*AngleScaling),int(ctrl[1]*LengthScaling),int(ctrl[2]*CurrentScaling)]
        yaw = int(0);
        roll = int(AngleScaling*(self.params.leftFreq[3]*self.vel[1] + self.params.leftFreq[4]*max(min(self.pos[1],self.params.leftFreq[5]),-self.params.leftFreq[5])));
        
        if self.MJ_state == 0:
            self.xbee_sending = 1
            self.yawCmd = yaw

            '''
            rot_rol = quaternion_about_axis(roll/AngleScaling,(1,1,-1))
            mat_rol = quaternion_matrix()
            rot_pit = quaternion_about_axis()
            mat_pit = quaternion_matrix()
            '''

            toSend = [ES[0], ES[1], ES[2], self.yawCmd, roll, CS[0], CS[1], CS[2]]
            for i in range(8):
                if toSend[i] > 32767:
                    toSend[i] = 32767
                elif toSend[i] < -32767:
                    toSend[i] = -32767
            xb_send(0, command.INTEGRATED_VICON, pack('8h',*toSend))
            self.xbee_sending = 0
            #print(self.step_ind,ctrl[0],ctrl[1],ctrl[2]) # Deadbeat
            #print(self.pos[0],self.pos[1],self.pos[2])
            #print(self.yawCmd,roll*57/AngleScaling,CS[0]*57/AngleScaling)
            #print(euler[0]*57,euler[1]*57,euler[2]*57)
            #print([ES[0], ES[1], ES[2], self.yawCmd, roll, CS[0]])
            #print(self.yawCmd, roll, CS[0], CS[1], CS[2]) # Commands
            #print(roll,ctrl[0],ctrl[1],ctrl[2]) # cmds before scaling
            #print(np.hstack((ctrl.T, [self.acc])))
            #print(self.despos, self.desvel) # Raibert position
            print(57*ES[0]/AngleScaling, 57*ES[1]/AngleScaling, 57*ES[2]/AngleScaling)

            # Publish commands
            self.ctrl_pub_rol.publish(roll)
            self.ctrl_pub_pit.publish(CS[0])
            self.ctrl_pub_yaw.publish(self.yawCmd)
            self.ctrl_pub_ret.publish(CS[1])
            self.ctrl_pub_ext.publish(CS[2])


        elif self.MJ_state == 1:
            self.xbee_sending = 1
            toSend = [ES[0], ES[1], ES[2], self.yawCmd, 0, 0, 2560, 2560]
            xb_send(0, command.INTEGRATED_VICON, pack('8h',*toSend))
            self.xbee_sending = 0
        elif self.MJ_state == 2:
            rospy.sleep(0.001)
            #stopSignal = [0]
            #xb_send(0, command.STOP_EXPERIMENT, pack('h', *stopSignal))
        elif self.MJ_state == 3:
            rospy.sleep(0.001)
            #if self.telemetry_read == 0:
            #    flashReadback(self.numSamples, self.params, self.manParams)
            #    self.telemetry_read = 1

                

        t = data.header.stamp.to_sec()
        x = np.array([data.transform.rotation.x,data.transform.rotation.y,data.transform.rotation.z,data.transform.rotation.w, data.transform.translation.x,data.transform.translation.y,data.transform.translation.z,t])
        self.tf_pub.sendTransform((x[4], x[5], x[6]), (x[0], x[1], x[2], x[3]), rospy.Time.now(), "jumper", "world")

        '''
        self.step_ind += 1
        if self.step_ind > 100:
            self.step_ind = 0

        self.R1.setServo(-self.step_ind*0.005-0.4)
        '''

if __name__ == '__main__':
    try:
        VRI()
    except KeyboardInterrupt:
        print "\nRecieved Ctrl+C, exiting."
        shared.xb.halt()
        shared.ser.close()
    except Exception as args:
        print "\nGeneral exception from main:\n",args,'\n'
        print "\n    ******    TRACEBACK    ******    "
        traceback.print_exc()
        print "    *****************************    \n"
        print "Attempting to exit cleanly..."
        shared.xb.halt()
        shared.ser.close()
        sys.exit()
    except serial.serialutil.SerialException:
        shared.xb.halt()
        shared.ser.close()
