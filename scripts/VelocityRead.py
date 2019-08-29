#!/usr/bin/env python

import rospy
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import Pose
from std_msgs.msg import Float32
from ros_ip25.srv import *

import math
import numpy as np
import tf
from tf.transformations import *
import scipy.io as sio
import time,sys,os,traceback
import serial
import pygame

sys.path.append('/home/justin/Documents/Studio1458/ImageProc/roach/python')
sys.path.append('/home/justin/Documents/Studio1458/ImageProc/roach/python/lib')
sys.path.append('/home/justin/Documents/Studio1458/ImageProc/imageproc-settings')
from lib import command
import shared_multi as shared # note this is local to this machine
from velociroach import *

from hall_helpers import *
import salto_optitrack_config

EXIT_WAIT = False

# Salto:
salto_name = 3 # 1: Salto-1P Santa, 2: Salto-1P Rudolph, 3: Salto-1P Dasher

# Parameters
alpha_v = 0.8 # velocity first-order low-pass
alpha_a = 0.6 # acceleration first-order low-pass
dt = 0.01#(1.0/120.0)# 0.01 # Optitrack frame time step
rot_off = quaternion_about_axis(0,(1,1,1)) # robot rotation from Vicon body frame
pos_off = [0.0,0.0,0.0] # coords of the robot origin in the Vicon body frame
#[0.00587, 0.0165, -0.07531]
#[0.0165,0.07531,-0.00587]
yaw_rate = 0.8 # yaw rate in rad/s

decimate_factor = 1

# Pre-processing
off_mat = quaternion_matrix(rot_off)
if salto_name == 1:
    mis_mat = salto_optitrack_config.offsets1
    retractOffset = 0
elif salto_name == 2:
    mis_mat = salto_optitrack_config.offsets2
    retractOffset = 2
elif salto_name == 3:
    mis_mat = salto_optitrack_config.offsets3
    retractOffset = 0

off_mat = np.dot(off_mat,mis_mat)
off_mat[0:3,3] = pos_off

#k_file = sio.loadmat('/home/justin/Berkeley/FearingLab/Jumper/Dynamics/3D/Hybrid3D/runGridMotor4.mat')
#k_file = sio.loadmat('/home/justin/Berkeley/FearingLab/Jumper/robotdata/physicalDeadbeatCurveFit1.mat')
#k = k_file['a_nl'].T


# Scaling constants
AngleScaling = 3667; # rad to 15b 2000deg/s integrated 1000Hz
    # 180(deg)/pi(rad) * 2**15(ticks)/2000(deg/s) * 1000(Hz) = 938734
    # 938734 / 2**8 = 3667
LengthScaling = 256; # radians to 23.8 fixed pt radians
CurrentScaling = 256; # radians to 23.8 fixed pt radians

class VelocityRead:
    def __init__(self):
        # Robot position variables
        self.pos = np.matrix([[0],[0],[0]])
        self.vel = np.matrix([[0],[0],[0]])
        self.acc = np.matrix([[0],[0],[0]])
        self.euler = np.array([0,0,0])
        self.step_ind = 0
        self.last_step = time.time()


        # Position Kalman filter
        self.F = np.matrix([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]])
        self.Q = np.matrix([
            [0.3*dt, 0, 0, 0, 0, 0],
            [0, 0.3*dt, 0, 0, 0, 0],
            [0, 0, 0.3*dt, 0, 0, 0],
            [0, 0, 0, 5*dt, 0, 0],
            [0, 0, 0, 0, 5*dt, 0],
            [0, 0, 0, 0, 0, 5*dt]])
        self.H = np.matrix([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]])
        self.R = np.matrix([
            [0.002, 0, 0],
            [0, 0.002, 0],
            [0, 0, 0.002]])
        self.P_init = np.matrix([
            [0.01, 0, 0, 0, 0, 0],
            [0, 0.01, 0, 0, 0, 0],
            [0, 0, 0.01, 0, 0, 0],
            [0, 0, 0, 10, 0, 0],
            [0, 0, 0, 0, 10, 0],
            [0, 0, 0, 0, 0, 10]])
        self.x = np.matrix([0,0,0,0,0,0]).T
        self.P = self.P_init



        # # Balance control tilt once to 9/4*a*T^2 rad and 1/2*a*T rad/s
        a = 30#-25.0# angular acceleration (rad/s^2)
        T = 0.07#0.05# # time scale (s)
        motorExtend = 76 # radians
        t_motor = 0.14 #0.16 # 0.17 # seconds
        t_spin = -0.01 #disable 0.08 # seconds

        k1 = 0
        k2 = -15
        airRetract = (motorExtend-80.0)*0.5 + 60.0
        
        toHop = 1 # make a small jump (1) or not (0)

        rollOff = -0.01#-0.02


        # BEGIN -----------------------
        setupSerial()
        queryRobot()

        rospy.init_node('ORI')
        rospy.Subscriber('Robot_1/pose', Pose, self.callback)

        self.decimate_count = 0
        self.telemetry_read = 0
        self.unheard_flag = 0
        self.xbee_sending = 1
        self.MJ_state = 0 # 0: run, 1: stand, 2: stop
        self.ctrl_mode = 1 # 0: Old Raibert, 1: deadbeat curve fit, 2: New Raibert velocity
        self.onboard_control = False # use onboard calculated control and feed velocity commands
        self.use_joystick = False
        self.started = 0 # 0: not started, 1: started
        self.extra_flag = 0
        s = rospy.Service('MJ_state_server',MJstate,self.handle_MJ_state)

        duration = 3000
        telemetry = True
        repeat = False

        zeroGains = [50,30,0, 80,70,0, 0,0,0,0]

        runTailGains = [50,30,0, 80,40,0, 100,13,0,0]

        standTailGains = [50,30,0, 80,50,0, 100,13,0,0]

        motorgains = [50,30,0, 80,40,0, 100,13,0,0]
        leftFreq = [0.16, 0.2, 0.5, .16, 0.12, 0.25]
        phase = [65, 80] # Raibert leg extension
        rightFreq = runTailGains # thruster gains

        self.manParams = manueverParams(0,0,0,0,0,0)
        self.params = hallParams(motorgains, duration, rightFreq, leftFreq, phase, telemetry, repeat)

        if self.params.telemetry:
            # Construct filename
            # path     = '/home/duncan/Data/'
            path     = 'Data/'
            name     = 'trial'
            datetime = time.localtime()
            dt_str   = time.strftime('%Y.%m.%d_%H.%M.%S', datetime)
            root     = path + dt_str + '_' + name
            shared.dataFileName = root + '_imudata.txt'
            print "Data file:  ", shared.dataFileName
            print os.curdir

            self.numSamples = int(ceil(1000 * (self.params.duration + shared.leadinTime + shared.leadoutTime) / 1000.0))
            eraseFlashMem(self.numSamples)
            raw_input("Press enter to start run ...") 
            startTelemetrySave(self.numSamples)

        exp = [2]
        arbitrary = [0]

        modeSignal = [0]
        xb_send(0, command.ONBOARD_MODE, pack('h', *modeSignal))
        time.sleep(0.02)

        #zeroGains = [0,0,0,0,0, 0,0,0,0,0]
        xb_send(0, command.SET_PID_GAINS, pack('10h',*zeroGains))
        time.sleep(0.02)

        viconTest = [0,0,0, 0,3667*rollOff,0, 0*256,0*256]#55*256,70*256]
        xb_send(0, command.INTEGRATED_VICON, pack('8h', *viconTest))
        time.sleep(0.02)

        xb_send(0, command.RESET_BODY_ANG, pack('h', *arbitrary))
        time.sleep(0.02)

        xb_send(0, command.GYRO_BIAS, pack('h', *arbitrary))
        time.sleep(0.02)

        xb_send(0, command.G_VECT_ATT, pack('h', *arbitrary))
        time.sleep(0.02)

        adjust = [0,-64,-128]
        xb_send(0, command.ADJUST_BODY_ANG, pack('3h', *adjust))
        time.sleep(0.02)

        modeSignal = [17]
        xb_send(0, command.ONBOARD_MODE, pack('h', *modeSignal))
        time.sleep(0.02)

        xb_send(0, command.START_EXPERIMENT, pack('h', *exp))
        time.sleep(1.5)
        
        xb_send(0, command.SET_PID_GAINS, pack('10h',*standTailGains))
        time.sleep(0.98)

        viconTest = [0,0,0, 0,3667*rollOff,0, 30*256,30*256]#55*256,70*256]
        xb_send(0, command.INTEGRATED_VICON, pack('8h', *viconTest))
        time.sleep(0.02)

        time.sleep(0.98)

        modeSignal = [16]
        xb_send(0, command.ONBOARD_MODE, pack('h', *modeSignal))
        time.sleep(0.02)

        t0 = time.time()
        t = 0.0
        tEnd = 13.0*T
        while t < tEnd:
          # Md is in 2^15/(2000*pi/180)~=938.7 ticks/rad
          t = time.time() - t0

          if t < 0.0:
            Mddd = 0.0
            Mdd = 0.0
            Md = 0.0
            M = 0.0
          elif t < T:
            tr = t - 0.0
            Mddd = a*(tr/(2*T) - 1)
            Mdd = -(a*tr*(4*T - tr))/(4*T)
            Md = -(a*tr**2*(6*T - tr))/(12*T)
            M = -(a*tr**3*(8*T - tr))/(48*T)
          elif t < 3*T:
            tr = t - T
            Mddd = -a*(tr/(2*T) - 1)
            Mdd = (a*tr*(4*T - tr))/(4*T) - (3*T*a)/4
            Md = - (5*T**2*a)/12 - (a*tr*(3*T - tr)**2)/(12*T)
            M = - (7*T**3*a)/48 - (a*tr*(9*T*tr + 10*T**2 - 4*tr**2))/24 - (a*tr**4)/(48*T)
          elif t < (11*T)/2:
            tr = t - 3*T
            Mddd = 0
            Mdd = (T*a)/4
            Md = (T*a*tr)/4 - (7*T**2*a)/12
            M = - (71*T**3*a)/48 - (T*a*tr*(14*T - 3*tr))/24
          elif t < (15*T)/2:
            tr = t - (11*T)/2
            Mddd = (a*tr)/(4*T)
            Mdd = (T*a)/4 + (a*tr**2)/(8*T)
            Md = (T**2*a)/24 + (a*tr*(6*T**2 + tr**2))/(24*T)
            M = (a*tr**4)/(96*T) - (69*T**3*a)/32 + (T*a*tr*(T + 3*tr))/24
          elif t < (17*T)/2:
            tr = t - (15*T)/2
            Mddd = (3*a*(tr/T - 1))/2
            Mdd = (3*T*a)/4 - (3*a*tr*(2*T - tr))/(4*T)
            Md = (7*T**2*a)/8 + (a*tr**3)/(4*T) + (3*a*tr*(T - tr))/4
            M = (a*tr*(3*T*tr + 7*T**2 - 2*tr**2))/8 - (45*T**3*a)/32 + (a*tr**4)/(16*T)
          elif t < (317/36+1)*T:
            tr = t - (17*T)/2
            Mddd = 0
            Mdd = 0
            Md = (9*T**2*a)/8
            M = (9*T**2*a*tr)/8 - (11*T**3*a)/32
          else:
            Mddd = 0.0
            Mdd = 0.0
            Md = 0.0
            M = 0.0

          t_launchStart = (317*T/36 - t_motor)

          # if t < 0.0: # balance
          #   Mddd = 0.0
          #   Mdd = 0.0
          #   Md = 0.0
          #   M = 0.0
          # elif t < T: # begin lean back
          #   Mddd = -a
          #   Mdd = -a*t
          #   Md = -1.0/2.0*a*t**2.0
          #   M = -1.0/6.0*a*t**3.0
          # elif t < 5.0*T: # reverse lean toward forward
          #   tr = t - T
          #   Mddd = 1.0/2.0*a
          #   Mdd = -a*T + 1.0/2.0*a*tr
          #   Md = -1.0/2.0*a*T**2.0 - a*T*tr + 1.0/4.0*a*tr**2.0
          #   M = -1.0/6.0*a*T**3.0 - 1.0/2.0*a*T**2.0*tr - 1.0/2.0*a*T*tr**2.0 + 1.0/12.0*a*tr**3.0
          # elif t < 6.0*T: # follow through forward tilt
          #   tr = t - 5.0*T
          #   Mddd = 0.0
          #   Mdd = a*T
          #   Md = -1.0/2.0*a*T**2.0 + a*T*tr
          #   M = -29.0/6.0*a*T**3.0 - 1.0/2.0*a*T**2.0*tr + 1.0/2.0*a*T*tr**2.0
          # elif t < 7.0*T: # slow forward tilt
          #   tr = t - 6.0*T
          #   Mddd = -1.0/2.0*a
          #   Mdd = a*T - 1.0/2.0*a*tr
          #   Md = 1.0/2.0*a*T**2.0 + a*T*tr - 1.0/4.0*a*tr**2.0
          #   M = -29.0/6.0*a*T**3.0 + 1.0/2.0*a*T**2.0*tr + 1.0/2.0*a*T*tr**2.0 - 1.0/12.0*a*tr**3.0;
          # elif t < (7.0+2.1815+2.0)*T: # hold forward tilt
          #   tr = t - 7.0*T
          #   Mddd = 0.0
          #   Mdd = 1.0/2.0*a*T
          #   Md = 5.0/4.0*a*T**2.0 + 1.0/2.0*a*T*tr
          #   M = -24/6*a*T**3.0 + 5.0/4.0*a*T**2.0*tr + 1.0/4.0*T*tr**2.0
          # else:
          #   Mddd = 0.0
          #   Mdd = 0.0
          #   Md = 0.0
          #   M = 0.0

          # t_launchStart = 9.1815*T - t_motor

          if t > t_launchStart and t < (t_launchStart + t_spin):
            Mdd = Mdd - 0.5

          # Send tilt command
          tiltCmd = [M*938.7, Md*938.7, Mdd*938.7, Mddd*938.7/2.0]
          for ind in range(4):
            if tiltCmd[ind] > 32767:
              tiltCmd[ind] = 32767
              print 'TOO LARGE'
            elif tiltCmd[ind] < -32768:
              tiltCmd[ind] = -32768
              print 'TOO SMALL'
          xb_send(0, command.TILT, pack('4h', *tiltCmd))
          print tiltCmd
          time.sleep(0.01)

          if t > t_launchStart and toHop == 1: # begin launch
            # Normal
            viconTest = [0,0,0, 0,3667*rollOff,0, motorExtend*256,motorExtend*256]
            xb_send(0, command.INTEGRATED_VICON, pack('8h', *viconTest))
            time.sleep(0.01)
            xb_send(0, command.SET_PID_GAINS, pack('10h',*runTailGains))
            time.sleep(0.02)
            toHop = 2

            # # Higher gains
            # modeSignal = [1]#[7]
            # xb_send(0, command.ONBOARD_MODE, pack('h', *modeSignal))
            # time.sleep(0.01)
              
          if t > 11.0*T and toHop == 2: # prepare for landing
            # # Make a few bounces, then stop
            # modeSignal = [6]
            # xb_send(0, command.ONBOARD_MODE, pack('h', *modeSignal))
            # time.sleep(0.01)
            # toSend = [-2000, 0, 6000, 0]
            # xb_send(0, command.SET_VELOCITY, pack('4h',*toSend))
            # time.sleep(0.01)

            # # Set angle bounce
            # viconTest = [0,0,0, 0,0,3667*-3.0*3.14159/180, 60*256,90*256]#55*256,70*256]
            # xb_send(0, command.INTEGRATED_VICON, pack('8h', *viconTest))
            # time.sleep(0.01)
            # modeSignal = [0]
            # xb_send(0, command.ONBOARD_MODE, pack('h', *modeSignal))
            # time.sleep(0.01)

            # Hop once and stop
            viconTest = [0,0,0, 0,0,0, airRetract*256,25*256]
            xb_send(0, command.INTEGRATED_VICON, pack('8h', *viconTest))
            time.sleep(0.01)

        # time.sleep(0.2)
        # modeSignal = [0]
        # xb_send(0, command.ONBOARD_MODE, pack('h', *modeSignal))
        # time.sleep(0.01)
        # viconTest = [0,0,0, 0,0,-1.5*3667, 40*256,20*256]
        # xb_send(0, command.INTEGRATED_VICON, pack('8h', *viconTest))
        # time.sleep(0.01)

        # tiltCmd = [0, 0, 0, 0]
        # xb_send(0, command.TILT, pack('4h', *tiltCmd))
        # time.sleep(0.02)
        
        if toHop:
          # # Make a few bounces, then stop
          # modeSignal = [23]
          # xb_send(0, command.ONBOARD_MODE, pack('h', *modeSignal))
          # time.sleep(0.2)
          # viconTest = [0,0,0, 0,0,0, 55*256,15*256]
          # xb_send(0, command.INTEGRATED_VICON, pack('8h', *viconTest))
          # time.sleep(0.01)

          # # Enable if using higher gains
          # time.sleep(0.1)
          # modeSignal = [23]
          # xb_send(0, command.ONBOARD_MODE, pack('h', *modeSignal))
          # time.sleep(0.01)

          # # New leg control
          time.sleep(0.2)
          cmd = [0,0,0,0,\
          (0.12)*2**16, 0.0*2000, (0.0)*1024,\
          k1, k2]
          xb_send(0, command.STANCE, pack('9h', *cmd))
          time.sleep(0.01)
          xb_send(0, command.SET_PID_GAINS, pack('10h',*standTailGains))
          time.sleep(0.02)
          
          # # Sit down
          # time.sleep(1.5)
          # viconTest = [0,0,0, 0,0,3667*-1*3.14159/180, 20*256,20*256]
          # xb_send(0, command.INTEGRATED_VICON, pack('8h', *viconTest))
          # time.sleep(1.0)
          # viconTest = [0,0,0, 0,0,3667*-0*3.14159/180, 15*256,15*256]
          # xb_send(0, command.INTEGRATED_VICON, pack('8h', *viconTest))
          # time.sleep(1.0)
          # viconTest = [0,0,0, 0,0,3667*-0*3.14159/180, 0*256,0*256]
          # xb_send(0, command.INTEGRATED_VICON, pack('8h', *viconTest))
          # time.sleep(2.0)

        time.sleep(4.0)
        stopSignal = [0]
        xb_send(0, command.STOP_EXPERIMENT, pack('h', *stopSignal))
        time.sleep(0.01)
        xb_send(0, command.STOP_EXPERIMENT, pack('h', *stopSignal))
        time.sleep(0.02)
        xb_send(0, command.STOP_EXPERIMENT, pack('h', *stopSignal))

        self.xbee_sending = 0
        print "Done"
        self.MJ_state = 2

        while not rospy.is_shutdown():
            rospy.sleep(0.1)
            #rospy.spin()

    def handle_MJ_state(self, data):
        # Handle ROS service
        print "RECEIVED " + str(data.a)

        if data.a == 1:
            self.onboard_control = False
            self.use_joystick = False
            if self.MJ_state == 0:
                self.MJ_state = 1
        if data.a == 2:
            self.onboard_control = False
            self.use_joystick = False
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
        elif data.a == 6: # use normal vicon attitude updates
            newMode = [0]
            self.onboard_control = False
            self.use_joystick = False
            xb_send(0, command.ONBOARD_MODE, pack('h', *newMode))
        elif data.a == 7: # use ONLY onboard gyro integration
            newMode = [1]
            xb_send(0, command.ONBOARD_MODE, pack('h', *newMode))
        elif data.a == 8: # use onboard Raibert controller
            newMode = [4]
            self.onboard_control = True
            self.use_joystick = False
            self.ctrl_mode = 2 # use Raibert velocity when we return
            xb_send(0, command.ONBOARD_MODE, pack('h', *newMode))
        elif data.a == 9: # use joystick with onboard velocity control
            self.use_joystick = True
            self.ctrl_mode = 2 # use Raibert velocity when we return
            newMode = [4]
            self.onboard_control = True
            xb_send(0, command.ONBOARD_MODE, pack('h', *newMode))
        elif data.a == 10: # use joystick with mocap attitude control
            self.use_joystick = True
            self.ctrl_mode = 2 # use Raibert velocity
            newMode = [0]
            self.onboard_control = False
            xb_send(0, command.ONBOARD_MODE, pack('h', *newMode))


        if data.a == 20:
            self.ctrl_mode = 0
        elif data.a == 21:
            self.ctrl_mode = 1

        return 0

    def callback(self, data):

        # VICON DATA ------------------------------------------------
        # Extract transform from message
        rot = data.orientation
        tr = data.position
        q = np.array([rot.x, rot.y, rot.z, rot.w])
        pos = np.array([tr.x, tr.y, tr.z])

        # Convert to homogeneous coordinates
        HV = quaternion_matrix(q)
        HV[0:3,3] = pos # Vicon to markers
        HR = HV.dot(off_mat) # Vicon to robot

        # Extract relevant robot coordinates
        pos = np.mat(HR[0:3,3]).T
        euler_temp = euler_from_matrix(HR, axes='rzxy')
            # HR = Rz * Rx * Ry
        euler = [euler_temp[0], euler_temp[1], euler_temp[2]]
        self.euler = euler

        vel = (pos - self.pos)/dt
        acc = (vel - self.vel)/dt
        if (time.time() - self.last_step) > 0.1 and self.x[2] > 0.2:
          # Kalman predict step
          self.x = self.F.dot(self.x) + np.matrix([0,0,0,0,0,-9.81*dt]).T
          self.P = self.F.dot(self.P.dot(self.F.T)) + self.Q
          # Kalman update step
          innovation = pos - self.H.dot(self.x)
          KalmanGain = self.P.dot(self.H.T).dot(np.linalg.inv(self.H.dot(self.P.dot(self.H.T))) + self.R)
          self.x = self.x + KalmanGain*innovation
          self.P = self.P - KalmanGain.dot(self.H.dot(self.P))
          self.pos = self.x[0:3]
          self.vel = self.x[3:6]
        else:
          self.pos = pos
          self.vel = alpha_v*vel + (1-alpha_v)*self.vel
          self.x = np.vstack((pos, self.vel))
          self.P = self.P_init
          self.acc = alpha_a*acc + (1-alpha_a)*self.acc
            #rospy.loginfo(rospy.get_caller_id() + ' ' + np.array_str(vel))
        
        #print(np.hstack((self.vel, euler[0], euler[1], euler[2])))

        # Sequence steps
        #   simple test: acceleration magnitude threshold for ground contact
        if (self.acc[2,0] > 6*9.81 ):  # or abs(self.acc[0]) > 5 or abs(self.acc[1]) > 5): 
          if (time.time() - self.last_step) > 0.3:
            self.step_ind += 1
            self.last_step = time.time()

        #t = data.header.stamp.to_sec()
        t = rospy.Time.now
        x = np.array([data.position.x,data.position.y,data.position.z,data.orientation.w, data.orientation.x,data.orientation.y,data.orientation.z,t])

        # Sending mocap velocity data

        for i in range(3):
          self.vel[i,0] = min(max(self.vel[i,0],-8),8)

        velocitySignal = [self.vel[0,0]*2000, self.vel[1,0]*2000, self.vel[2,0]*2000, euler[0]*AngleScaling]
        if self.MJ_state == 0:
          xb_send(0, command.CMD_SET_MOCAP_VEL, pack('4h', *velocitySignal))
          rospy.sleep(0.001)
          #print velocitySignal
        else:
          rospy.sleep(0.001)


if __name__ == '__main__':
    try:
        VelocityRead()
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