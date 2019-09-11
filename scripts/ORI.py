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
salto_name = 2 # 1: Salto-1P Santa, 2: Salto-1P Rudolph, 3: Salto-1P Dasher

# Parameters
alpha_v = 0.8 # velocity first-order low-pass
alpha_a = 0.6 # acceleration first-order low-pass
dt = 0.01#(1.0/120.0)# 0.01 # Optitrack frame time step
rot_off = quaternion_about_axis(0,(1,1,1)) # robot rotation from Vicon body frame
pos_off = [0,0,0] #[0.0165,0.07531,-0.04] # coords of the robot origin in the Vicon body frame
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

#'''
k = [
   [-0.2000,         0,         0], # supposed to be -0.2124
   [      0,         0,   -0.0111],
   [ 0.0668,         0,         0],
   [      0,   -0.0662,         0],
   [      0,         0,   -0.0086],
   [      0,         0,    0.0009],
   [      0,         0,   -0.0014],
   [      0,         0,   -0.0029],
   [      0,         0,   -0.0056],
   [      0,         0,    0.0050],
   [-0.0364,         0,         0],
   [ 0.0240,         0,         0],
   [      0,   -0.0270,         0],
   [-0.0060,         0,         0],
   [-0.0416,         0,         0],
   [      0,    0.0421,         0],
   [ 0.0026,         0,         0],
   [      0,         0,    0.0008],
   [-0.0040,         0,         0],
   [      0,    0.0049,         0],
   [      0,         0,    0.0017],
   [-0.0014,   -0.0000,   -0.0000],
   [-0.0052,    0.0000,    0.0000],
   [ 0.0000,   -0.0016,   -0.0000],
   [-0.0000,    0.0028,    0.0000],
   [ 0.0000,   -0.0000,    0.0005],
   [ 0.0000,    0.0000,   -0.0006],
   [-0.0000,   -0.0000,   -0.0011],
   [ 0.0000,   -0.0000,   -0.0011],
   [ 0.0000,    0.0000,   -0.0005],
   [-0.0000,   -0.0000,   -0.0034]
] # New Dasher gains from runridMotor16.

k = np.matrix(k).T

x_op = np.array([0, -3.3, 0, 0, 3.3]) # Operating (equilibrium) point state
u_op = np.array([0, 0, 0.2275]) # Operating point equilibrium control input

usePlatform = 1
#'''

'''
k = [
   [-0.2093,         0,         0],
   [      0,         0,   -0.0102],
   [ 0.0726,         0,         0],
   [      0,   -0.0724,         0],
   [      0,         0,   -0.0107],
   [      0,         0,    0.0010],
   [      0,         0,   -0.0011],
   [      0,         0,   -0.0030],
   [      0,         0,   -0.0055],
   [      0,         0,   -0.0005],
   [-0.0359,         0,         0],
   [ 0.0233,         0,         0],
   [      0,   -0.0265,         0],
   [-0.0053,         0,         0],
   [-0.0400,         0,         0],
   [      0,    0.0403,         0],
   [ 0.0025,         0,         0],
   [      0,         0,    0.0005],
   [-0.0036,         0,         0],
   [      0,    0.0046,         0],
   [      0,         0,   -0.0010],
   [-0.0013,   -0.0000,    0.0000],
   [-0.0049,    0.0000,   -0.0000],
   [ 0.0000,   -0.0015,   -0.0000],
   [-0.0000,    0.0025,    0.0000],
   [-0.0000,    0.0000,    0.0003],
   [ 0.0000,   -0.0000,   -0.0007],
   [ 0.0000,    0.0000,   -0.0012],
   [-0.0000,    0.0000,   -0.0010],
   [ 0.0000,   -0.0000,   -0.0008],
   [-0.0000,    0.0000,   -0.0036]
] # New Dasher gains from runridMotor15.

k = np.matrix(k).T

x_op = np.array([0, -3.3, 0, 0, 3.3]) # Operating (equilibrium) point state
u_op = np.array([0, 0, 0.2275]) # Operating point equilibrium control input

usePlatform = 1
'''

'''
k = [[   -0.2080,         0,         0], # Gains now modified from runGridMotor10.mat
[         0,         0,    -0.0143],
[    0.0780,         0,         0],
[         0,   -0.0840,         0],
[         0,         0,   -0.0201],
[         0,         0,    0.0022],
[         0,         0,   -0.0002],
[         0,         0,   -0.0022],
[         0,         0,   -0.0017],
[         0,         0,    0.0031],
[   -0.0385,         0,         0],
[    0.0196,         0,         0],
[         0,   -0.0199,         0],
[   -0.0052,         0,         0],
[   -0.0400,         0,         0],
[         0,    0.0413,         0],
[    0.0030,         0,         0],
[         0,         0,    0.0008],
[   -0.0036,         0,         0],
[         0,    0.0053,         0],
[         0,         0,    0.0043],
[   -0.0030,         0,         0],
[   -0.0040,         0,         0],
[         0,   -0.0007,         0],
[         0,    0.0023,         0],
[         0,         0,    0.0004],
[         0,         0,   -0.0004],
[         0,         0,   -0.0006],
[         0,         0,   -0.0001],
[         0,         0,    0.0002],
[         0,         0,    0.0005]]
k = np.matrix(k).T

x_op = np.array([0, -3.3, 0, 0, 3.3]) # Operating (equilibrium) point state
u_op = np.array([0, 0, 0.24]) # Operating point equilibrium control input

usePlatform = 1
'''

'''
k = [[   -0.3000,         0,         0],
[         0,         0,    0.0070],
[    0.2500,         0,         0],
[         0,   -0.2500,         0],
[         0,         0,    0.0392],
[         0,         0,    0.0000],
[         0,         0,    0.0000],
[         0,         0,   -0.0000],
[         0,         0,   -0.0000],
[         0,         0,   -0.0000],
[   -0.0000,         0,         0],
[    0.0000,         0,         0],
[         0,   -0.0000,         0],
[   -0.0000,         0,         0],
[   -0.0000,         0,         0],
[         0,    0.0000,         0],
[    0.0000,         0,         0],
[         0,         0,    0.0000],
[   -0.0000,         0,         0],
[         0,    0.0000,         0],
[         0,         0,    0.0000],
[   -0.0000,         0,         0],
[   -0.0000,         0,         0],
[         0,   -0.0000,         0],
[         0,    0.0000,         0],
[         0,         0,    0.0000],
[         0,         0,   -0.0000],
[         0,         0,   -0.0000],
[         0,         0,   -0.0000],
[         0,         0,    0.0000],
[         0,         0,    0.0000]]
k = np.matrix(k).T
'''

# Scaling constants
AngleScaling = 3667; # rad to 15b 2000deg/s integrated 1000Hz
    # 180(deg)/pi(rad) * 2**15(ticks)/2000(deg/s) * 1000(Hz) = 938734
    # 938734 / 2**8 = 3667
LengthScaling = 256; # radians to 23.8 fixed pt radians
CurrentScaling = 256; # radians to 23.8 fixed pt radians

class ORI:
    def __init__(self):
        # Robot position variables
        self.pos = np.matrix([[0],[0],[0]])
        self.vel = np.matrix([[0],[0],[0]])
        self.acc = np.matrix([[0],[0],[0]])
        self.euler = np.array([0,0,0])
        self.step_ind = 0
        self.last_step = time.time()

        # Platform position variables
        self.platPos = np.matrix([[0],[0],[0]])
        self.platTrans = np.matrix([[1, 0, 0 ,0],[0, 1, 0 ,0],[0, 0, 0, 1],[0, 0, 0, 1]])
        self.platPos2 = np.matrix([[0],[0],[0]])
        self.platTrans2 = np.matrix([[1, 0, 0 ,0],[0, 1, 0 ,0],[0, 0, 0, 1],[0, 0, 0, 1]])

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
            [0.003, 0, 0],
            [0, 0.003, 0],
            [0, 0, 0.003]])
        self.P_init = np.matrix([
            [0.01, 0, 0, 0, 0, 0],
            [0, 0.01, 0, 0, 0, 0],
            [0, 0, 0.01, 0, 0, 0],
            [0, 0, 0, 10, 0, 0],
            [0, 0, 0, 0, 10, 0],
            [0, 0, 0, 0, 0, 10]])
        self.x = np.matrix([0,0,0,0,0,0]).T
        self.P = self.P_init



        # Flags and counters
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

        # ROS
        self.tf_pub = tf.TransformBroadcaster()

        # Control logs
        self.ctrl_pub_rol = rospy.Publisher('control/rol',Float32)
        self.ctrl_pub_pit = rospy.Publisher('control/pit',Float32)
        self.ctrl_pub_yaw = rospy.Publisher('control/yaw',Float32)
        self.ctrl_pub_ret = rospy.Publisher('control/ret',Float32)
        self.ctrl_pub_ext = rospy.Publisher('control/ext',Float32)
        self.ctrl_pub_flag = rospy.Publisher('control/flag',Float32)

        # Raibert Controller
        # Robot setpoint variables
        self.desx = 0.0
        self.desvx = 0.0
        self.desy = 0.0
        self.desvy = 0.0
        self.desax = 0.0
        self.desay = 0.0
        self.desyaw = 0.0
        self.startTime = time.time()
        self.stepOpt = 0

        self.desvx2 = 0.0
        self.desvy2 = 0.0

        self.landz = 0.0
        self.nextz = 0.0

        self.ind = 0
        self.wpT = time.time()

        # SETUP -----------------------
        setupSerial()
        queryRobot()

                # Motor gains format:
        #  [ Kp , Ki , Kd , Kaw , Kff     ,  Kp , Ki , Kd , Kaw , Kff ]
        #    ----------LEFT----------        ---------_RIGHT----------
        motorgains = [80,0,12,0,0, 0,0,0,0,0]
        thrustgains = [170,0,140,100,0,120]
        #motorgains = [0,0,0,0,0, 0,0,0,0,0]
        #thrustgains = [0,0,0, 0,0,0]
        # roll kp, ki, kd; yaw kp, ki, kd

        if salto_name == 1:
            motorgains = [80,80,0, 160,100,0, 120,17,0,0]
        elif salto_name == 2:
            motorgains = [80,80,0, 160,110,0, 110,18,0,0]
        elif salto_name == 3:
            motorgains = [80,40,0, 110,120,0, 75,12,0,0]

        duration = 1000#15000
        rightFreq = thrustgains # thruster gains
        if salto_name == 1:
            leftFreq = [0.16, 0.2, 0.5, .16, 0.12, 0.25] # Raibert-like gains
            #           xv xp xsat yv yp ysat
        elif salto_name == 2:
            leftFreq = [0.16, 0.2, 0.5, .16, 0.12, 0.25]
        elif salto_name == 3:
            leftFreq = [0.16, 0.2, 0.5, .16, 0.12, 0.25]

        phase = [65, 80] # Raibert leg extension
        #       retract extend
        telemetry = True
        repeat = False

        # Gains for actual Raibert controller
        #leftFreq = [2, 0.008, 1.5, 2, 0.008, 0.75]
        leftFreq = [2, 0.008, 2.0, 2, 0.008, 1.0]
        #           KPx  Kx  Vxmax  KPy  Ky  Vymax          

        self.manParams = manueverParams(0,0,0,0,0,0)
        self.params = hallParams(motorgains, duration, rightFreq, leftFreq, phase, telemetry, repeat)
        xb_send(0, command.SET_THRUST_OPEN_LOOP, pack('6h',*thrustgains))
        setMotorGains(motorgains)


        # Joystick --------------------
        pygame.init()
        self.joy = pygame.joystick.Joystick(0)
        self.joy.init()
        self.joyaxes = self.joy.get_numaxes()
        self.joyinputs = [0,0,0,0,0,0]
        self.joyyaw = 0


        # BEGIN -----------------------
        rospy.init_node('ORI')
        rospy.Subscriber('Robot_1/pose', Pose, self.callback)
        #rospy.Subscriber('vicon/Dasher/Dasher', TransformStamped, self.callback) # Vicon

        if usePlatform:
            rospy.Subscriber('Body_2/pose', Pose, self.callbackPlatform)
            rospy.Subscriber('Body_3/pose', Pose, self.callbackPlatform2)

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

        # get gyro bias
        unusedBiasSignal = [0]
        xb_send(0, command.GYRO_BIAS, pack('h', *unusedBiasSignal))
        time.sleep(0.1)

        # BALANCING ON TOE
        #'''
        modeSignal = [0]
        xb_send(0, command.ONBOARD_MODE, pack('h', *modeSignal))
        time.sleep(0.02)

        standTailGains = [180,0,20,0,0, 0,0,0,0,0]
        standThrusterGains = [250,0,180, 100,0,150]
        #motorgains = [100,100,0, 150,120,0, 80,15,0,0]
        #thrustgains = [0,0,0,0,0,0]

        zeroGains = [0,0,0,0,0, 0,0,0,0,0]
        xb_send(0, command.SET_THRUST_OPEN_LOOP, pack('6h', *standThrusterGains))
        time.sleep(0.02)

        xb_send(0, command.SET_PID_GAINS, pack('10h',*zeroGains))
        time.sleep(0.02)

        xb_send(0, command.RESET_BODY_ANG, pack('h', *unusedBiasSignal))
        time.sleep(0.02)

        xb_send(0, command.G_VECT_ATT, pack('h', *unusedBiasSignal))
        time.sleep(0.02)

        modeSignal = [3]
        xb_send(0,command.ONBOARD_MODE, pack('h', *modeSignal))
        time.sleep(0.02)
        #'''

        # start
        startTelemetrySave(self.numSamples)
        exp = [2]
        stopSignal = [0]
        xb_send(0, command.START_EXPERIMENT, pack('h', *exp))
        time.sleep(0.03)
        
        # BALANCING ON TOE
        '''
        time.sleep(1.5)
        xb_send(0, command.SET_PID_GAINS, pack('10h',*standTailGains))
        time.sleep(3.5)
        '''
        
        #startTelemetrySave(self.numSamples)

        # BALANCING ON TOE
        #'''
        # Mocap control ------------------
        modeSignal = [0]
        # Immediately on onboard control ------
        #self.onboard_control = True
        #self.use_joystick = False
        #modeSignal = [24]#[8]#[56]
        
        xb_send(0, command.ONBOARD_MODE, pack('h', *modeSignal))
        time.sleep(0.03)

        xb_send(0, command.SET_PID_GAINS, pack('10h', *motorgains))
        time.sleep(0.03)
        xb_send(0, command.SET_THRUST_OPEN_LOOP, pack('6h', *thrustgains))
        time.sleep(0.03)
        #'''

        self.started = 1

        self.startTime = time.time()
        self.wpT = time.time()
        self.ind = 0
        self.step_ind = 0
        self.last_step = self.startTime


        self.xbee_sending = 0
        print "Done"
        #'''

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

    def callbackPlatform(self, data):
        rot = data.orientation
        tr = data.position
        q = np.array([rot.x, rot.y, rot.z, rot.w])
        pos = np.array([tr.x, tr.y, tr.z])

        # Convert to homogeneous coordinates
        HV = quaternion_matrix(q)
        HV[0:3,3] = pos # Vicon to markers

        self.platPos = np.matrix([[tr.x],[tr.y],[tr.z]])
        self.platTrans = HV

    def callbackPlatform2(self, data):
        rot = data.orientation
        tr = data.position
        q = np.array([rot.x, rot.y, rot.z, rot.w])
        pos = np.array([tr.x, tr.y, tr.z])

        # Convert to homogeneous coordinates
        HV = quaternion_matrix(q)
        HV[0:3,3] = pos # Vicon to markers

        self.platPos2 = np.matrix([[tr.x],[tr.y],[tr.z]])
        self.platTrans2 = HV

    def callback(self, data):
        # Process Vicon data and send commands
        self.decimate_count += 1
        if self.decimate_count == decimate_factor:
            self.decimate_count = 0
        else:
            return

        # HACKY EXPERIMENT -----------
        if self.started and time.time() - self.startTime > 10 and self.extra_flag == 0 and self.vel[2,0] > 1.0 and self.pos[2,0] > 0.5:
            disturb = [0, 0, -320] # 64 ticks per degree
            xb_send(0, command.ADJUST_BODY_ANG, pack('3h', *disturb))
            self.extra_flag = 1

        '''
        if self.started and time.time() - self.startTime > 30 and self.extra_flag == 1 and self.vel[2,0] > 0 and self.pos[2,0] > 0.5:
            disturb = [0, 0, -192]
            xb_send(0, command.ADJUST_BODY_ANG, pack('3h', *disturb))
            self.extra_flag = 2
        '''

        # VICON DATA ------------------------------------------------
        # Extract transform from message
        #'''
        # Original Optitrack
        rot = data.orientation
        tr = data.position
        q = np.array([rot.x, rot.y, rot.z, rot.w])
        pos = np.array([tr.x, tr.y, tr.z])
        #'''

        '''
        # Vicon
        rot = data.transform.rotation
        tr = data.transform.translation
        q = np.array([rot.x, rot.y, rot.z, rot.w])
        pos = np.array([tr.x, tr.y, tr.z])
        '''

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

        if abs(self.euler[1]) > math.pi/4 or abs(self.euler[2]) > math.pi/4:
            print("Bad tracking!")
            return

        if self.unheard_flag == 0: # first message
            self.unheard_flag = 1
            self.pos = pos
            self.x = np.vstack((pos,0,0,0))
            #rospy.loginfo(rospy.get_caller_id() + ' FIRST CONTACT ')
        else: # subsequent messages after the first
            vel = (pos - self.pos)/dt
            acc = (vel - self.vel)/dt

            if (time.time() - self.last_step) > 0.1:
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
        if (self.acc[2,0] > 6*9.81  ):  # or abs(self.acc[0]) > 5 or abs(self.acc[1]) > 5): 
            if (time.time() - self.last_step) > 0.3:
                self.step_ind += 1
                self.last_step = time.time()


        # DEADBEAT --------------------------------------------------
        #waypts = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 3.27, 80, 3.0, 2]]) # in place deadbeat
        steppts = np.array([[0.0, 0.0, 0.05, 0.0,     3.0, 80, 0]]) # deadbeat step planner in place


        '''
        #Cantilever hops
        steppts = np.array([
            [-1.0, 0.2, 0.05, 0.0,     3.4, 80, 1],
            [-1.0, 0.2, 0.05, 0.0,     3.4, 80, 0],
            [-1.0, 0.2, 0.05, 0.0,     3.4, 80, 0],
            [-1.0, 0.2, 0.05, 0.0,     3.4, 80, 0],
            [-0.8, 0.2, 0.05, 0.0,     3.4, 80, 0],
            [-0.4, 0.2, 0.05, 0.0,     3.4, 80, 0],
            [0.0, 0.2, 0.05, 0.0,     3.4, 80, 0],
            [0.4, 0.2, 0.05, 0.0,     3.4, 80, 0],
            [0.8, 0.2, 0.05, 0.0,     3.4, 80, 0],
            [1.2, 0.2, 0.05, 0.0,     3.4, 80, 0],
            [1.3, 0.2, 0.05, 0.0,     3.4, 80, 0],
            [1.3, 0.2, 0.05, 0.0,     3.4, 80, 0],
            [1.3, 0.2, 0.05, 0.0,     3.4, 80, 0],
            [1.3, 0.2, 0.05, 0.0,     3.4, 80, 0],
            [1.3, 0.2, 0.05, 0.0,     3.4, 80, 0],
            [1.3, 0.2, 0.05, 0.0,     3.4, 80, 0],
            [1.3, 0.2, 0.05, 0.0,     3.4, 80, 0],
            [1.3, 0.2, 0.05, 0.0,     3.4, 80, 0],
            [1.3, 0.2, 0.05, 0.0,     3.4, 80, 0],
            [1.3, 0.2, 0.05, 0.0,     3.4, 80, 0],
            [1.2, 0.2, 0.05, 0.0,     3.4, 80, 0],
            [0.8, 0.2, 0.05, 0.0,     3.4, 80, 0],
            [0.4, 0.2, 0.05, 0.0,     3.4, 80, 0],
            [0.0, 0.2, 0.05, 0.0,     3.4, 80, 0],
            [-0.4, 0.2, 0.05, 0.0,     3.4, 80, 0],
            [-0.8, 0.2, 0.05, 0.0,     3.4, 80, 0],
            ]) # deadbeat step planner up ramp onto cantilever
        '''


        '''
        # Hopping to different heights
        steppts = np.array([[0.0, 0.0, 0.05, 0.0,     3.4, 80, 1],
            [0.0, 0.0, 0.05, 0.0,     3.4, 80, 0],
            [0.0, 0.0, 0.05, 0.0,     3.4, 80, 0],
            [0.0, 0.0, 0.05, 0.0,     2.6, 80, 0],
            [0.0, 0.0, 0.05, 0.0,     2.6, 80, 0],
            [0.0, 0.0, 0.05, 0.0,     3.0, 80, 0],
            [0.0, 0.0, 0.05, 0.0,     3.0, 80, 0],
            [0.0, 0.0, 0.05, 0.0,     3.8, 80, 0],
            [0.0, 0.0, 0.05, 0.0,     3.8, 80, 0]]) # deadbeat step planner in place
        '''


        '''
        # High and low
        steppts = np.array([
            [0.5, -0.3, 0.05, 0.00,   3.3, 70, 0],
            [0.5, -0.3, 0.05, 0.00,   2.5, 80, 0],
            [0.5, -0.3, 0.05, 0.00,   2.5, 80, 0],
            [0.5, -0.3, 0.05, 0.00,   2.5, 80, 0],
            [0.5, -0.3, 0.05, 0.00,   2.5, 80, 0],
            [0.5, -0.3, 0.05, 0.00,   2.5, 80, 0],
            [0.5, -0.3, 0.05, 0.00,   2.5, 80, 0],
            [0.5, -0.3, 0.05, 0.00,   2.5, 80, 0],
            [0.5, -0.3, 0.05, 0.00,   2.5, 80, 0],
            [0.5, -0.3, 0.05, 0.00,   2.5, 80, 0],
            [0.5, -0.3, 0.05, 0.00,   2.5, 80, 0],
            [0.5, -0.3, 0.05, 0.00,   2.5, 80, 0],
            [0.5, -0.3, 0.05, 0.00,   2.5, 80, 0],
            [0.5, -0.3, 0.05, 0.00,   2.5, 80, 0],
            [0.5, -0.3, 0.05, 0.00,   2.5, 80, 0],
            [0.5, -0.3, 0.05, 0.00,   2.5, 80, 0],
            [0.5, -0.3, 0.05, 0.00,   2.5, 80, 0],
            [0.5, -0.3, 0.05, 0.00,   2.5, 80, 0],
            [0.5, -0.3, 0.05, 0.00,   2.5, 80, 0],
            [0.5, -0.3, 0.05, 0.00,   2.5, 80, 0],
            [0.5, -0.3, 0.05, 0.00,   2.5, 80, 0],
            [0.5, -0.3, 0.05, 0.00,   2.8, 80, 0],
            [0.5, -0.3, 0.05, 0.00,   3.3, 80, 0],
            [0.5, -0.3, 0.05, 0.00,   2.8, 80, 0],
            [0.5, -0.3, 0.05, 0.00,   3.3, 80, 0],
            [0.5, -0.3, 0.05, 0.00,   2.8, 80, 0],
            [0.5, -0.3, 0.05, 0.00,   3.3, 80, 0],
            [0.5, -0.3, 0.05, 0.00,   2.8, 80, 0],
            [0.5, -0.3, 0.05, 0.00,   3.3, 80, 0],
            [0.5, -0.3, 0.05, 0.00,   2.8, 80, 0],
            [0.5, -0.3, 0.05, 0.00,   3.3, 80, 0],
            [0.5, -0.3, 0.05, 0.00,   2.8, 80, 0],
            [0.5, -0.3, 0.05, 0.00,   3.3, 80, 0],
            [0.5, -0.3, 0.05, 0.00,   2.8, 80, 0],
            [0.5, -0.3, 0.05, 0.00,   3.3, 80, 0],
            [0.5, -0.3, 0.05, 0.00,   3.5, 80, 0],
            ])
        '''

        '''
        # Turn in place
        steppts = np.array([
            [0.0, 0.5, 0.05, 0.00,    3.3, 70, 0],
            [0.0, 0.5, 0.05, 0.78,    3.3, 80, 0],
            [0.0, 0.5, 0.05, 1.57,    3.3, 80, 0],
            [0.0, 0.5, 0.05, 2.35,    3.3, 80, 0],
            [0.0, 0.5, 0.05, 3.14,    3.3, 80, 0],
            [0.0, 0.5, 0.05, 3.93,    3.3, 80, 0],
            [0.0, 0.5, 0.05, 4.71,    3.3, 80, 0],
            [0.0, 0.5, 0.05, 5.49,    3.3, 80, 0],
            [0.0, 0.5, 0.05, 0.00,    3.3, 80, 0],
            ])
        '''

        '''
        # Turn in place
        steppts = np.array([
            [-1.0, 0.5, 0.05, 0.0,    3.6, 80, 0],
            [-1.0, 0.5, 0.05, 0.0,    3.6, 80, 0],
            [-1.0, 0.5, 0.05, 0.0,    3.6, 80, 0],
            [-1.0, 0.5, 0.05, 1.5,    3.6, 80, 0],
            [-1.0, 0.5, 0.05, 3.1,    3.6, 80, 0],
            [-1.0, 0.5, 0.05, 3.1,    3.6, 80, 0],
            [-1.0, 0.5, 0.05, 3.1,    3.6, 80, 0],
            [-1.0, 0.5, 0.05, 3.1,    3.6, 80, 0],
            [-1.0, 0.5, 0.05, 3.1,    3.6, 80, 0],
            [-1.0, 0.5, 0.05, 1.5,    3.6, 80, 0],
            [-1.0, 0.5, 0.05, 0.0,    3.6, 80, 0],
            [-1.0, 0.5, 0.05, 0.0,    3.6, 80, 0],
            ])
        '''

        '''
        steppts = np.array([
            [-1.0, 0.0, 0.0, 0.05,    3.6, 80, 0],
            [-1.0, 0.0, 0.0, 0.05,    3.6, 80, 0],
            [-1.0, 0.0, 0.0, 0.05,    3.6, 80, 0],
            [-1.0, 0.0, 0.0, 0.19,    3.6, 80, 0],
            [-0.4, 0.0, 0.0, 0.19,    3.6, 80, 0],
            [0.2, 0.3, 0.0, 0.19,     3.6, 80, 0],
            [0.8, 0.0, 0.0, 0.19,     3.6, 80, 0],
            [1.0, 0.0, 0.0, 0.19,     3.6, 80, 0],
            [1.6, -0.3, 0.0, 0.19,    3.6, 80, 0],
            [2.0, -0.3, 0.0, 0.05,    3.6, 80, 0]
            ])
        '''

        '''
        # Jumping over hurdles
        steppts = np.array([
            [-1.6, 1.2, 0.07, 0.0,    3.0, 60, 0],
            [-1.6, 1.2, 0.07, 0.0,    3.0, 80, 0],
            [-1.6, 1.2, 0.07, 0.0,    3.0, 80, 0],
            [-1.6, 1.2, 0.07, 0.0,    3.4, 80, 0],
            [-0.7, 1.2, 0.07, 0.0,    3.4, 80, 0],
            [0.2, 1.2, 0.07, 0.0,     3.6, 80, 0],
            [2.0, 1.2, 0.07, 0.0,     3.3, 90, 0],
            [2.5, 1.2, 0.07, 0.0,     2.8, 80, 0],
            [2.7, 1.2, 0.07, 0.0,     3.0, 80, 0]
            ])
        '''

        '''
        # Back and forth hop on platforms
        steppts = np.array([
            [0.0, 0.0, 0.05, -1.57,  3.5, 80, 1],
            [0.0, 0.0, 0.05, -1.57,  3.5, 80, 2],
            [0.0, 0.0, 0.05, -1.57,  3.5, 80, 2],
            [2.0, 0.0, 0.05, -1.57,  3.5, 80, 0],
            [2.0, 0.0, 0.05, -1.57,  3.5, 80, 0]
            ])
        '''

        '''
        # Zigzag
        steppts = np.array([
            [-0.4, 0.0, 0.07, 0.0,    3.5, 80, 0],
            [-0.4, 0.0, 0.07, 0.0,    3.5, 80, 0],
            [-0.4, 0.0, 0.07, 0.0,    3.5, 80, 0],
            [0.4, -0.4, 0.27, 0.0,    3.5, 80, 0],
            [-0.4, -0.8, 0.37, 0.0,   3.5, 80, 0],
            [0.4, -1.2, 0.47, 0.0,    3.5, 80, 0],
            [0.8, -1.2, 0.57, 0.0,    3.0, 80, 0],
            [1.0, -1.2, 0.57, 0.0,    3.2, 80, 0],
            [2.0, -1.2, 0.07, 0.0,    3.2, 80, 0],
            [2.5, -1.2, 0.07, 0.0,    4.0, 80, 0],
            [2.5, -1.2, 0.07, 0.0,    3.0, 80, 0]
            ])
        '''

        '''
        # Filming
        steppts = np.array([
            [-1.6, 1.2, 0.07, 0.0,    3.0, 70, 0], # Start START hurdles
            [-1.6, 1.2, 0.07, 0.0,    3.0, 80, 0],
            [-1.6, 1.2, 0.07, 0.0,    3.0, 80, 0],
            [-1.6, 1.2, 0.07, 0.0,    3.4, 80, 0],
            [-0.7, 1.2, 0.07, 0.0,    3.4, 80, 0],
            [0.2, 1.2, 0.17, 0.0,     3.4, 80, 0],
            [1.6, 1.2, 0.07, 0.0,     3.4, 90, 0],
            [2.1, 1.2, 0.07, 0.0,     2.8, 80, 0],
            [2.1, 1.2, 0.07, 0.0,     3.0, 80, 0], # END Hurdles
            [2.1, 0.9, 0.05, 0.0,     3.0, 80, 0],
            [1.9, 0.6, 0.05, -0.79,   3.0, 80, 0],
            [1.7, 0.0, 0.05, -1.57,   3.5, 80, 0],
            [1.7, 0.0, 0.05, -1.57,   3.5, 80, 0], # START Platform back and forth
            [1.7, 0.0, 0.05, -1.57,   3.5, 80, 0],
            [0.0, 0.0, 0.05, -1.57,   3.5, 80, 2], 
            [0.0, 0.0, 0.05, -1.57,   3.5, 80, 2],
            [1.7, 0.0, 0.05, -1.57,   3.5, 80, 0],
            [1.7, 0.0, 0.05, -1.57,   3.5, 80, 0],
            [0.0, 0.0, 0.05, -1.57,   3.5, 80, 2],
            [0.0, 0.0, 0.05, -1.57,   3.5, 80, 2],
            [1.7, 0.0, 0.05, -1.57,   3.5, 80, 0],
            [1.7, 0.0, 0.05, -1.57,   3.5, 80, 0],
            [0.0, 0.0, 0.05, -1.57,   3.5, 80, 2],
            [0.0, 0.0, 0.05, -1.57,   3.5, 80, 2],
            [1.7, 0.0, 0.05, -1.57,   3.5, 80, 0],
            [1.7, 0.0, 0.05, -1.57,   3.5, 80, 0], # END Patform back and forth
            [1.7, 0.3, 0.05, -1.57,   3.2, 80, 0],
            [1.7, 0.3, 0.05, -0.79,   3.2, 80, 0],
            [1.7, 0.3, 0.05, 0.0,     3.2, 80, 0],
            [1.7, 0.3, 0.05, 0.0,     3.2, 80, 0],
            [1.2, 0.3, 0.05, 0.0,     3.2, 80, 0],
            [0.7, 0.3, 0.05, 0.0,     3.2, 80, 0],
            [0.2, 0.3, 0.05, 0.0,     3.2, 80, 0],
            [-0.3, 0.3, 0.05, 0.0,    3.2, 80, 0],
            [-0.6, 0.3, 0.05, 0.0,    3.2, 80, 0],
            [-0.6, 0.0, 0.05, 0.0,    3.2, 80, 0],
            [-0.4, 0.0, 0.07, 0.0,    3.2, 80, 0], 
            [-0.4, 0.0, 0.07, 0.0,    3.5, 80, 0], # START Zigzag
            [-0.4, 0.0, 0.07, 0.0,    3.5, 80, 0],
            [0.4, -0.4, 0.27, 0.0,    3.5, 80, 0],
            [-0.4, -0.8, 0.37, 0.0,   3.5, 80, 0],
            [0.4, -1.2, 0.47, 0.0,    3.5, 80, 0],
            [0.8, -1.2, 0.57, 0.0,    3.0, 80, 0],
            [0.8, -1.2, 0.57, 0.0,    3.0, 80, 0],
            [1.0, -1.2, 0.57, 0.0,    3.2, 80, 0],
            [2.0, -1.2, 0.07, 0.0,    3.2, 80, 0],
            [2.5, -1.2, 0.07, 0.0,    4.0, 80, 0],
            [2.5, -1.2, 0.07, 0.0,    3.0, 80, 0] # End END Zigzag
            ])
        '''


        '''
        # Random velocity commands
        steppts = np.array([
            [0.0, 0.0, 0.05, 0.0,     3.7, 80, 1],
            [0.0, 0.0, 0.05, 0.0,     3.7, 80, 0],
            [0.0, 0.0, 0.05, 0.0,     3.7, 80, 0],
            [0.0, 0.0, 0.05, 0.0,     3.7, 80, 0],
            [1.29, 0.22, 0.05, 0.0,   3.33, 80, 3],
            [0.95, -0.32, 0.05, 0.0,  3.70, 80, 3],
            [-1.15, 0.26, 0.05, 0.0,  3.20, 80, 3],
            [-0.66, -0.35, 0.05, 0.0, 3.03, 80, 3],
            [-0.44, 0.19, 0.05, 0.0,  3.74, 80, 3],
            ])
        '''

        '''
        # Random position commands
        steppts = np.array([
            [0,0,0.05,0,3.4,80,0],
            [0,0,0.05,0,3.4,80,0],
            [0,0,0.05,0,3.4,80,0],
            [0,0,0.05,0,3.4,80,0],
            [0,0,0.05,0,3.6,80,0],
            [0.6789,0.2263,0.05,0,3.7,80,0],
            [0.49976,0.075119,0.05,0,3.3097,80,0],
            [0.13346,-0.0098404,0.05,0,3.5231,80,0],
            [0.63651,-0.089558,0.05,0,3.7588,80,0],
            [1.9414,-0.2772,0.05,0,3.8181,80,0],
            [1.903,0.0015268,0.05,0,3.5586,80,0],
            [1.3948,-0.096677,0.05,0,3.587,80,0],
            [0.46556,-0.35597,0.05,0,3.4256,80,0],
            [1.4839,-0.24635,0.05,0,3.403,80,0],
            [1.7646,-0.35539,0.05,0,3.3382,80,0],
            [0.73514,-0.22398,0.05,0,3.6598,80,0],
            [0.0074349,0.037022,0.05,0,3.8348,80,0],
            [-1.2101,0.034415,0.05,0,3.7116,80,0],
            [-0.8531,-0.11154,0.05,0,3.7809,80,0],
            [-0.036723,-0.47239,0.05,0,3.853,80,0],
            [0.2762,-0.2681,0.05,0,3.4184,80,0],
            [0.94713,-0.38545,0.05,0,3.2825,80,0],
            [0.88426,-0.42921,0.05,0,2.8877,80,0],
            [1.238,-0.56011,0.05,0,2.903,80,0],
            [0.9234,-0.47594,0.05,0,2.6159,80,0],
            [1.2128,-0.44894,0.05,0,2.5336,80,0],
            [0.9723,-0.39974,0.05,0,2.4941,80,0],
            [0.7567,-0.35658,0.05,0,2.8786,80,0],
            [1.2118,-0.24798,0.05,0,2.7336,80,0],
            [1.3711,-0.15685,0.05,0,2.8151,80,0],
            [1.6339,-0.021904,0.05,0,3.212,80,0],
            [1.853,0.20396,0.05,0,3.604,80,0],
            [0.88125,0.26297,0.05,0,3.56,80,0],
            [0.19655,0.26105,0.05,0,3.3688,80,0],
            [0.011485,0.39138,0.05,0,3.0514,80,0],
            [-0.24227,0.26483,0.05,0,2.9969,80,0],
            [-0.49566,0.39503,0.05,0,2.8161,80,0],
            [-0.5489,0.3608,0.05,0,2.4055,80,0],
            [-0.28822,0.42199,0.05,0,2.8806,80,0],
            [-0.18575,0.39822,0.05,0,2.4364,80,0],
            [-0.15825,0.32813,0.05,0,2.8204,80,0],
            [-0.62347,0.31821,0.05,0,3.0789,80,0],
            [-0.51141,0.40896,0.05,0,3.2195,80,0],
            [-0.56844,0.55441,0.05,0,3.53,80,0],
            [-0.32475,0.67823,0.05,0,3.1034,80,0],
            [0.30894,0.60984,0.05,0,3.0005,80,0],
            [-0.2885,0.55681,0.05,0,3.3206,80,0],
            [0.13467,0.65949,0.05,0,3.0604,80,0],
            [0.36638,0.52766,0.05,0,3.1521,80,0],
            [0.5256,0.42972,0.05,0,2.7432,80,0],
            [0.36786,0.45557,0.05,0,2.6712,80,0],
            [0.80106,0.41499,0.05,0,2.8088,80,0],
            [1.146,0.28984,0.05,0,3.2028,80,0],
            [1.6582,0.12989,0.05,0,3.6738,80,0],
            [1.39,0.11083,0.05,0,3.6594,80,0],
            [0.75796,0.049975,0.05,0,3.3578,80,0],
            [0.22434,0.050241,0.05,0,3.2299,80,0],
            [0.078038,0.018854,0.05,0,3.3464,80,0],
            [-0.20826,-0.027991,0.05,0,3.3776,80,0],
            [0.31895,-0.19252,0.05,0,3.236,80,0],
            [-0.020059,-0.16686,0.05,0,2.8931,80,0],
            [-0.040325,-0.19106,0.05,0,2.6701,80,0],
            [0.32256,-0.17953,0.05,0,2.6102,80,0],
            [0.057159,-0.13901,0.05,0,2.7065,80,0],
            [-0.11808,-0.22841,0.05,0,3.1113,80,0],
            [0.33145,-0.10892,0.05,0,3.2693,80,0],
            [0.3301,-0.001128,0.05,0,2.8374,80,0],
            [0.028158,-0.011263,0.05,0,3.3076,80,0],
            [-0.59531,0.094278,0.05,0,3.5386,80,0],
            [-0.72668,0.2913,0.05,0,3.762,80,0],
            [0.21209,-0.030854,0.05,0,3.8695,80,0],
            [1.1413,-0.29904,0.05,0,3.5226,80,0],
            [1.232,-0.26519,0.05,0,3.344,80,0],
            [0.40218,-0.31189,0.05,0,3.542,80,0],
            [-0.19244,-0.54302,0.05,0,3.3835,80,0],
            [-0.8238,-0.75363,0.05,0,3.5237,80,0],
            [-0.81649,-0.87559,0.05,0,3.0599,80,0],
            [-0.41649,-0.94746,0.05,0,3.1323,80,0],
            [-1.0609,-0.80609,0.05,0,3.5794,80,0],
            [-0.15173,-1.0417,0.05,0,3.3552,80,0],
            [-0.69751,-1.0456,0.05,0,3.2219,80,0],
            [-0.56818,-1.0037,0.05,0,3.0642,80,0],
            [-0.45139,-1.106,0.05,0,3.3636,80,0],
            [-0.21965,-0.96463,0.05,0,2.8747,80,0],
            [-0.022182,-1.0131,0.05,0,2.6037,80,0],
            [0.11154,-0.9917,0.05,0,2.741,80,0],
            [0.084407,-0.90247,0.05,0,3.2231,80,0],
            [-0.0073246,-1.0941,0.05,0,3.3372,80,0],
            [-0.46387,-0.86252,0.05,0,3.3957,80,0],
            [-0.90388,-1.1566,0.05,0,3.8913,80,0],
            [-0.074108,-1.3504,0.05,0,3.7622,80,0],
            [0.67059,-1.0206,0.05,0,3.7822,80,0],
            [0.22429,-0.72536,0.05,0,3.8016,80,0],
            [0.29074,-0.96823,0.05,0,3.5542,80,0],
            [-0.78686,-0.88688,0.05,0,3.5132,80,0],
            [-0.85731,-0.77811,0.05,0,3.3925,80,0],
            [-1.8071,-0.86366,0.05,0,3.5788,80,0],
            [-0.60063,-0.75324,0.05,0,3.6659,80,0],
            [-0.032003,-0.9012,0.05,0,3.412,80,0],
            [-0.69893,-1.1384,0.05,0,3.3192,80,0],
            [-0.80133,-0.92017,0.05,0,3.5028,80,0],
            [-1.578,-0.92568,0.05,0,3.7693,80,0],
            [-1.6002,-1.1916,0.05,0,3.6505,80,0],
            [-0.5761,-1.0943,0.05,0,3.5915,80,0],
            ])
        '''

        '''
        # Random position commands
        steppts = np.array([
            [0,0,0.05,0,3.4,80,0],
            [0,0,0.05,0,3.4,80,0],
            [0,0,0.05,0,3.4,80,0],
            [0,0,0.05,0,3.4,80,0],
            [0,0,0.05,0,3.6,80,0],
            [0.6789,0.2263,0.05,0,3.7,80,0],
            [0.49976,0.075119,0.05,0,3.3097,80,0],
            [0.13346,-0.0098404,0.05,0,3.5231,80,0],
            [0.63651,-0.089558,0.05,0,3.7588,80,0],
            [1.9414,-0.2772,0.05,0,3.8181,80,0],
            [1.903,0.0015268,0.05,0,3.5586,80,0],
            [1.3948,-0.096677,0.05,0,3.587,80,0],
            [0.46556,-0.35597,0.05,0,3.4256,80,0],
            [1.4839,-0.24635,0.05,0,3.403,80,0],
            [1.7646,-0.35539,0.05,0,3.3382,80,0],
            [0.73514,-0.22398,0.05,0,3.6598,80,0],
            [0.0074349,0.037022,0.05,0,3.8348,80,0],
            [-1.2101,0.034415,0.05,0,3.7116,80,0],
            [-0.8531,-0.11154,0.05,0,3.7809,80,0],
            [-0.036723,-0.47239,0.05,0,3.853,80,0],
            [0.2762,-0.2681,0.05,0,3.4184,80,0],
            [0.94713,-0.38545,0.05,0,3.2825,80,0],
            [0.88426,-0.42921,0.05,0,2.8877,80,0],
            [1.238,-0.56011,0.05,0,2.903,80,0],
            [0.9234,-0.47594,0.05,0,2.6159,80,0],
            [1.2128,-0.44894,0.05,0,2.5336,80,0],
            [0.9723,-0.39974,0.05,0,2.4941,80,0],
            [0.7567,-0.35658,0.05,0,2.8786,80,0],
            [1.2118,-0.24798,0.05,0,2.7336,80,0],
            [1.3711,-0.15685,0.05,0,2.8151,80,0],
            [1.6339,-0.021904,0.05,0,3.212,80,0],
            [1.853,0.20396,0.05,0,3.604,80,0]])
        '''

        '''
        # Random position commands
        steppts = np.array([
            [0.0, 0.0, 0.05, 0.0,  3.5, 80, 0],
            [0.0, 0.0, 0.05, 0.0,  3.5, 80, 0],
            [0.0, 0.0, 0.05, 0.0,  3.5, 80, 0],
            [0.0, 0.0, 0.05, 0.0,  3.5, 80, 0],
            [   0   ,   0   ,   0.05    ,   0   ,   3.5 ,   80  ,   0   ],
            [   -0.27448    ,   0.198745    ,   0.05    ,   0   ,   2.8383  ,   80  ,   0   ],
            [   -0.543195   ,   0.39425 ,   0.05    ,   0   ,   3.1812  ,   80  ,   0   ],
            [   -1.08748    ,   0.4015  ,   0.05    ,   0   ,   3.2035  ,   80  ,   0   ],
            [   -0.72765    ,   0.42009 ,   0.05    ,   0   ,   3.685   ,   80  ,   0   ],
            [   -0.34825    ,   0.234208    ,   0.05    ,   0   ,   2.9361  ,   80  ,   0   ],
            [   0.02055 ,   0.313425    ,   0.05    ,   0   ,   2.8177  ,   80  ,   0   ],
            [   0.5693  ,   0.324079    ,   0.05    ,   0   ,   3.399   ,   80  ,   0   ],
            [   0.7317  ,   0.52271 ,   0.05    ,   0   ,   3.7409  ,   80  ,   0   ],
            [   0.5594  ,   0.217246    ,   0.05    ,   0   ,   3.6073  ,   80  ,   0   ],
            [   0.33956 ,   0.43951 ,   0.05    ,   0   ,   3.2089  ,   80  ,   0   ],
            [   0.7333  ,   0.2454589   ,   0.05    ,   0   ,   2.9425  ,   80  ,   0   ],
            [   1.2841  ,   -0.08502    ,   0.05    ,   0   ,   3.5554  ,   80  ,   0   ],
            [   0.9773  ,   -0.12623    ,   0.05    ,   0   ,   3.2178  ,   80  ,   0   ],
            [   0.33502 ,   -0.19386    ,   0.05    ,   0   ,   3.2842  ,   80  ,   0   ],
            [   -0.10515    ,   -0.30542    ,   0.05    ,   0   ,   3.1455  ,   80  ,   0   ],
            [   0.40022 ,   -0.46746    ,   0.05    ,   0   ,   3.1417  ,   80  ,   0   ],
            [   0.29031 ,   -0.34202    ,   0.05    ,   0   ,   2.958   ,   80  ,   0   ],
            [   0.7911  ,   -0.62313    ,   0.05    ,   0   ,   3.3079  ,   80  ,   0   ],
            [   0.10754 ,   -0.43884    ,   0.05    ,   0   ,   3.57    ,   80  ,   0   ],
            [   -0.28975    ,   -0.31366    ,   0.05    ,   0   ,   3.413   ,   80  ,   0   ],
            [   0.25245 ,   -0.26839    ,   0.05    ,   0   ,   3.792   ,   80  ,   0   ],
            [   0.14291 ,   -0.36493    ,   0.05    ,   0   ,   3.3379  ,   80  ,   0   ],
            [   -0.18679    ,   -0.07502    ,   0.05    ,   0   ,   3.3134  ,   80  ,   0   ],
            [   -0.545152   ,   -0.12512    ,   0.05    ,   0   ,   3.5728  ,   80  ,   0   ],
            [   -0.69417    ,   -0.36783    ,   0.05    ,   0   ,   3.7419  ,   80  ,   0   ],
            [   -0.589667   ,   -0.44629    ,   0.05    ,   0   ,   3.0439  ,   80  ,   0   ],
            [   -0.96512    ,   -0.43143    ,   0.05    ,   0   ,   3.2768  ,   80  ,   0   ],
            [   -0.597915   ,   -0.1046 ,   0.05    ,   0   ,   3.7321  ,   80  ,   0   ],
            [   -0.32329    ,   -0.38196    ,   0.05    ,   0   ,   3.7442  ,   80  ,   0   ],
            [   -0.7391 ,   -0.30762    ,   0.05    ,   0   ,   3.3973  ,   80  ,   0   ],
            [   -1.24147    ,   -0.67534    ,   0.05    ,   0   ,   3.6486  ,   80  ,   0   ],
            [   -0.93241    ,   -0.72141    ,   0.05    ,   0   ,   2.8625  ,   80  ,   0   ],
            [   -0.80649    ,   -0.48216    ,   0.05    ,   0   ,   3.6015  ,   80  ,   0   ],
            [   -0.75324    ,   -0.6739 ,   0.05    ,   0   ,   3.8088  ,   80  ,   0   ],
            [   -0.450493   ,   -0.447  ,   0.05    ,   0   ,   2.8757  ,   80  ,   0   ],
            [   -0.9107 ,   -0.27462    ,   0.05    ,   0   ,   3.5664  ,   80  ,   0   ],
            [   -0.541598   ,   -0.27335    ,   0.05    ,   0   ,   3.268   ,   80  ,   0   ],
            [   -0.38113    ,   -0.01674    ,   0.05    ,   0   ,   3.5379  ,   80  ,   0   ],
            [   -0.34675    ,   -0.16335    ,   0.05    ,   0   ,   3.5744  ,   80  ,   0   ],
            [   -0.529057   ,   -0.11124    ,   0.05    ,   0   ,   3.7766  ,   80  ,   0   ],
            [   0.00692 ,   0.200396    ,   0.05    ,   0   ,   3.8329  ,   80  ,   0   ],
            [   0.36749 ,   -0.0846 ,   0.05    ,   0   ,   2.8034  ,   80  ,   0   ],
            [   -0.10771    ,   -0.22241    ,   0.05    ,   0   ,   2.8251  ,   80  ,   0   ],
            [   -0.21282    ,   -0.33243    ,   0.05    ,   0   ,   3.3955  ,   80  ,   0   ],
            [   0.33836 ,   -0.46273    ,   0.05    ,   0   ,   3.1719  ,   80  ,   0   ],
            [   0.771   ,   -0.55874    ,   0.05    ,   0   ,   2.9519  ,   80  ,   0   ],
            [   0.7813  ,   -0.3244 ,   0.05    ,   0   ,   3.2227  ,   80  ,   0   ],
            [   1.0442  ,   -0.23847    ,   0.05    ,   0   ,   3.2954  ,   80  ,   0   ],
            [   1.0131  ,   0.02666 ,   0.05    ,   0   ,   2.8918  ,   80  ,   0   ],
            [   0.7037  ,   -0.01057    ,   0.05    ,   0   ,   3.4463  ,   80  ,   0   ],
            [   1.209   ,   -0.03124    ,   0.05    ,   0   ,   3.2812  ,   80  ,   0   ],
            [   0.8329  ,   -0.05575    ,   0.05    ,   0   ,   3.7469  ,   80  ,   0   ],
            [   0.7873  ,   -0.05704    ,   0.05    ,   0   ,   3.3362  ,   80  ,   0   ],
            [   0.47026 ,   -0.29989    ,   0.05    ,   0   ,   2.8741  ,   80  ,   0   ],
            [   1.0638  ,   -0.50377    ,   0.05    ,   0   ,   3.7478  ,   80  ,   0   ],
            [   1.3946  ,   -0.21253    ,   0.05    ,   0   ,   3.8318  ,   80  ,   0   ],
            [   0.8248  ,   -0.29642    ,   0.05    ,   0   ,   3.8786  ,   80  ,   0   ],
            [   1.0214  ,   -0.02727    ,   0.05    ,   0   ,   3.3305  ,   80  ,   0   ],
            [   0.41627 ,   0.04925 ,   0.05    ,   0   ,   3.0542  ,   80  ,   0   ],
            [   0.45507 ,   0.208332    ,   0.05    ,   0   ,   3.4682  ,   80  ,   0   ],
            [   0.5656  ,   0.166689    ,   0.05    ,   0   ,   3.0686  ,   80  ,   0   ],
            [   0.46513 ,   -0.1798 ,   0.05    ,   0   ,   3.4697  ,   80  ,   0   ],
            [   0.9953  ,   -0.41397    ,   0.05    ,   0   ,   2.8391  ,   80  ,   0   ],
            [   1.4378  ,   -0.55893    ,   0.05    ,   0   ,   2.8098  ,   80  ,   0   ],
            [   0.954   ,   -0.28278    ,   0.05    ,   0   ,   3.7679  ,   80  ,   0   ],
            [   0.38434 ,   -0.38656    ,   0.05    ,   0   ,   3.4528  ,   80  ,   0   ],
            [   0.5063  ,   -0.26647    ,   0.05    ,   0   ,   3.5128  ,   80  ,   0   ],
            [   0.4078  ,   -0.53286    ,   0.05    ,   0   ,   3.6271  ,   80  ,   0   ],
            [   0.01371 ,   -0.41804    ,   0.05    ,   0   ,   3.7431  ,   80  ,   0   ],
            [   -0.466679   ,   -0.14522    ,   0.05    ,   0   ,   2.8346  ,   80  ,   0   ],
            [   -0.07625    ,   0.05017 ,   0.05    ,   0   ,   2.8548  ,   80  ,   0   ],
            [   -0.01725    ,   0.335102    ,   0.05    ,   0   ,   3.1536  ,   80  ,   0   ],
            [   0.44118 ,   0.41094 ,   0.05    ,   0   ,   3.6686  ,   80  ,   0   ],
            [   0.8208  ,   0.1252  ,   0.05    ,   0   ,   3.1115  ,   80  ,   0   ],
            [   1.0629  ,   0.11704 ,   0.05    ,   0   ,   3.8701  ,   80  ,   0   ],
            [   1.3799  ,   0.160318    ,   0.05    ,   0   ,   3.1289  ,   80  ,   0   ],
            [   1.0526  ,   0.41972 ,   0.05    ,   0   ,   3.2915  ,   80  ,   0   ],
            [   0.57    ,   0.113   ,   0.05    ,   0   ,   3.7456  ,   80  ,   0   ],
            [   -0.15994    ,   0.42142 ,   0.05    ,   0   ,   3.7899  ,   80  ,   0   ],
            [   -0.13049    ,   0.189466    ,   0.05    ,   0   ,   2.9956  ,   80  ,   0   ],
            [   0.10804 ,   0.38121 ,   0.05    ,   0   ,   2.8383  ,   80  ,   0   ],
            [   0.44206 ,   0.6773  ,   0.05    ,   0   ,   3.1772  ,   80  ,   0   ],
            [   0.6133  ,   0.58006 ,   0.05    ,   0   ,   3.0381  ,   80  ,   0   ],
            [   0.9759  ,   0.72136 ,   0.05    ,   0   ,   3.1067  ,   80  ,   0   ],
            [   1.0734  ,   0.67463 ,   0.05    ,   0   ,   2.9013  ,   80  ,   0   ],
            [   0.47056 ,   0.66903 ,   0.05    ,   0   ,   3.1061  ,   80  ,   0   ],
            [   0.27533 ,   0.53949 ,   0.05    ,   0   ,   2.988   ,   80  ,   0   ],
            [   0.15111 ,   0.66134 ,   0.05    ,   0   ,   3.024   ,   80  ,   0   ],
            [   0.37333 ,   0.62331 ,   0.05    ,   0   ,   3.2766  ,   80  ,   0   ],
            [   -0.08723    ,   0.40576 ,   0.05    ,   0   ,   3.4781  ,   80  ,   0   ],
            [   -0.448835   ,   0.45247 ,   0.05    ,   0   ,   3.8393  ,   80  ,   0   ],
            [   -0.11169    ,   0.59334 ,   0.05    ,   0   ,   3.8553  ,   80  ,   0   ],
            [   0.31973 ,   0.67774 ,   0.05    ,   0   ,   3.8428  ,   80  ,   0   ],
            [   -0.3779 ,   0.49453 ,   0.05    ,   0   ,   3.8853  ,   80  ,   0   ],
            [   0.01601 ,   0.4767  ,   0.05    ,   0   ,   3.549   ,   80  ,   0   ],
            [   -0.08677    ,   0.40255 ,   0.05    ,   0   ,   3.0346  ,   80  ,   0   ],
            [   -0.24522    ,   0.08424 ,   0.05    ,   0   ,   3.3196  ,   80  ,   0   ],
            [   -0.476971   ,   0.41515 ,   0.05    ,   0   ,   3.411   ,   80  ,   0   ],
            [   -0.00984    ,   0.35314 ,   0.05    ,   0   ,   3.3082  ,   80  ,   0   ],
            [   0.43939 ,   0.69126 ,   0.05    ,   0   ,   3.3763  ,   80  ,   0   ],
            [0.0, 0.0, 0.05, 0.0,  3.5, 80, 0],
            [0.0, 0.0, 0.05, 0.0,  3.5, 80, 0]
            ])
        '''

        '''
        # Velocity commands
        steppts = np.array([
            [0.0, 0.0, 0.05, 0.0,   3.7, 80, 1],
            [0.0, 0.0, 0.05, 0.0,   3.7, 80, 0],
            [0.0, 0.0, 0.05, 0.0,   3.7, 80, 0],
            [0.0, 0.0, 0.05, 0.0,   3.7, 80, 0],
            [1.0, 0.0, 0.05, 0.0,   3.5, 80, 3],
            [-0.5, 0.0, 0.05, 0.0,  3.5, 80, 3],
            [-0.5, 0.0, 0.05, 0.0,  3.9, 80, 3],
            [0.5, 0.0, 0.05, 0.0,   3.9, 80, 3],
            [0.5, 0.0, 0.05, 0.0,   3.5, 80, 3],
            [-1.0, 0.0, 0.05, 0.0,  3.5, 80, 3],
            [0.0, -0.5, 0.05, 0.0,  3.5, 80, 3],
            [0.5, 0.0, 0.05, 0.0,   3.5, 80, 3],
            [0.0, 0.5, 0.05, 0.0,   3.5, 80, 3],
            [-0.5, 0.0, 0.05, 0.0,  3.5, 80, 3],
            [0.0, 0.0, 0.05, 0.0,   3.7, 80, 3],
            ])
        '''

        '''
        # Back and forth hop on platform
        steppts = np.array([
            [0.0, 0.0, 0.05, 0.0,   3.7, 80, 1],
            [0.0, 0.0, 0.05, 0.0,   3.7, 80, 0],
            [-0.1, 0.0, 0.05, 0.0,  3.7, 80, 2],
            ])
        '''

        '''
        # Hop on platform
        steppts = np.array([
            [0.0, 0.0, 0.05, 0.0,   3.9, 80, 0],
            [0.0, 0.0, 0.05, 0.0,   3.5, 80, 0],
            [0.0, 0.0, 0.05, 0.0,   3.9, 80, 0],
            [-0.1, 0.0, 0.0, 0.0,   3.5, 80, 2],
            [-0.1, 0.0, 0.0, 0.0,   3.5, 80, 2],
            [0.0, 0.0, 0.05, 0.0,   3.5, 80, 0],
            [0.0, 0.0, 0.05, 0.0,   3.9, 80, 0],
            [-0.1, 0.0, 0.0, 0.0,   3.5, 80, 2],
            [-0.1, 0.0, 0.0, 0.0,   3.5, 80, 2],
            [0.0, 0.0, 0.05, 0.0,   3.5, 80, 0],
            [0.0, 0.0, 0.05, 0.0,   3.9, 80, 0],
            [-0.1, 0.0, 0.0, 0.0,   3.5, 80, 2],
            [-0.1, 0.0, 0.0, 0.0,   3.5, 80, 2],
            [0.0, 0.0, 0.05, 0.0,   3.5, 80, 0],
            [0.0, 0.0, 0.05, 0.0,   3.9, 80, 0],
            [-0.1, 0.0, 0.0, 0.0,   3.5, 80, 2],
            [-0.1, 0.0, 0.0, 0.0,   3.5, 80, 2],
            [0.0, 0.0, 0.05, 0.0,   3.5, 80, 0],
            [0.0, 0.0, 0.05, 0.0,   3.9, 80, 0],
            [-0.1, 0.0, 0.0, 0.0,   3.5, 80, 2],
            [-0.1, 0.0, 0.0, 0.0,   3.5, 80, 2],
            [0.0, 0.0, 0.05, 0.0,   3.5, 80, 0],
            [0.0, 0.0, 0.05, 0.0,   3.9, 80, 0],
            [-0.1, 0.0, 0.0, 0.0,   3.5, 80, 2],
            [-0.1, 0.0, 0.0, 0.0,   3.5, 80, 2],
            ])
        '''

        '''
        # Hop onto chair and desk
        steppts = np.array([
            [-1.00, 0.00, 0.05, 0.00,    3.9, 80, 1],
            [-1.00, 0.00, 0.05, 0.00,    3.9, 80, 0],
            [-1.00, 0.00, 0.05, 0.00,    3.9, 80, 0],
            [-1.00, 0.00, 0.05, 0.00,    3.9, 80, 0],
            [-0.50, 0.00, 0.05, 0.00,    3.9, 80, 0],
            [-0.00, 0.00, 0.45, 0.00,    3.9, 80, 0], # up onto chair
            [-0.00, 0.00, 0.45, 0.00,    3.2, 80, 0], # stabilize
            [-0.00, 0.00, 0.45, 0.00,    3.7, 80, 0],
            [-0.00,-0.05, 0.45, 0.00,    3.9, 80, 0],
            [-0.00,-0.50, 0.77, 0.00,    3.9, 80, 0], # up onto desk
            [-0.00,-0.50, 0.77, 0.00,    3.2, 80, 0], # stabilize
            [-0.30,-0.50, 0.77, 0.00,    3.5, 80, 0],
            [-0.70,-0.50, 0.77, 0.00,    3.9, 80, 0],
            [-0.70,-0.50, 0.77, 0.00,    3.9, 80, 0],
            [-0.30,-0.50, 0.77, 0.00,    3.9, 80, 0],
            [ 0.05,-0.50, 0.77, 0.00,    3.5, 80, 0],
            [ 0.05,-0.45, 0.77, 0.00,    3.3, 80, 0],
            [ 0.05,-0.00, 0.45, 0.00,    3.0, 80, 0], # jump down to chair
            [ 0.05,-0.00, 0.45, 0.00,    3.5, 80, 0], # stabilize
            [ 0.05,-0.00, 0.45, 0.00,    3.0, 80, 0], # jump down to ground
            [-0.50, 0.00, 0.05, 0.00,    3.5, 80, 0], # run back
            [-1.00, 0.00, 0.05, 0.00,    3.5, 80, 0],
            ])
        '''

        
        '''
        # Box hop
        steppts = np.array([
            [-0.5, 0.0, 0.0, 0.05,    3.8, 80, 1],
            [-0.5, 0.0, 0.0, 0.05,    3.8, 80, 0],
            [-0.5, 0.0, 0.0, 0.05,    3.8, 80, 0],
            [-0.5, 0.0, 0.0, 0.05,    3.8, 80, 0],
            [0.5,  0.0, 0.0, 0.05,    3.8, 80, 0],
            [0.5, -0.5, 0.0, 0.05,    3.8, 80, 0],
            [-0.5,-0.5, 0.0, 0.05,    3.8, 80, 0],
            [-0.5, 0.0, 0.0, 0.05,    3.4, 80, 0],
            [-0.5, 0.0, 0.0, 0.05,    3.4, 80, 0],
            [-0.5, 0.0, 0.0, 0.05,    3.4, 80, 0],
            [0.5,  0.0, 0.0, 0.05,    3.4, 80, 0],
            [0.5,  0.5, 0.0, 0.05,    3.4, 80, 0],
            [-0.5, 0.5, 0.0, 0.05,    3.4, 80, 0],
            ])
        '''

        '''
        # Try other right angles
        steppts = np.array([
            [-1.0, 0.0, 0.0, 0.05,    3.8, 80, 0],
            [-1.0, 0.0, 0.0, 0.05,    3.8, 80, 0],
            [-1.0, 0.0, 0.0, 0.05,    3.8, 80, 0],
            [0.0, 0.0, 0.0, 0.05,     3.8, 80, 0],
            [0.5, 0.0, 0.0, 0.05,     3.8, 80, 0],
            [0.5, 0.5, 0.0, 0.05,     3.8, 80, 0],
            [0.0, 0.5, 0.0, 0.05,     3.8, 80, 0],
            [0.3, 0.5, 0.0, 0.05,     3.2, 80, 0],
            [0.6, 0.5, 0.0, 0.05,     3.2, 80, 0],
            [0.9, 0.5, 0.0, 0.05,     3.2, 80, 0],
            [1.2, 0.5, 0.0, 0.05,     3.8, 80, 0],
            [1.5, 0.5, 0.0, 0.05,     3.8, 80, 0],
            [1.8, 0.5, 0.0, 0.05,     3.2, 80, 0]
            ])
        '''

        '''
        # Too hard because of right angle turns
        steppts = np.array([
            [-1.0, 0.0, 0.0, 0.05,    3.8, 80, 0],
            [-1.0, 0.0, 0.0, 0.05,    3.8, 80, 0],
            [-1.0, 0.0, 0.0, 0.05,    3.8, 80, 0],
            [-0.5, 0.0, 0.0, 0.05,    3.8, 80, 0],
            [0.5, 0.0, 0.0, 0.05,     3.8, 80, 0],
            [0.0, 0.0, 0.0, 0.05,     3.8, 80, 0],
            [0.0, 0.5, 0.0, 0.05,     3.8, 80, 0],
            [0.3, 0.5, 0.0, 0.05,     3.8, 80, 0],
            [0.6, 0.5, 0.0, 0.05,     3.2, 80, 0],
            [0.9, 0.5, 0.0, 0.05,     3.2, 80, 0],
            [1.2, 0.5, 0.0, 0.05,     3.8, 80, 0],
            [1.5, 0.5, 0.0, 0.05,     3.8, 80, 0],
            [1.8, 0.5, 0.0, 0.05,     3.2, 80, 0]
            ])
        '''

        #'''
        # New Raibert velocity (first FPV path)
        steppts = np.array([
            [-0.3, 0.0, 0.05, 0.0,    3.5, 80, 1],
            [-0.3, 0.0, 0.05, 0.0,    3.5, 80, 0],
            [-0.3, 0.0, 0.05, 0.0,    3.5, 80, 0],
            [-0.3, 0.0, 0.05, 0.0,    3.5, 80, 0],
            [0.1, 0.0, 0.05, 0.0,    3.5, 80, 0],
            [0.5, 0.0, 0.05, 0.0,    3.5, 80, 0],
            [0.9, 0.0, 0.05, 0.0,    3.5, 80, 0],
            [0.9, 0.0, 0.05, 1.6,    3.5, 80, 0],
            [0.9, 0.0, 0.05, 1.6,    3.5, 80, 0],
            [0.9, 0.0, 0.05, 1.6,    3.5, 80, 0],
            [0.9, 0.3, 0.05, 1.6,    3.5, 80, 0],
            [0.9, 0.6, 0.05, 1.6,    3.5, 80, 0],
            [0.9, 0.6, 0.05, 0.0,    3.5, 80, 0],
            [0.9, 0.6, 0.05, 0.0,    3.5, 80, 0],
            [0.9, 0.6, 0.05, 0.0,    3.5, 80, 0],
            [0.5, 0.6, 0.05, 0.0,    3.5, 80, 0],
            [0.1, 0.6, 0.05, 0.0,    3.5, 80, 0],
            [-0.3, 0.6, 0.05, 0.0,    3.5, 80, 0],
            [-0.3, 0.4, 0.05, 0.0,    3.5, 80, 0],
            [-0.3, 0.2, 0.05, 0.0,    3.5, 80, 0],
            [-0.3, 0.0, 0.05, 0.0,    3.5, 80, 0],
            ])
        #'''

        '''
        # Wall jump
        steppts = np.array([
            [-0.3, 0.0, 0.05, 0.0,    3.5, 80, 0],
            [-0.3, 0.0, 0.05, 0.0,    3.5, 80, 0],
            [-0.3, 0.0, 0.05, 0.0,    3.5, 80, 0],
            [-0.3, 0.0, 0.05, 0.0,    3.5, 80, 0],
            [-0.3, 0.0, 0.05, 0.0,    3.8, 80, 0],
            [-0.3, 0.0, 0.05, 0.0,    3.8, 80, 0],
            [0.0, 0.0, 0.05, 0.0,    3.8, 80, 0],
            [0.5, 0.0, 0.05, 0.0,    3.8, 80, 0],
            [0.9, 0.0, 0.4,  0.0,    3.0, 80, 0],
            [-0.6, 0.0, 0.05, 0.0,    3.8, 80, 0],
            [-0.6, 0.0, 0.05,  0.0,    3.5, 80, 0],
            ])
        '''

        '''
        # Chimney ascent
        steppts = np.array([
            [-0.3, -0.6, 0.05, 1.6,    3.5, 80, 0],
            [-0.3, -0.6, 0.05, 1.6,    3.5, 80, 0],
            [-0.3, -0.6, 0.05, 1.6,    3.5, 80, 0],
            [-0.3, -0.6, 0.05, 1.6,    3.5, 80, 0],
            [-0.3, -0.6, 0.05, 1.6,    3.8, 80, 0],
            [-0.3, -0.6, 0.05, 1.6,    3.8, 80, 0],
            [0.0, -0.1, 0.05, 1.6,     3.8, 80, 0],
            [0.3, 0.6, 0.4, 1.6,       3.8, 80, 0],
            [0.5, -0.6, 0.6,  1.6,     3.0, 80, 0],
            [0.5, 0.6, 0.8, 1.6,       3.0, 80, 0],
            [0.5, -0.6, 0.6,  1.6,     3.0, 80, 0],
            [0.3, 0.6, 0.4,  1.6,      3.0, 80, 0],
            [-0.1, -0.6, 0.0,  1.6,    3.5, 80, 0],
            [-0.3, -0.8, 0.0,  1.6,    3.0, 80, 0],
            ])
        '''

        '''
        # Chimney ascent try 2
        steppts = np.array([
            [-0.3,  0.6, 0.05,  0.0,    3.3, 80, 0],
            [-0.3,  0.6, 0.05,  0.0,    3.3, 80, 0],
            [-0.3,  0.6, 0.05,  0.0,    3.5, 80, 0],
            [-0.3,  0.6, 0.05,  0.0,    3.5, 80, 0],
            [-0.3,  0.6, 0.05,  0.0,    3.8, 80, 0],
            [-0.3,  0.6, 0.05,  0.0,    3.8, 80, 0],
            [0.2,   0.2, 0.05,  0.0,    3.8, 80, 0],
            [1.2,  -0.2,  0.4,  0.0,    2.7, 80, 0],
            [-0.3, -0.3,  0.6,  0.0,    2.7, 80, 0],
            [1.2,  -0.3,  0.8,  0.0,    2.7, 80, 0],
            [-0.3, -0.3,  0.6,  0.0,    2.7, 80, 0],
            [1.2,  -0.2,  0.4,  0.0,    2.7, 80, 0],
            [0.4,   0.3,  0.0,  0.0,    3.5, 80, 0],
            [-0.3,  0.7,  0.0,  0.0,    3.0, 80, 0],
            ])
        '''


        '''
        # Good!2
        steppts = np.array([
            [-1.0, 0.0, 0.05, 0.0,    3.27, 80, 1],
            [-1.0, 0.0, 0.05, 0.0,    3.27, 80, 0],
            [-1.0, 0.0, 0.05, 0.0,    3.27, 80, 0],
            [-1.0, 0.0, 0.05, 0.0,    3.68, 80, 0],
            [-1.0, 0.0, 0.05, 0.0,    3.68, 80, 0],
            [-0.5, 0.0, 0.05, 0.0,    3.68, 80, 0],
            [0.5, 0.0, 0.05, 0.0,     3.68, 80, 0],
            [1.5, 0.0, 0.05, 0.0,     3.68, 80, 0],
            [2.0, 0.0, 0.05, 0.0,     3.27, 80, 0],
            [2.0, 0.0, 0.05, 0.0,     3.27, 80, 0],
            [2.0, 0.0, 0.05, 0.0,     3.68, 80, 0],
            [2.0, 0.0, 0.05, 0.0,     3.68, 80, 0],
            [1.5, 0.0, 0.05, 0.0,     3.68, 80, 0],
            [0.5, 0.0, 0.05, 0.0,     3.68, 80, 0],
            [-0.5, 0.0, 0.05, 0.0,    3.68, 80, 0],
            ])
        '''

        '''
        # Good!
        steppts = np.array([
            [-1.0, 0.0, 0.0, 0.05,    3.27, 80, 1],
            [-1.0, 0.0, 0.0, 0.05,    3.27, 80, 0],
            [-1.0, 0.0, 0.0, 0.05,    3.27, 80, 0],
            [-1.0, 0.0, 0.0, 0.05,    3.68, 80, 0],
            [-1.0, 0.0, 0.0, 0.05,    3.68, 80, 0],
            [-0.5, 0.0, 0.0, 0.05,    3.68, 80, 0],
            [0.5, 0.0, 0.0, 0.05,     3.68, 80, 0],
            [1.5, 0.0, 0.0, 0.05,     3.68, 80, 0],
            [2.0, 0.0, 0.0, 0.05,     3.27, 80, 0],
            [2.0, 0.0, 0.0, 0.05,     3.27, 80, 0],
            [2.0, 0.0, 0.0, 0.05,     3.68, 80, 0],
            [2.0, 0.0, 0.0, 0.05,     3.68, 80, 0],
            [1.5, 0.0, 0.0, 0.05,     3.68, 80, 0],
            [0.5, 0.0, 0.0, 0.05,     3.68, 80, 0],
            [-0.5, 0.0, 0.0, 0.05,    3.68, 80, 0],
            ])
        '''

        '''
        # Up and down
        steppts = np.array([
            [0.0, 0.0, 0.0, 0.0,     3.0, 80, 1],
            [0.0, 0.0, 0.0, 0.0,     3.0, 80, 0],
            [0.0, 0.0, 0.0, 0.0,     3.0, 80, 0],
            [0.0, 0.0, 0.0, 0.0,     3.0, 80, 0],
            [0.0, 0.0, 0.0, 0.0,     3.5, 80, 0],
            [0.0, 0.0, 0.0, 0.0,     3.5, 80, 0],
            [0.0, 0.0, 0.0, 0.0,     3.5, 80, 0],
            [0.0, 0.0, 0.0, 0.0,     2.5, 80, 0],
            [0.0, 0.0, 0.0, 0.0,     2.5, 80, 0],
            [0.0, 0.0, 0.0, 0.0,     2.5, 80, 0],
            [0.0, 0.0, 0.0, 0.0,     2.0, 80, 0],
            [0.0, 0.0, 0.0, 0.0,     2.0, 80, 0],
            [0.0, 0.0, 0.0, 0.0,     2.0, 80, 0],
            ])
        '''

        # HOPPING ---------------------------------------------------
        # ctrl = self.Deadbeat()

        #waypts = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 60, 80, 3.0, 2]]) # in place

        '''
        # Up and down Raibert
        waypts = np.array([
            [0.0, 0.0, 0.0,     0.0, 0.0, 60, 80, 3.0, 1], #stabilize
            [0.0, 0.0, 0.0,     0.0, 0.0, 65, 80, 4.0, 3],
            [0.0, 0.0, 0.0,     0.0, 0.0, 60, 80, 4.0, 3],
            [0.0, 0.0, 0.0,     0.0, 0.0, 70, 80, 4.0, 3],
            [0.0, 0.0, 0.0,     0.0, 0.0, 55, 80, 4.0, 3],
            ])
        '''


        '''
        # Forwards backwards deadbeat
        waypts = np.array([
            [0.0, 0.0, 0.0,     0.0, 0.0, 3.27, 80, 3.0, 1], # stabilize
            [1.8, 0.0, 0.0,     0.9, 0.0, 3.27, 80, 2.0, 3], #forwards
            [0.0, 0.0, 0.0,    -0.9, 0.0, 3.27, 80, 2.0, 3], #backwards
            ])
        '''

        '''
        # Box deadbeat
        waypts = np.array([
            [-0.6, -0.6, 0.0,   0.0, 0.0, 3.68, 80, 3.0, 1], # stabilize
            [1.8, -0.6, 0.0,    0.8, 0.0, 3.68, 80, 3, 3], #forwards
            [1.8, 0.6, 0.0,     0.0, 0.4, 3.68, 80, 3, 3], #left
            [-0.6, 0.6, 0.0,   -0.8, 0.0, 3.68, 80, 3, 3], #backwards
            [-0.6, -0.6, 0.0,   0.0, -0.4, 3.68, 80, 3, 3], #right
            ])
        '''

        '''
        # Forwards backwards
        waypts = np.array([
            [-0.3, 0.0, 0.0,    0.0, 0.0, 60, 80, 3.0, 1], # stabilize
            [1.8, 0.0, 0.0,     0.0, 0.0, 60, 80, 4.0, 2], #forwards
            [-0.3, 0.0, 0.0,    0.0, 0.0, 60, 80, 4.0, 2], #backwards
            ])
        #'''

        '''
        # D
        if salto_name == 1:
            waypts = np.array([
                [-0.5, -0.8, 0.0,       0.0, 0.0, 65, 78, 3.0, 0], # stabilize
                [1.5, -0.8, 0.0,        0.5, 0.0, 65, 78, 4.0, 2], # forwards
                [2.5, 0.2, math.pi/2,   0.0, 0.5, 65, 78, 4.0, 2], # curve
                [2.5, 1.2, math.pi/2,   0.0, 0.0, 65, 78, 2.0, 2], # straight
                [2.5, 1.2, math.pi,     0.0, 0.0, 65, 78, 3.0, 2], # turn
                [-0.5, 1.2, math.pi,    0.0, 0.0, 65, 78, 3.0, 2], # fast
                [-0.5, 1.2, math.pi,    0.0, 0.0, 65, 78, 2.0, 2], # recover
                [-0.5, 1.2, math.pi/2,  0.0, 0.0, 65, 78, 3.0, 2], # turn
                [-0.5, 0.2, math.pi/2, 0.0, 0.0, 65, 78, 2.0, 2], # backwards
                [-0.5, 0.2, 0.0,       0.0, 0.0, 65, 78, 3.0, 2], # turn
                [-0.5, 0.2, 0.0,       0.0, 0.0, 70, 78, 3.0, 2]]) # lower
            waypts = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 65, 78, 3.0, 2]])
        elif salto_name == 2 or salto_name == 3:
            waypts = np.array([
                [-0.5, -0.8, 0.0,       0.0, 0.0, 60, 80, 3.0, 0], # stabilize
                [1.5, -0.8, 0.0,        0.5, 0.0, 60, 80, 4.0, 2], # forwards
                [2.5, 0.2, math.pi/2,   0.0, 0.5, 60, 80, 4.0, 2], # curve
                [2.5, 1.2, math.pi/2,   0.0, 0.0, 60, 80, 2.0, 2], # straight
                [2.5, 1.2, math.pi,     0.0, 0.0, 60, 80, 3.0, 2], # turn
                [-0.5, 1.2, math.pi,    0.0, 0.0, 60, 80, 3.0, 2], # fast
                [-0.5, 1.2, math.pi,    0.0, 0.0, 60, 80, 2.0, 2], # recover
                [-0.5, 1.2, math.pi/2,  0.0, 0.0, 60, 80, 3.0, 2], # turn
                [-0.5, 0.2, math.pi/2, 0.0, 0.0, 60, 80, 2.0, 2], # backwards
                [-0.5, 0.2, 0.0,       0.0, 0.0, 60, 80, 3.0, 2], # turn
                [-0.5, 0.2, 0.0,       0.0, 0.0, 65, 80, 3.0, 2]]) # lower

        '''

        '''
        waypts = np.array([ # circle (Dasher)
            [0.0, 0.0, 0.0,         0.0, 0.0,  65, 80, 5.0, 1], # loop
            [0.5, 0.0, 0.0,         0.0, 0.3,  60, 80, 3.0, 0],
            [0.0, 0.5, math.pi/2,   -0.3, 0.0, 60, 80, 3.0, 0],
            [-0.5, 0.0, math.pi,    0.0, -0.3, 60, 80, 3.0, 0],
            [0.0, -0.5, -math.pi/2, 0.3, 0.0,  60, 80, 3.0, 0]])
        '''
        '''
        waypts = np.array([ # figure 8 (Rudolph)
            [0.0, 0.0, 0.0,         0.0, 0.0,  68, 80, 3.0, 1], # loop
            [0.6, 0.0, 0.0,         0.0, 0.5,  65, 80, 2.0, 0],
            [0.0, 0.6, 0.0,         -0.5, 0.0, 65, 80, 2.0, 0],
            [-0.6, 0.0, 0.0,        0.0, -0.5, 65, 80, 2.0, 0],
            [0.0, -0.6, math.pi/2,  0.5, 0.0,  65, 80, 2.0, 0],
            [0.6, 0.0, math.pi,     0.0, 0.5,  65, 80, 2.0, 0],
            [1.2, 0.6, math.pi,     -0.5, 0.0, 65, 80, 2.0, 0],
            [1.8, 0.0, math.pi,     0.0, -0.5, 65, 80, 2.0, 0],
            [1.2, -0.6, math.pi/2,  0.5, 0.0,  65, 80, 2.0, 0]])
        '''
        '''
        # Step up and down: step 0.4m high starts at 0.5m, ends at 1.1m
        waypts = np.array([
            [-0.2, 0.0, 0.0,  0.0, 0.0, 62, 80, 5.0, 0], # transition out on apex
            [0.8, 0.0, 0.0,  0.0, 0.0, 60, 80, 2.0, 0], # cubic vel. transition out on clock
            [0.8, 0.0, 0.0,  0.0, 0.0, 65, 80, 5.0, 2], # transition out on apex
            [1.8, 0.0, 0.0,  0.0, 0.0, 65, 80, 2.0, 0], # cubic vel. transition out at apex
            [1.8, 0.0, 0.0,  0.0, 0.0, 65, 80, 3.0, 2]])
        '''
        '''
        waypts = np.array([ # Santa
            [-0.2, 0.0, 0.0,  0.0, 0.0, 67, 80, 10.0, 0], # transition out on apex
            [0.8, 0.0, 0.0,  0.0, 0.0, 65, 80, 2.0, 0], # cubic vel. transition out on clock
            [0.8, 0.0, 0.0,  0.0, 0.0, 70, 80, 5.0, 2], # transition out on apex
            [1.8, 0.0, 0.0,  0.0, 0.0, 70, 80, 2.0, 0], # cubic vel. transition out at apex
            [1.8, 0.0, 0.0,  0.0, 0.0, 70, 80, 3.0, 2]])
        '''
        '''
        if salto_name == 3:
            waypts = np.array([ # obstacles (Dasher)
                [-0.3, -0.3, 0.0,       0.0, 0.0, 65, 80, 5.0, 0], # don't loop
                [0.5, -0.3, 0.0,        0.0, 0.0, 60, 80, 2.0, 0], # onto foam
                [0.5, -0.3, 0.0,        0.0, 0.0, 60, 82, 5.0, 0], # stay on foam
                [1.0, -0.3, 0.0,        0.0, 0.0, 60, 85, 3.0, 0], #
                [1.7, 0.0, math.pi/6,   0.0, 0.0, 60, 85, 3.0, 0], # onto ramp
                [1.7, 0.4, -math.pi/6,  0.0, 0.0, 60, 80, 3.0, 0], # stay on ramp
                [1.0, 0.7, 0.0,         0.0, 0.0, 65, 80, 5.0, 0], # down the ramp
                [1.0, 0.7, 0.0,         0.0, 0.0, 60, 80, 4.0, 0], # stabilize
                [0.3, 0.7, 0.0,         1.0, 0.0, 60, 85, 0.7, 1], # onto table
                [0.3, 0.7, 0.0,         0.0, 0.0, 60, 80, 5.0, 0], # stay on table
                [-0.3, 0.7, 0.0,        0.0, 0.0, 60, 80, 2.0, 0], # off the table
                [-0.3, 0.7, -math.pi/2, 0.0, 0.0, 60, 80, 4.0, 0], # turn corner
                [-0.3, -0.3, -math.pi/2,0.0, 0.0, 60, 80, 2.0, 0], # return to start
                [-0.3, -0.3, 0.0,       0.0, 0.0, 60, 80, 4.0, 0],
                [-0.3, -0.3, 0.0,       0.0, 0.0, 65, 80, 3.0, 0]]) # slow down
        elif salto_name == 2:
            waypts = np.array([ # obstacles (Rudolph)
                [-0.3, -0.3, 0.0,       0.0, 0.0, 67, 80, 5.0, 0], # don't loop
                [0.5, -0.3, 0.0,        0.0, 0.0, 63, 80, 2.0, 0], # onto foam
                [0.5, -0.3, 0.0,        0.0, 0.0, 60, 80, 5.0, 0], # stay on foam
                [1.0, -0.3, 0.0,        0.0, 0.0, 63, 80, 3.0, 0], #
                [1.7, 0.0, math.pi/6,   0.0, 0.0, 60, 85, 3.0, 0], # onto ramp
                [1.7, 0.4, -math.pi/6,  0.0, 0.0, 63, 80, 3.0, 0], # stay on ramp
                [1.0, 0.7, 0.0,         0.0, 0.0, 67, 80, 5.0, 0], # down the ramp
                [1.0, 0.7, 0.0,         0.0, 0.0, 63, 80, 4.0, 0], # stabilize
                [0.3, 0.7, 0.0,         1.0, 0.0, 60, 85, 0.7, 1], # onto table
                [0.3, 0.7, 0.0,         0.0, 0.0, 63, 80, 5.0, 0], # stay on table
                [-0.3, 0.7, 0.0,        0.0, 0.0, 63, 80, 2.0, 0], # off the table
                [-0.3, 0.7, -math.pi/2, 0.0, 0.0, 63, 80, 4.0, 0], # turn corner
                [-0.3, -0.3, -math.pi/2,0.0, 0.0, 63, 80, 2.0, 0], # return to start
                [-0.3, -0.3, 0.0,       0.0, 0.0, 63, 80, 4.0, 0],
                [-0.3, -0.3, 0.0,       0.0, 0.0, 67, 80, 3.0, 0]]) # slow down
        #'''

        # Read joystick
        pygame.event.pump() # joystick

        #self.Waypoints(waypts)
        self.Steppoints(steppts)
        #self.Trajectory('Rectangle')

        if self.use_joystick and not self.onboard_control: #joystick override velocity
            # still using mocap attitude control
            for i in range(self.joyaxes):
                self.joyinputs[i] = self.joy.get_axis(i)
            self.joyyaw = self.joyyaw -self.joyinputs[0]/100
            self.stepOpt = 3
            self.desvx = -2*self.joyinputs[4]
            self.desvy = -self.joyinputs[3]
            self.params.phase[0] = 3 - self.joyinputs[1] #3.67#3.27#3.5
            self.params.phase[1] = 80
            self.desyaw = self.joyyaw

        if self.ctrl_mode == 0:
            ctrl = self.Raibert()
        elif self.ctrl_mode == 1:
            ctrl = self.DeadbeatCurveFit1()
        else:
            ctrl = self.RaibertVelocity()
        # self.RaibertInspired()

        # ROBOT COMMANDS --------------------------------------------
        ES = [int(euler[0]*AngleScaling),int(euler[1]*AngleScaling),int(euler[2]*AngleScaling)]

        if np.isnan(ctrl[0]): # NaN check
            ctrl[0] = 0
        if np.isnan(ctrl[1]):
            ctrl[1] = 0
        if np.isnan(ctrl[2]):
            ctrl[2] = 65
        if np.isnan(ctrl[3]):
            ctrl[3] = 65

        CS = [int(ctrl[1]*AngleScaling),int(ctrl[2]*LengthScaling),int(ctrl[3]*CurrentScaling)]
        Cyaw = int(AngleScaling*self.desyaw)
        Croll = int(AngleScaling*ctrl[0])

        if self.MJ_state == 0 and self.started == 1:
            self.xbee_sending = 1

            '''
            rot_rol = quaternion_about_axis(Croll/AngleScaling,(1,1,-1))
            mat_rol = quaternion_matrix()
            rot_pit = quaternion_about_axis()
            mat_pit = quaternion_matrix()
            '''

            if self.onboard_control == False:
                toSend = [ES[0], ES[1], ES[2], Cyaw, Croll, CS[0], CS[1], CS[2]]
                for i in range(8):
                    if toSend[i] > 32767:
                        toSend[i] = 32767
                    elif toSend[i] < -32767:
                        toSend[i] = -32767
                xb_send(0, command.INTEGRATED_VICON, pack('8h',*toSend))
                self.xbee_sending = 0
            else:
                if self.use_joystick == False:
                    # Using deadbeat velocity planner (too agressive)
                    #vx1 = int(2000*(self.desvx2*np.cos(self.euler[0])+self.desvy2*np.sin(self.euler[0])))
                    #vy1 = int(2000*(self.desvy2*np.cos(self.euler[0])-self.desvx2*np.sin(self.euler[0])))
                    #vz1 = int(self.params.phase[0]*2000)

                    # Less agressive with yaw
                    vx1 = int(2000*(-1.0*(np.cos(self.euler[0])*self.pos[0,0]+np.sin(self.euler[0])*self.pos[1,0])))# - self.vel[0,0]))
                    vy1 = int(2000*(-1.0*(-np.sin(self.euler[0])*self.pos[0,0]+np.cos(self.euler[0])*self.pos[1,0])))# - self.vel[1,0]))
                    vz1 = int(2000*2.5)

                    # Less agressive without yaw
                    #vx1 = int(2000*(-1.0*self.pos[0,0]))# - self.vel[0,0]))
                    #vy1 = int(2000*(-1.0*self.pos[1,0]))# - self.vel[1,0]))
                    #vz1 = int(2000*2.5)

                    # Slow spin
                    #Cyaw = int(AngleScaling*max(0.1*(time.time()-self.startTime-10),0))

                    # Up and down
                    vz1 = int(2000*(3 + 0.5*np.sin(time.time() - self.startTime)))
                    vx1 = int(vx1*(vz1-1)/6000)
                    vy1 = int(vy1*(vz1-1)/6000)
                else:
                    # Read joystick ---------------------------------------------
                    for i in range(self.joyaxes):
                        self.joyinputs[i] = self.joy.get_axis(i)
                    self.joyyaw = self.joyyaw -0.5*self.joyinputs[0]/100
                    vz1 = int(np.sqrt(self.joyinputs[2]*1.7+2.3)*4000)
                    vx1 = int(-self.joyinputs[4]*6000*(vz1-2000)/6000)
                    vy1 = int(-self.joyinputs[3]*3000*(vz1-2000)/6000)
                    Cyaw = int(self.joyyaw*AngleScaling)

                    if self.joy.get_button(5):
                        stopSignal = [0]
                        xb_send(0,command.STOP_EXPERIMENT, pack('h', *stopSignal))
                        time.sleep(0.02)
                        xb_send(0,command.STOP_EXPERIMENT, pack('h', *stopSignal))
                        time.sleep(0.02)



                Croll = vy1
                CS[0] = vx1
                CS[2] = vz1

                toSend = [vx1,vy1,vz1, Cyaw]
                for i in range(4):
                    if toSend[i] > 32767:
                        toSend[i] = 32767
                    elif toSend[i] < -32767:
                        toSend[i] = -32767
                xb_send(0, command.SET_VELOCITY, pack('4h',*toSend))
                self.xbee_sending = 0
                #print(vx1, vy1, vz1, Cyaw)

            # Printing
            #print(self.step_ind,ctrl[1],ctrl[2],ctrl[3]) # Deadbeat
            #print(self.pos[0],self.pos[1],self.pos[2])
            ##print(Cyaw*57/AngleScaling,Croll*57/AngleScaling,CS[0]*57/AngleScaling)
            #print(euler[0]*57,euler[1]*57,euler[2]*57)
            #print([ES[0], ES[1], ES[2], Cyaw, Croll, CS[0]])
            #print(Cyaw, Croll, CS[0], CS[1], CS[2]) # Commands
            #print(ctrl[0],ctrl[1],ctrl[2],ctrl[3]) # cmds before scaling
            #print(np.hstack((ctrl.T, [self.acc])))
            #print(self.desx, self.desvx) # Raibert position
            #print(57*ES[0]/AngleScaling, 57*ES[1]/AngleScaling, 57*ES[2]/AngleScaling)
            #print "pos: %3.2f %3.2f %3.2f \tvel: %3.2f %3.2f \tacc: %3.2f %3.2f" % (self.desx, self.desy, self.desyaw, self.desvx, self.desvy, self.desax, self.desay)

            #print(int(self.acc[2]), self.step_ind, self.last_step)
            print(self.step_ind, ctrl[0], ctrl[1], ctrl[2], ctrl[3])

            # Publish commands
            self.ctrl_pub_rol.publish(Croll)
            self.ctrl_pub_pit.publish(CS[0])
            self.ctrl_pub_yaw.publish(Cyaw)
            self.ctrl_pub_ret.publish(CS[1])
            self.ctrl_pub_ext.publish(CS[2])

        elif self.MJ_state == 1:
            self.xbee_sending = 1
            toSend = [ES[0], ES[1], ES[2], Cyaw, int(0), int(0), int(70*256), int(70*256)]
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


        #'''
        # Original Optitrack
        #t = data.header.stamp.to_sec()
        t = rospy.Time.now
        x = np.array([data.position.x,data.position.y,data.position.z,data.orientation.w, data.orientation.x,data.orientation.y,data.orientation.z,t])
        self.tf_pub.sendTransform((x[4], x[5], x[6]), (x[0], x[1], x[2], x[3]), rospy.Time.now(), "jumper", "world")
        #'''

        '''
        # Vicon
        t = data.header.stamp.to_sec()
        x = np.array([data.transform.rotation.x,data.transform.rotation.y,data.transform.rotation.z,data.transform.rotation.w, data.transform.translation.x,data.transform.translation.y,data.transform.translation.z,t])
        self.tf_pub.sendTransform((x[4], x[5], x[6]), (x[0], x[1], x[2], x[3]), rospy.Time.now(), "jumper", "world")
        '''


    def Steppoints(self, pts): 
        # pts: [x, y, z, psi, vz, ext, options]
        # options: 
        #   0: usual format as above
        #   1: (only first node): loop
        #   2: jump onto the platform

        n_pts = pts.shape[0] # number of steps in our list

        # Loop or hold position when we reach the end of the step list
        prev_step = self.step_ind - 1
        if (self.step_ind == n_pts):
            if (pts[0,6] == 1): # loop and go back to the beginning
                self.step_ind = 1
                prev_step = n_pts - 1
            else: # no loop
                self.step_ind = n_pts-1
                prev_step = self.step_ind -1

        # Where do we want to aim after we touch down?

        if (pts[self.step_ind, 6] == 0 or self.step_ind == 0): # normal jump point
            self.desx = pts[self.step_ind, 0]
            self.desy = pts[self.step_ind, 1]
            self.nextz = pts[self.step_ind, 2]
            self.stepOpt = 0
            self.desvx = 0
            self.desvy = 0
        elif (pts[self.step_ind, 6] == 2): # jump to the platform
            ptPlat = np.matrix([[pts[self.step_ind,0]],[pts[self.step_ind,1]],[pts[self.step_ind,2]],[1]])
            ptPlatGlobal = self.platTrans.dot(ptPlat)
            self.desx = ptPlatGlobal[0,0]
            self.desy = ptPlatGlobal[1,0]
            self.nextz = ptPlatGlobal[2,0]
            self.stepOpt = 0
            self.desvx = 0
            self.desvy = 0
        elif (pts[self.step_ind, 6] == 3): # velocity command
            self.desvx = pts[self.step_ind, 0]
            self.desvy = pts[self.step_ind, 1]
            self.nextz = pts[self.step_ind, 2]
            self.stepOpt = 3
        elif (pts[self.step_ind, 6] == 4): # jump to the second platform
            ptPlat2 = np.matrix([[pts[self.step_ind,0]],[pts[self.step_ind,1]],[pts[self.step_ind,2]],[1]])
            ptPlatGlobal2 = self.platTrans2.dot(ptPlat2)
            self.desx = ptPlatGlobal2[0,0]
            self.desy = ptPlatGlobal2[1,0]
            self.nextz = ptPlatGlobal2[2,0]
            self.stepOpt = 0
            self.desvx = 0
            self.desvy = 0

        # Yaw and desired takeoff velocity
        self.desyaw = self.desyaw + min(max(pts[self.step_ind, 3]-self.desyaw,-yaw_rate*dt),yaw_rate*dt)
        if (self.ctrl_mode == 1 or self.ctrl_mode == 2): # Deadbeat
            self.params.phase[0] = pts[self.step_ind, 4]
            self.params.phase[1] = pts[self.step_ind, 5]
        elif (self.ctrl_mode == 0): # Raibert
            self.params.phase[0] = 65
            self.params.phase[1] = 80

        # Where are we going to touch down?
        if (self.step_ind == 0): # we are at the first point
            self.landz = pts[0, 2]
        else:
            if pts[prev_step, 6] == 2: # platform step
                ptPlat = np.matrix([[pts[self.step_ind-1,0]],[pts[self.step_ind-1,1]],[pts[self.step_ind-1,2]],[1]])
                ptPlatGlobal = self.platTrans.dot(ptPlat)
                self.landz = ptPlatGlobal[2,0]
            elif pts[prev_step, 6] == 4: # platform 2 step
                ptPlat2 = np.matrix([[pts[self.step_ind-1,0]],[pts[self.step_ind-1,1]],[pts[self.step_ind-1,2]],[1]])
                ptPlatGlobal2 = self.platTrans2.dot(ptPlat2)
                self.landz = ptPlatGlobal2[2,0]
            else: # normal step
                self.landz = pts[prev_step, 2]

        # Velocities for Raibert control
        if (self.ctrl_mode == 0): # Raibert control
            if (self.step_ind == 0): # we are at the first point
                self.desvx = 0
                self.desvy = 0
            else:
                #des_t = pts[self.step_ind, 4]/9.81 + (2*(self.landz + 0.5*pts[self.step_ind, 4]**2/9.81 - self.nextz)/9.81)**0.5
                des_t = 0.8
                self.desvx = (pts[self.step_ind, 0] - pts[self.step_ind -1, 0])/des_t
                self.desvy = (pts[self.step_ind, 1] - pts[self.step_ind -1, 0])/des_t


    def Waypoints(self, pts):
    	# pts: [x, y, psi, vx, vy, retract, extend, duration, options]
        #   x and y are the waypoint coordinates in meters
        #   psi is the yaw heading in radians
        #   vx and vy are the velocities at x and y
        #       ignored for the 1st waypoint (alway stationary) waypoint
        #   retract and extend are the leg setpoints in motor radians
        #   duration is the time for the leg that ends at x and y in seconds
        #   options:
        #       1st waypoint: 0: no loop, 1: loop
        #       others: 0,2: cubic velocity, 1,3: constant velocity
        #               0,1: clock start, 2,3: start at apex
        

        n_pts = pts.shape[0]
        t = time.time()
        
        # Waypoint selection
        if (self.ind == 0): # first point
            dur = pts[0,7]
            option = 2
            ptx0 = pts[0,0:3]
            ptv0 = np.array([0,0])
            ptDx = np.array([0,0,0])
            ptDv = np.array([0,0,0])
            self.params.phase[0] = pts[self.ind,5]
            self.params.phase[1] = pts[self.ind,6]
        elif (self.ind == n_pts): # last point
            if (pts[0,8] == 0 or n_pts <= 2): # no loop
                self.desx = pts[n_pts-1,0]
                self.desy = pts[n_pts-1,1]
                self.desyaw = pts[n_pts-1,2]
                self.desvx = 0.0
                self.desvy = 0.0
                self.desax = 0.0
                self.desay = 0.0
                self.params.phase[0] = pts[n_pts-1,5]
                self.params.phase[1] = pts[n_pts-1,6]
                return
            else: # loop
                dur = pts[1,7]
                option = pts[1,8]
                ptx0 = pts[n_pts-1,0:3]
                ptv0 = pts[n_pts-1,3:5]
                ptDx = pts[1,0:3] - pts[n_pts-1,0:3]
                ptDv = pts[1,3:5] - pts[n_pts-1,3:5]
                self.params.phase[0] = pts[1,5]
                self.params.phase[1] = pts[1,6]
        else: # intermediate point
            dur = pts[self.ind,7]
            option = pts[self.ind,8]
            ptx0 = pts[self.ind-1,0:3]
            ptv0 = pts[self.ind-1,3:5]
            ptDx = pts[self.ind,0:3] - pts[self.ind-1,0:3]
            ptDv = pts[self.ind,3:5] - pts[self.ind-1,3:5]
            self.params.phase[0] = pts[self.ind,5]
            self.params.phase[1] = pts[self.ind,6]
        
        # Yaw wrapping
        while (ptDx[2] > math.pi):
            ptDx[2] = ptDx[2] - 2*math.pi
        while (ptDx[2] < -math.pi):
            ptDx[2] = ptDx[2] + 2*math.pi

        # Interpolation coefficients
        # solution to:
        # xf-x0 = a*tf**3 + b*tf**2 + v0*tf
        # vf-v0 = 3*a*tf**2 + 2*b*tf
        ax = (2*dur*ptv0[0] - 2*ptDx[0] + dur*ptDv[0])/dur**3
        bx = (-3*dur*ptv0[0] + 3*ptDx[0] - dur*ptDv[0])/dur**2
        ay = (2*dur*ptv0[1] - 2*ptDx[1] + dur*ptDv[1])/dur**3
        by = (-3*dur*ptv0[1] + 3*ptDx[1] - dur*ptDv[1])/dur**2

        # Waypoint interpolation
        T = t - self.wpT
        T = min(T,dur)
        if (option == 1 or option == 3): # constant speed point
            self.desx = ptDx[0]*T/dur + ptx0[0]
            self.desy = ptDx[1]*T/dur + ptx0[1]
            self.desyaw = ptDx[2]*T/dur + ptx0[2]
            velInd = self.ind
            if velInd == n_pts:
                velInd = 1
            self.desvx = pts[velInd,3]
            self.desvy = pts[velInd,4]
            self.desax = 0.0
            self.desay = 0.0
        else: # normal point
            self.desx = ax*T**3 + bx*T**2 + ptv0[0]*T + ptx0[0]
            self.desy = ay*T**3 + by*T**2 + ptv0[1]*T + ptx0[1]
            self.desyaw = ptDx[2]*T/dur + ptx0[2]
            self.desvx = 3*ax*T**2 + 2*bx*T + ptv0[0]
            self.desvy = 3*ay*T**2 + 2*by*T + ptv0[1]
            self.desax = 6*ax*T + 2*bx
            self.desay = 6*ay*T + 2*by

        
        # next waypoint index
        if (T >= dur):
            if (option == 2 or option == 3) and \
                not(self.vel[2,0] < 0.5 and self.vel[2,0] > -0.5 and self.acc[2,0] < 0): 
                return # requires start from apex
                # TODO: this seems like a hack

            if (self.ind == n_pts):
                self.ind = 2
            else:
                self.ind = self.ind + 1
            self.wpT = time.time()

        #print "%u %3.2f %3.2f %3.2f %3.2f %3.2f %3.2f" % (self.ind, ptx0[0], ptx0[1], ptx0[2] ,ptDx[0], ptDx[1], ptDx[2])

    def Raibert(self):
        # Raibert controller
        # Parameters
        KPx = self.params.leftFreq[0] #Raibert position control gain ((m/s)/m)
        Kx = self.params.leftFreq[1] #Raibert velocity control gain (m/(m/s))
        Vxmax = self.params.leftFreq[2] #Raibert control max speed (m/s)
        KPy = self.params.leftFreq[3]
        Ky = self.params.leftFreq[4]
        Vymax = self.params.leftFreq[5]
        Ax = 0.01   # leg deflection for desired acceleration (m/(m/s^2))
        Ay = 0.01
        if salto_name == 1:
            Ts = 0.06
        else:
            Ts = 0.07   # stance time (seconds)
        L = 0.225   # leg length (meters)
        KV = 1.0    # between 0 and 1 (unitless)

        # Desired velocities
        RB = np.matrix([[np.cos(self.euler[0]),np.sin(self.euler[0])],[-np.sin(self.euler[0]),np.cos(self.euler[0])]])
        Berr = np.dot(RB,[[self.pos[0,0]-self.desx],[self.pos[1,0]-self.desy]])
        Bv = np.dot(RB,[[self.vel[0,0]],[self.vel[1,0]]])
        Bvdes = np.dot(RB,[[self.desvx],[self.desvy]])
        Bades = np.dot(RB,[[self.desax],[self.desay]])
        vxdes = -KPx*Berr[0] + KV*Bvdes[0]
        vxdes = max(min(vxdes,Vxmax),-Vxmax)
        vydes = -KPy*Berr[1] + KV*Bvdes[1]
        vydes = max(min(vydes,Vymax),-Vymax)

        # Desired foot distances
        xf = Bv[0]*Ts/2 + Kx*(Bv[0] - vxdes) - Ax*Bades[0]
        xf = max(min(xf,L/2),-L/2) # limit foot deflection
        yf = Bv[1]*Ts/2 + Ky*(Bv[1] - vydes) - Ay*Bades[1]
        yf = max(min(yf,L/2),-L/2) # limit foot deflection

        # Roll and pitch angles
        th = -math.asin(-xf/L) #TODO: fix pitch sign error
        ph = math.asin(yf/L)

        ctrl = [ph, -th, self.params.phase[0], self.params.phase[1]]
        #print(int(1000*xf),int(1000*yf))
        #print(int(57*th),int(57*ph))

        return ctrl

    '''
    def RaibertInspired(self):
        # Raibert-inspired controller
        ctrl = [-self.params.leftFreq[0]*(self.vel[0]-self.desvx) - self.params.leftFreq[1]*max(min(self.pos[0]-self.desx, self.params.leftFreq[2]),-self.params.leftFreq[2]), self.params.phase[0], self.params.phase[1]] # Simple controller
        yaw = 0.0 #TODO: make this a parameter
        roll = self.params.leftFreq[3]*(self.vel[1]-self.desvy) + self.params.leftFreq[4]*max(min(self.pos[1]-self.desy,self.params.leftFreq[5]),-self.params.leftFreq[5])

        # Correct deflections for yaw deflections
        roll_b = np.cos(self.euler[0])*roll + np.sin(self.euler[0])*ctrl[0]
        pitch_b = np.cos(self.euler[0])*ctrl[0] - np.sin(self.euler[0])*roll
        ctrl[0] = pitch_b
        ctrl = np.concatenate(([roll],ctrl), 1)

        return ctrl
    '''

    def DeadbeatCurveFit1(self):
        # 3D deadbeat controller
        desvz = self.params.phase[0] #3.67#3.27#3.5
        ext = self.params.phase[1]

        # Ballistic flight phase
        # (Currently assuming the ground height changes stepwise without slope)
        # time remain in current flight phase
        t_pred = self.vel[2,0]/9.81 + (self.vel[2,0]**2 + 2*9.81*max(self.pos[2,0] -0.25 -self.landz ,0))**0.5/9.81
        # predicted touchdown vertical velocity
        vv_pred = -(2*(0.5*self.vel[2,0]**2 + max(9.81*(self.pos[2,0] -0.25 -self.landz) ,0)))**0.5
        x_pred = self.pos[0,0] + t_pred*self.vel[0,0] # predicted x touchdown
        y_pred = self.pos[1,0] + t_pred*self.vel[1,0] # predicted y touchdown

        # Desired velocities on takeoff
        if (self.stepOpt == 0):
            errx = x_pred-self.desx
            erry = y_pred-self.desy
            des_t = desvz/9.81 + (2*(self.landz + 0.5*desvz**2/9.81 - self.nextz)/9.81)**0.5
            self.desvx2 = self.desvx - errx/des_t
            self.desvy2 = self.desvy - erry/des_t
        elif (self.stepOpt == 3):
            self.desvx2 = self.desvx
            self.desvy2 = self.desvy


        #if time.time() - self.last_step > 0.05: # Flight phase (normal operation)

        vh = (self.vel[0,0]**2 + self.vel[1,0]**2)**0.5 # incoming horizontal velocity
        vv = vv_pred # incoming vertical velocity
        heading = math.atan2(self.vel[1,0], self.vel[0,0]) # incoming horizontal velocity heading
        vxo = math.cos(heading)*self.desvx2 + math.sin(heading)*self.desvy2 # desired incoming-velocity-frame outgoing longitudinal velocity
        vyo = -math.sin(heading)*self.desvx2 + math.cos(heading)*self.desvy2 # desired outgoing lateral velocity in the incoming frame
        vzo = desvz # TODO: make this into a parameters

        vxo = max(min(vxo,2.5),-2.5)
        vyo = max(min(vyo,1.0),-1.0)

        x = np.array([vh,vv,vxo,vyo,vzo]) - x_op;

        x2 = np.hstack((x,
            x**2,
            x[0]*x[1], x[2]*x[1], x[3]*x[1],
            x[0]*x[4], x[2]*x[4], x[3]*x[4],
            x**3,
            x[0]*x[3]**2, x[2]*x[3]**2, x[0]**2*x[3], x[2]**2*x[3],
            x[0]**2*x[1], x[2]**2*x[1], x[3]**2*x[1],
            x[0]**2*x[4], x[2]**2*x[4], x[3]**2*x[4]))
        prl = k.dot(x2) # curve-fit controller
        prl = prl.T

        # Converting from incoming velocity frame to world frame for touchdown angles
        R1 = euler_matrix(heading - self.euler[0], prl[0], prl[1], 'rzyx')
        euler1 = euler_from_matrix(R1, 'ryxz')

        roll = euler1[1] + u_op[0] # roll angle (with added operating point input)
        pitch = euler1[0] + u_op[1] # pitch angle (with added operating point input)
        l = prl[2] - 0.1 + u_op[2] # leg length (subtracting offset and adding operating point input)

        #else: # Stance phase
        #    pitch = math.atan2(self.desvx2, self.desvz)
        #    roll = math.atan2(-self.desvy2, self.desvz)
        #    l = 80


        # Raibert comparison
        '''
        KRaibert = 0.010
        #l = np.interp(desvz,[0.0, 2.0, 3.0, 3.6, 3.7, 5.0],[70.0, 70.0, 65.0, 60.0, 55.0, 55.0])
        l = np.interp(desvz,[0.0, 2.0, 3.0, 3.6, 3.7, 5.0], [0.2436, 0.2436, 0.2394, 0.2347, 0.2293, 0.2293])
        dur = np.interp(desvz,[0.0, 2.0, 3.0, 3.6, 3.7, 5.0],[0.06, 0.06, 0.08, 0.09, 0.10, 0.10])
        xd = max(min(-dur*self.vel[0,0]/2 + KRaibert*(self.desvx2 - self.vel[0,0]),l/2),-l/2)
        pitch = math.sin(xd/l)
        yd = max(min(dur*self.vel[1,0]/2 - KRaibert*(self.desvy2 - self.vel[1,0]),l*math.cos(pitch)/2),-l*math.cos(pitch)/2)
        roll = math.sin(yd/(l*math.cos(pitch)))
        l = l - 0.1 # shift for crank function
        '''

        # calculate crank angle from foot extension
        crank = 1.109*10**5*l**5 - 2.511*10**4*l**4 + 1377*l**3 + 155.4*l**2 + 0.3454*l + 0.1792
        #crank = 2.034*10**5*l**5 - 6.661*10**4*l**4 + 7215*l**3 - 141.8*l**2 + 1.632*l + 0.03071
        motor = 25*crank + retractOffset # motor angle from crank angle and gear ratio

        motor = min(max(motor, 54),80)
        roll = min(max(roll, -math.pi/4), math.pi/4)
        pitch = min(max(pitch, -math.pi/4), math.pi/4)

        ctrl = [roll,pitch,motor,ext]

        return ctrl

    def RaibertVelocity(self):
        # 3D deadbeat controller
        desvz = self.params.phase[0] #3.67#3.27#3.5
        ext = self.params.phase[1]

        # Ballistic flight phase
        # (Currently assuming the ground height changes stepwise without slope)
        # time remain in current flight phase
        t_pred = self.vel[2,0]/9.81 + (self.vel[2,0]**2 + 2*9.81*max(self.pos[2,0] -0.25 -self.landz ,0))**0.5/9.81
        # predicted touchdown vertical velocity
        vv_pred = -(2*(0.5*self.vel[2,0]**2 + max(9.81*(self.pos[2,0] -0.25 -self.landz) ,0)))**0.5
        x_pred = self.pos[0,0] + t_pred*self.vel[0,0] # predicted x touchdown
        y_pred = self.pos[1,0] + t_pred*self.vel[1,0] # predicted y touchdown

        # Desired velocities on takeoff
        if (self.stepOpt == 3):
            self.desvx2 = self.desvx
            self.desvy2 = self.desvy
        else:
            errx = x_pred-self.desx
            erry = y_pred-self.desy
            des_t = desvz/9.81 + (2*(self.landz + 0.5*desvz**2/9.81 - self.nextz)/9.81)**0.5
            self.desvx2 = self.desvx - errx/des_t
            self.desvy2 = self.desvy - erry/des_t
            

        # Desired velocities
        RB = np.matrix([[np.cos(self.euler[0]),np.sin(self.euler[0])],[-np.sin(self.euler[0]),np.cos(self.euler[0])]])
        Bv = np.dot(RB,[[self.vel[0,0]],[self.vel[1,0]]])
        Bvdes = np.dot(RB,[[self.desvx2],[self.desvy2]])

        KRaibert = 0.006#0.008
        #l = np.interp(desvz,[0.0, 2.0, 3.0, 3.6, 3.7, 5.0],[70.0, 70.0, 65.0, 60.0, 55.0, 55.0])
        #l = np.interp(desvz,[0.0, 2.0, 3.0, 3.6, 3.7, 5.0], [0.2436, 0.2436, 0.2394, 0.2347, 0.2293, 0.2293])
        #dur = np.interp(desvz,[0.0, 2.0, 3.0, 3.6, 3.7, 5.0],[0.06, 0.06, 0.08, 0.09, 0.10, 0.10])
        l = 0.24#0.2347
        dur = 0.08
        xd = max(min(-dur*Bv[0]/2 + KRaibert*(Bvdes[0] - Bv[0]),l/2),-l/2)
        pitch = math.asin(xd/l)
        yd = max(min(dur*Bv[1]/2 - KRaibert*(Bvdes[1] - Bv[1]),l*math.cos(pitch)/2),-l*math.cos(pitch)/2)
        roll = math.asin(yd/(l*math.cos(pitch)))
        l = l - 0.1 # shift for crank function


        # calculate crank angle from foot extension
        crank = 2.034*10**5*l**5 - 6.661*10**4*l**4 + 7215*l**3 - 141.8*l**2 + 1.632*l + 0.03071 
        motor = 25*crank # motor angle from crank angle and gear ratio

        motor = min(max(motor, 52),80)
        roll = min(max(roll, -math.pi/6), math.pi/6)
        pitch = min(max(pitch, -math.pi/6), math.pi/6)

        ctrl = [roll,pitch,motor,ext]

        return ctrl


if __name__ == '__main__':
    try:
        ORI()
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
