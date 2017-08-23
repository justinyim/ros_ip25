#!/usr/bin/env python

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
import salto_config

EXIT_WAIT = False

# Salto:
salto_name = 1 # 1: Salto-1P Santa, 2: Salto-1P Rudolph, 3: Salto-1P Dasher

# Parameters
alpha_v = 0.5 # velocity first-order low-pass
alpha_a = 0.1 # acceleration first-order low-pass
dt = 0.01#(1.0/120.0)# 0.01 # Vicon frame time step
rot_off = quaternion_about_axis(2.094,(1,1,-1)) # robot rotation from Vicon body frame
pos_off = [0.0165,0.07531,-0.04] # coords of the robot origin in the Vicon body frame
#[0.00587, 0.0165, -0.07531]
#[0.0165,0.07531,-0.00587]

decimate_factor = 1

step_list = np.array([[1.0,0.0,4.0]])

# Pre-processing
off_mat = quaternion_matrix(rot_off)
if salto_name == 1:
    mis_mat = salto_config.offsets1
elif salto_name == 2:
    mis_mat = salto_config.offsets2
elif salto_name == 3:
    mis_mat = salto_config.offsets3
off_mat = np.dot(off_mat,mis_mat)
off_mat[0:3,3] = pos_off

k_file = sio.loadmat('/home/justin/Berkeley/FearingLab/Jumper/jumper/8_Bars/salto1p_v_poly_ctrler4b.mat')#salto1p_poly_ctrler1.mat')
k = k_file['a_nl'].T
k = np.delete(k, (2), axis = 0)

n_steps = len(step_list)

class VRI:
    def __init__(self):
        # Robot position variables
        self.pos = np.array([0,0,0])
        self.vel = np.array([0,0,0])
        self.acc = np.array([0,0,0])
        self.euler = np.array([0,0,0])
        self.step_ind = 0
        self.last_step = time.time()

        # Flags and counters
        self.decimate_count = 0
        self.telemetry_read = 0
        self.unheard_flag = 0
        self.xbee_sending = 1
        self.MJ_state = 0 # 0: run, 1: stand, 2: stop

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
        
        self.ind = 0
        self.wpT = time.time()

        # SETUP -----------------------
        setupSerial()
        queryRobot()

                # Motor gains format:
        #  [ Kp , Ki , Kd , Kaw , Kff     ,  Kp , Ki , Kd , Kaw , Kff ]
        #    ----------LEFT----------        ---------_RIGHT----------
        motorgains = [160,0,30,0,0,0,0,0,0,0]
        thrustgains = [170,0,120,170,0,120]
        # roll kp, ki, kd; yaw kp, ki, kd

        duration = 5000
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


        # BEGIN -----------------------
        rospy.init_node('VRI')
        if salto_name == 1:
            rospy.Subscriber('vicon/jumper1/jumper1', TransformStamped, self.callback)
            #rospy.Subscriber('vicon/jumper/body', TransformStamped, self.callback)
        elif salto_name == 2:
            rospy.Subscriber('vicon/jumper2/jumper2', TransformStamped, self.callback)
            #rospy.Subscriber('vicon/Rudolph/body', TransformStamped, self.callback)
        elif salto_name == 3:
            rospy.Subscriber('vicon/jumper3/jumper3', TransformStamped, self.callback)
            #rospy.Subscriber('vicon/Dasher/body', TransformStamped, self.callback)
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
            if self.MJ_state == 0:
                self.MJ_state = 1
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
        # Process Vicon data and send commands
        self.decimate_count += 1
        if self.decimate_count == decimate_factor:
            self.decimate_count = 0
        else:
            return

        # VICON DATA ------------------------------------------------
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
        self.euler = euler

        if abs(self.euler[1]) > math.pi/4 or abs(self.euler[2]) > math.pi/4:
            print("Bad tracking!")
            return

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

        # HOPPING ---------------------------------------------------
        # ctrl = self.Deadbeat()

        waypts = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 60, 80, 3.0, 2]]) # in place

        '''
        # MAST Boxes
        waypts = np.array([
            [-2.1, -2.1, 0.0,        0.0, 0.0, 62, 80, 6.0, 0], # stabilize
            [0.0, -2.1, 0.0,         0.0, 0.0, 55, 85, 3.0, 2], # jump up the step
            [0.0, -2.1, 0.0,         0.0, 0.0, 62, 80, 4.0, 2], # stabilize
            [2.0, -2.1, 0.0,         0.5, 0.0, 62, 80, 4.0, 2], # jump over terrain
            [4.0, -2.1, 0.0,         0.5, 0.0, 65, 80, 4.0, 2], # jump down ramp
            [4.75, -1.25, math.pi/2, 0.0, 0.5, 62, 80, 3.0, 0], # turning ...
            [4.0, -0.5, math.pi,     -0.5, 0.0, 62, 80, 3.0, 0], # turning ...
            [-3.0, -0.5, math.pi,    0.0, 0.0, 62, 80, 10.0, 0], # go across
            [-3.0, -0.5, math.pi/2,  0.0, 0.0, 62, 80, 4.0, 0], # turn
            [-3.0, -2.0, math.pi/2,  0.0, 0.0, 62, 80, 3.0, 0], # go across
            [-3.0, -2.0, math.pi/2,  0.0, 0.0, 67, 80, 4.0, 2]]) # stop

        if salto_name == 1:
            waypts = np.array([
                [-2.0, -2.1, 0.0,        0.0, 0.0, 67, 80, 6.0, 0], # stabilize
                [0.0, -2.1, 0.0,         0.0, 0.0, 60, 80, 3.0, 2], # jump up the step
                [0.0, -2.1, 0.0,         0.0, 0.0, 67, 80, 4.0, 2], # stabilize
                [2.0, -2.1, 0.0,         0.5, 0.0, 67, 80, 4.0, 2], # jump over terrain
                [4.0, -2.1, 0.0,         0.5, 0.0, 70, 80, 4.0, 2], # jump down ramp
                [4.75, -1.25, math.pi/2, 0.0, 0.5, 67, 80, 3.0, 0], # turning ...
                [4.0, -0.5, math.pi,     -0.5, 0.0, 67, 80, 3.0, 0], # turning ...
                [-3.0, -0.5, math.pi,    0.0, 0.0, 67, 80, 10.0, 0], # go across
                [-3.0, -0.5, math.pi/2,  0.0, 0.0, 67, 80, 4.0, 0], # turn
                [-3.0, -2.0, math.pi/2,  0.0, 0.0, 67, 80, 3.0, 0], # go across
                [-3.0, -2.0, math.pi/2,  0.0, 0.0, 67, 80, 4.0, 2]]) # stop
        '''

        #'''
        # MAST Flat
        waypts = np.array([
            [-1.0, -0.5, math.pi,     0.0, 0.0, 65, 80, 4.0, 0], # initial
            [-3.0, -0.5, math.pi,    0.0, 0.0, 60, 80, 4.0, 0], # go back
            [-3.0, -0.5, math.pi,    0.0, 0.0, 60, 80, 1.0, 0],
            [4.0, -0.5, math.pi,     0.5, 0.0, 60, 80, 4.0, 2], # go fast
            [4.75, 0.25, -math.pi/2, 0.0, 0.5, 62, 80, 3.0, 2], # turning ...
            [4.0, 1.0, 0.0,          -0.5, 0.0, 62, 80, 3.0, 0], # turning ...
            [3.0, 1.0, math.pi/2,    0.0, 0.0, 62, 80, 3.0, 0], # turn at end ...
            [2.0, 1.0, math.pi,      0.0, 0.0, 62, 80, 3.0, 0], # turn at end
            [-3.0, 1.0, math.pi,     0.0, 0.0, 62, 80, 8.0, 0],
            [-3.0, 1.0, -math.pi/2,  0.0, 0.0, 62, 80, 3.0, 0], # turn
            [-3.0, -2.0, -math.pi/2, 0.0, 0.0, 62, 80, 6.0, 0], # go across
            [-3.0, -2.0, -math.pi/2, 0.0, 0.0, 65, 80, 6.0, 2] # stop
            ])

        if salto_name == 1:
            waypts = np.array([
                [-1.0, -0.5, math.pi,     0.0, 0.0, 70, 80, 4.0, 0], # initial
                [-3.0, -0.5, math.pi,    0.0, 0.0, 65, 80, 4.0, 0], # go back
                [-3.0, -0.5, math.pi,    0.0, 0.0, 65, 80, 1.0, 0],
                [4.0, -0.5, math.pi,     0.5, 0.0, 65, 80, 4.0, 2], # go fast
                [4.75, 0.25, -math.pi/2, 0.0, 0.5, 67, 80, 3.0, 2], # turning ...
                [4.0, 1.0, 0.0,          -0.5, 0.0, 67, 80, 3.0, 0], # turning ...
                [3.0, 1.0, math.pi/2,    0.0, 0.0, 67, 80, 3.0, 0], # turn at end ...
                [2.0, 1.0, math.pi,      0.0, 0.0, 67, 80, 3.0, 0], # turn at end
                [-3.0, 1.0, math.pi,     0.0, 0.0, 67, 80, 8.0, 0],
                [-3.0, 1.0, -math.pi/2,  0.0, 0.0, 67, 80, 3.0, 0], # turn
                [-3.0, -2.0, -math.pi/2, 0.0, 0.0, 67, 80, 6.0, 0], # go across
                [-3.0, -2.0, -math.pi/2, 0.0, 0.0, 70, 80, 6.0, 2] # stop
                ])
        #'''

        '''
        # MAST try 2
        waypts = np.array([
            [2.0, 0.0, 0.0,          0.0, 0.0, 65, 80, 4.0, 0], # initial
            [4.0, 0.0, 0.0,          0.0, 0.0, 60, 80, 4.0, 0], # go forward
            [4.0, 0.0, 0.0,          0.0, 0.0, 60, 80, 1.0, 0],
            [-2.0, 0.0, 0.0,         0.0, 0.0, 60, 80, 4.0, 2], # go fast
            [-2.0, 0.0, 0.0,         0.0, 0.0, 62, 80, 3.0, 0], # stabilize
            [-2.0, 0.0, -math.pi/2,  0.0, 0.0, 62, 80, 3.0, 0], # turn
            [-2.0, -2.1, -math.pi/2, 0.0, 0.0, 62, 80, 4.0, 0], # go across
            [-2.0, -2.1, 0.0,        0.0, 0.0, 62, 80, 3.0, 0], # turn
            [-2.0, -2.1, 0.0,        0.0, 0.0, 62, 80, 5.0, 2], # stabilize
            [0.0, -2.1, 0.0,         0.0, 0.0, 55, 85, 3.0, 2], # jump up the step
            [0.0, -2.1, 0.0,         0.0, 0.0, 62, 80, 4.0, 2], # stabilize
            [2.0, -2.1, 0.0,         0.5, 0.0, 62, 80, 4.0, 2], # jump over terrain
            [4.0, -2.1, 0.0,         0.0, 0.0, 65, 80, 4.0, 2], # jump down ramp
            [4.0, -2.0, 0.0,         0.0, 0.0, 67, 80, 4.0, 0]]) # stop

        if salto_name == 1:
            waypts = np.array([
                [2.0, 0.0, 0.0,          0.0, 0.0, 70, 80, 4.0, 0], # initial
                [4.0, 0.0, 0.0,          0.0, 0.0, 65, 80, 4.0, 0], # go forward
                [4.0, 0.0, 0.0,          0.0, 0.0, 65, 80, 1.0, 0],
                [-2.0, 0.0, 0.0,         0.0, 0.0, 65, 80, 4.0, 2], # go fast
                [-2.0, 0.0, 0.0,         0.0, 0.0, 67, 80, 3.0, 0], # stabilize
                [-2.0, 0.0, -math.pi/2,  0.0, 0.0, 67, 80, 3.0, 0], # turn
                [-2.0, -2.1, -math.pi/2, 0.0, 0.0, 67, 80, 4.0, 0], # go across
                [-2.0, -2.1, 0.0,        0.0, 0.0, 67, 80, 3.0, 0], # turn
                [-2.0, -2.1, 0.0,        0.0, 0.0, 67, 80, 5.0, 2], # stabilize
                [0.0, -2.1, 0.0,         0.0, 0.0, 60, 85, 3.0, 2], # jump up the step
                [0.0, -2.1, 0.0,         0.0, 0.0, 67, 80, 4.0, 2], # stabilize
                [2.0, -2.1, 0.0,         0.5, 0.0, 67, 80, 4.0, 2], # jump over terrain
                [4.0, -2.1, 0.0,         0.0, 0.0, 70, 80, 4.0, 2], # jump down ramp
                [4.0, -2.0, 0.0,         0.0, 0.0, 72, 80, 4.0, 0]]) # stop
        '''

        '''
        # MAST try 1
        waypts = np.array([
            [-1.0, -0.5, math.pi,     0.0, 0.0, 65, 80, 4.0, 0], # initial
            [-3.0, -0.5, math.pi,    0.0, 0.0, 60, 80, 4.0, 0], # go back
            [-3.0, -0.5, math.pi,    0.0, 0.0, 60, 80, 1.0, 0],
            [4.0, -0.5, math.pi,     0.5, 0.0, 60, 80, .0, 2], # go fast
            [4.75, 0.25, -math.pi/2, 0.0, 0.5, 62, 80, 3.0, 0], # turning ...
            [4.0, 1.0, 0.0,          -0.5, 0.0, 62, 80, 3.0, 0], # turning ...
            [3.0, 1.0, math.pi/2,    0.0, 0.0, 62, 80, 3.0, 0], # turn at end ...
            [2.0, 1.0, math.pi,      0.0, 0.0, 62, 80, 3.0, 0], # turn at end
            [-2.0, 1.0, math.pi,     0.0, 0.0, 62, 80, 6.0, 0],
            [-2.0, 1.0, -math.pi/2,  0.0, 0.0, 62, 80, 3.0, 0], # turn
            [-2.0, -2.0, -math.pi/2, 0.0, 0.0, 62, 80, 6.0, 0], # go across
            [-2.0, -2.0, 0.0,        0.0, 0.0, 60, 80, 3.0, 0], # turn
            [-2.0, -2.0, 0.0,        0.0, 0.0, 60, 80, 5.0, 2], # stabilize
            [0.0, -2.0, 0.0,         0.0, 0.0, 55, 85, 3.0, 2], # jump up the step
            [0.0, -2.0, 0.0,         0.0, 0.0, 62, 80, 4.0, 2], # stabilize
            [2.0, -2.0, 0.0,         0.0, 0.0, 62, 80, 6.0, 2], # jump over terrain
            [2.0, -2.0, 0.0,         0.0, 0.0, 62, 80, 4.0, 2], # stabilize
            [3.5, -2.0, 0.0,         0.0, 0.0, 62, 80, 5.0, 2], # jump off the step (at 2.5)
            [4.0, -2.0, 0.0,         0.0, 0.0, 67, 80, 4.0, 0]]) # stop

        if salto_name == 1:
            for i in range(waypts.shape[0]):
                waypts[i, 5] = waypts[i, 5] + 5
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
        self.Waypoints(waypts)
        #self.Trajectory('Rectangle')
        ctrl = self.Raibert()
        # self.RaibertInspired()

        # ROBOT COMMANDS --------------------------------------------
        AngleScaling = 3667; # rad to 15b 2000deg/s integrated 1000Hz
            # 180(deg)/pi(rad) * 2**15(ticks)/2000(deg/s) * 1000(Hz) = 938734
            # 938734 / 2**8 = 3667
        LengthScaling = 256; # radians to 23.8 fixed pt radians
        CurrentScaling = 256; # radians to 23.8 fixed pt radians
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

        if self.MJ_state == 0:
            self.xbee_sending = 1

            '''
            rot_rol = quaternion_about_axis(Croll/AngleScaling,(1,1,-1))
            mat_rol = quaternion_matrix()
            rot_pit = quaternion_about_axis()
            mat_pit = quaternion_matrix()
            '''

            toSend = [ES[0], ES[1], ES[2], Cyaw, Croll, CS[0], CS[1], CS[2]]
            for i in range(8):
                if toSend[i] > 32767:
                    toSend[i] = 32767
                elif toSend[i] < -32767:
                    toSend[i] = -32767
            xb_send(0, command.INTEGRATED_VICON, pack('8h',*toSend))
            self.xbee_sending = 0

            # Printing
            #print(self.step_ind,ctrl[1],ctrl[2],ctrl[3]) # Deadbeat
            #print(self.pos[0],self.pos[1],self.pos[2])
            print(Cyaw*57/AngleScaling,Croll*57/AngleScaling,CS[0]*57/AngleScaling)
            #print(euler[0]*57,euler[1]*57,euler[2]*57)
            #print([ES[0], ES[1], ES[2], Cyaw, Croll, CS[0]])
            #print(Cyaw, Croll, CS[0], CS[1], CS[2]) # Commands
            #print(Croll,ctrl[1],ctrl[2],ctrl[3]) # cmds before scaling
            #print(np.hstack((ctrl.T, [self.acc])))
            #print(self.desx, self.desvx) # Raibert position
            #print(57*ES[0]/AngleScaling, 57*ES[1]/AngleScaling, 57*ES[2]/AngleScaling)
            #print "pos: %3.2f %3.2f %3.2f \tvel: %3.2f %3.2f \tacc: %3.2f %3.2f" % (self.desx, self.desy, self.desyaw, self.desvx, self.desvy, self.desax, self.desay)

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


        t = data.header.stamp.to_sec()
        x = np.array([data.transform.rotation.x,data.transform.rotation.y,data.transform.rotation.z,data.transform.rotation.w, data.transform.translation.x,data.transform.translation.y,data.transform.translation.z,t])
        self.tf_pub.sendTransform((x[4], x[5], x[6]), (x[0], x[1], x[2], x[3]), rospy.Time.now(), "jumper", "world")

        '''
        self.step_ind += 1
        if self.step_ind > 100:
            self.step_ind = 0

        self.R1.setServo(-self.step_ind*0.005-0.4)
        '''

    '''
    def Deadbeat(self):
        # DEADBEAT CONTROLLER ---------------------------------------
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

        return np.concatenate(([0],ctrl),1)
    '''

    '''
    def Trajectory(self, traj_name):
        # RAIBERT-INSPIRED SIMPLE HOPPING ---------------------------
        # TRAJECTORIES --------------------------
        # Forwards-backwards
        if traj_name == 'Forwards-Backwards':

            FBvel = 1.0/2.0
            FBslowdown = 1 # factor to reduce commanded velocity
            startDwell = 3.0
            offset = 0
            yoffset = 0
            yscale = 0.0
            endPt = 1.5
            if (time.time()-self.startTime) < startDwell:
                self.desx = offset
                self.desvx = 0.0

                self.desy = -yscale*endPt/2.0 + yoffset
                self.desvy = 0.0
            elif (time.time()-self.startTime-startDwell) % (2.0*endPt/FBvel) > (endPt/FBvel): # backwards
                self.desx = endPt - FBvel*((time.time()-self.startTime-startDwell)%(endPt/FBvel)) + offset
                self.desvx = -FBvel*FBslowdown
                self.desy = yscale*(endPt - FBvel*((time.time()-self.startTime-startDwell)%(endPt/FBvel)) - endPt/2.0) + yoffset
                self.desvy = yscale*self.desvx
            else: # forwards
                self.desx = FBvel*((time.time()-self.startTime-startDwell)%(endPt/FBvel)) + offset
                self.desvx = FBvel*FBslowdown
                self.desy = yscale*(FBvel*((time.time()-self.startTime-startDwell)%(endPt/FBvel)) - endPt/2.0) + yoffset
                self.desvy = yscale*self.desvx
            #print(self.desx,self.desvx,self.desy,self.desvy)
            # end Forwards-backwards
        
        elif traj_name == 'Vertical Variation':
            # Vertical variation
            if (time.time()-self.startTime) % 12.0 > 9.0:
                self.params.phase[1] = 70
            elif (time.time()-self.startTime) % 12.0 > 6.0:
                self.params.phase[1] = 80
            elif (time.time()-self.startTime) % 12.0 > 3.0:
                self.params.phase[1] = 75
            else:
                self.params.phase[1] = 80
            # end Vertical variation

        elif traj_name == 'Yaw Sweep':
            yawVel = 0.5
            yawDwell = 3.0

            yawAmp = 2.0
            if (time.time()-self.startTime) < yawDwell:
                self.desyaw = 0.0
            elif (time.time()-self.startTime-yawDwell+yawAmp/yawVel) % (4.0*yawAmp/yawVel) < (2.0*yawAmp/yawVel):
                self.desyaw = yawVel*(time.time()-self.startTime-yawDwell+yawAmp/yawVel)%(2.0*yawAmp/yawVel) - yawAmp
            else:
                self.desyaw = -yawVel*(time.time()-self.startTime-yawDwell+yawAmp/yawVel)%(2.0*yawAmp/yawVel) - yawAmp 
            print(self.desyaw)

        elif traj_name == 'Full Rotation':
            yawVel = 0.5
            yawDwell = 3.0

            if (time.time()-self.startTime) < yawDwell:
                self.desyaw = 0.0
            else:
                self.desyaw = (yawVel*(time.time()-self.startTime-yawDwell)+math.pi)%(2*math.pi) - math.pi

        elif traj_name == 'Rectangle':
            # Rectangular path
            rectPathDwell = 3.0
            x1 = -0.25 # [m] starting x
            y1 = -0.25 # [m] starting y
            x2 = 1.25 # [m] opposite corner x
            y2 = 0.75 # [m] opposite corner y
            ta = 4.0 # [s] first leg time
            tb = 4.0 # [s] second leg time
            tc = 4.0 # [s] third leg time
            td = 3.0 # [s] fourth leg time
            ya = 0.0
            yb = 3*math.pi/2#math.pi/2
            yc = 0.0#math.pi
            yd = math.pi/2#3*math.pi/2
            tturn = 3.0 # [s] turning time
            per = ta + tb + tc + td + 4*tturn
            t = (time.time()-self.startTime-rectPathDwell) % per
            if time.time()-self.startTime < rectPathDwell: # Dwell
                self.desx = x1
                self.desy = y1
                self.desvx = 0.0
                self.desvy = 0.0
                self.desyaw = ya
            elif t < ta: # First leg
                self.desx = x1 + (x2-x1)*(t/ta)
                self.desvx = (x2-x1)/ta
                self.desy = y1
                self.desvy = 0.0
                self.desyaw = ya
            elif t-ta < tturn: # First Turn
                self.desx = x2
                self.desvx = 0.0
                self.desy = y1
                self.desvy = 0.0
                if yb-ya > math.pi:
                    self.desyaw = ya + (-2*math.pi+yb-ya)*((t-ta)/tturn)
                elif yb-ya < -math.pi:
                    self.desyaw = ya + (2*math.pi+yb-ya)*((t-ta)/tturn)
                else:
                    self.desyaw = ya + (yb-ya)*((t-ta)/tturn)
            elif t-ta-tturn < tb: # Second leg
                self.desx = x2
                self.desvx = 0.0
                self.desy = y1 + (y2-y1)*((t-ta-tturn)/tb)
                self.desvy = (y2-y1)/tb
                self.desyaw = yb
            elif t-ta-tb-tturn < tturn: # Second turn
                self.desx = x2
                self.desvx = 0.0
                self.desy = y2
                self.desvy = 0.0
                if yc-yb > math.pi:
                    self.desyaw = yb + (-2*math.pi+yc-yb)*((t-ta-tb-tturn)/tturn)
                elif yc-yb < -math.pi:
                    self.desyaw = yb + (2*math.pi+yc-yb)*((t-ta-tb-tturn)/tturn)
                else:
                    self.desyaw = yb + (yc-yb)*((t-ta-tb-tturn)/tturn)
            elif t-ta-tb-2*tturn < tc: # Third leg
                self.desx = x2 + (x1-x2)*((t-ta-tb-2*tturn)/tc)
                self.desvx = (x1-x2)/tc
                self.desy = y2
                self.desvy = 0.0
                self.desyaw = yc
            elif t-ta-tb-tc-2*tturn < tturn: # Third turn
                self.desx = x1
                self.desvx = 0.0
                self.desy = y2
                self.desvy = 0.0
                if yd-yc > math.pi:
                    self.desyaw = yc + (-2*math.pi+yd-yc)*((t-ta-tb-tc-2*tturn)/tturn)
                elif yd-yc < -math.pi:
                    self.desyaw = yc + (2*math.pi+yd-yc)*((t-ta-tb-tc-2*tturn)/tturn)
                else:
                    self.desyaw = yc + (yd-yc)*((t-ta-tb-tc-2*tturn)/tturn)
            elif t-ta-tb-tc-3*tturn < td: # Fourth leg
                self.desx = x1
                self.desvx = 0.0
                self.desy = y2 + (y1-y2)*((t-ta-tb-tc-3*tturn)/td)
                self.desvy = (y1-y2)/td
                self.desyaw = yd
            else: # Fourth turn
                self.desx = x1
                self.desvx = 0.0
                self.desy = y1
                self.desvy = 0.0
                if ya-yd > math.pi:
                    self.desyaw = yd + (-2*math.pi+ya-yd)*((t-per+tturn)/tturn)
                elif ya-yd < -math.pi:
                    self.desyaw = yd + (2*math.pi+ya-yd)*((t-per+tturn)/tturn)
                else:
                    self.desyaw = yd + (ya-yd)*((t-per+tturn)/tturn)
            #print(self.desx, self.desy, self.desyaw) 
            #print(self.desvx,self.desvy)
    '''

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
            self.desvx = pts[self.ind,3]
            self.desvy = pts[self.ind,4]
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
                not(self.vel[2] < 0.5 and self.vel[2] > -0.5 and self.acc[2] < 0): 
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
        Berr = np.dot(RB,[[self.pos[0]-self.desx],[self.pos[1]-self.desy]])
        Bv = np.dot(RB,[[self.vel[0]],[self.vel[1]]])
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
