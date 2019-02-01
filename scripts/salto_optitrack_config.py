from tf.transformations import *

# APG
'''
offsets1 = euler_matrix(-0.03, 0.04, -0.02, 'rxyz') # Santa
offsets2 = euler_matrix(0.01, 0.0, 0.0,'rxyz') # Rudolph
offsets3 = euler_matrix(-0.04, -0.01, 0.06,'rxyz') # Dasher
'''

# Cory 391
#'''
offsets1 = euler_matrix(-0.04, 0.00, 0.00, 'rxyz') # Santa
offsets2 = euler_matrix(0.01, -0.01, 0.0,'rxyz') # Rudolph
offsets3 = euler_matrix(0.015, 0.005, 0.00,'rxyz') # Dasher with shell
#offsets3 = euler_matrix(0.025, 0.01, 0.00,'rxyz') # Dasher
#offsets3 = euler_matrix(0.00, 0.00, 0.00,'rxyz') # Dasher unladen
#'''

# if the robot stabilizes too far to the left, decrease roll.