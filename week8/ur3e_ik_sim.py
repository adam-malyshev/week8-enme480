import rclpy
from rclpy.node import Node

from ur3e_mrc.msg import CommandUR3e

import sys
import numpy as np
class InverseKinematicsUR3e(Node):

    def __init__(self, args): 
        super().__init__("fk_ur3e_pub")

        self.publisher_ = self.create_publisher(CommandUR3e, '/ur3/command', 10)

        args = rclpy.utilities.remove_ros_args(args)
        xWgrip, yWgrip, zWgrip, yawWgrip = tuple([float(x) for x in args[1:5]])
        #timer_period = 0.5  # seconds                                                   
        #self.timer = self.create_timer(timer_period, self.move_robot)
        self.move_robot(xWgrip, yWgrip, zWgrip, yawWgrip)

    def to_rad(self, value):
        return value * np.pi/180

    def move_robot(self, xWgrip, yWgrip, zWgrip, yawWgrip):
        q = self.inverse_kinematics(xWgrip, yWgrip, zWgrip, yawWgrip) 
        msg = CommandUR3e(destination = list(q), v= 1.0,a = 1.0,io_0 = False)
        self.publisher_.publish(msg)
        
        self.get_logger().info(f'q:{np.rad2deg(q).astype(int).tolist()}')
        T = self.calculate_fk_from_dh(q)
        self.get_logger().info(f'\n{np.array_str(T, precision=3, suppress_small=True)}')


    def calculate_fk_from_dh(self,q):
        # L1 = 0.152
        # L2 = 0.120
        # L3 = 0.244
        # L4 = 0.093
        # L5 = 0.213
        # L6 = 0.104
        # L7 = 0.083
        # L8 = 0.092
        # L9 = 0.0535
        # L10 = 0.059
       
        # A1 = self.get_a_matrix(0,    L1,     q[0],          -np.pi/2) 
        # A2 = self.get_a_matrix(0,    L2,     q[1]-np.pi/2,  -np.pi/2) 
        # A3 = self.get_a_matrix(0,    L3,     0,             -np.pi/2) 
        # A4 = self.get_a_matrix(0,    L4,     q[2],          np.pi/2) 
        # A5 = self.get_a_matrix(0,    L5,     0,             np.pi/2) 
        # A6 = self.get_a_matrix(0,    L6,     q[3],          -np.pi/2) 
        # A7 = self.get_a_matrix(0,    L7,     q[4],          np.pi/2) 
        # A8 = self.get_a_matrix(0,    L8,     q[5]+np.pi/2,  np.pi/2) 
        # A9 = self.get_a_matrix(0,    L9,     0,             np.pi/2) 
        # A10 = self.get_a_matrix(0,   -L10,    np.pi/2,       0) 
        
        # #compute T
        # T = A1@A2@A3@A4@A5@A6@A7@A8@A9@A10

        # return T
        A0 = np.array([[1, 0, 0, -0.15], [0, 1, 0, 0.15], [0, 0, 1, 0.01], [0, 0, 0, 1]])
        A1 = self.get_a_matrix(0, -np.pi/2, 0.15185, q[0])
        A2 = self.get_a_matrix(0.24355, 0, 0, q[1])
        A3 = self.get_a_matrix(0.2132, 0, 0, q[2])
        A4 = self.get_a_matrix(0, np.pi/2, 0.13105, q[3] + np.pi/2)
        A5 = self.get_a_matrix(0, -np.pi/2, 0.08535, q[4])
        A6 = self.get_a_matrix(0, 0, 0.0921, q[5])
        A7 = self.get_a_matrix(0.0535, 0, 0.052, np.pi)
        return A0@A1@A2@A3@A4@A5@A6@A7



    def get_a_matrix(self, r, alpha, d, theta):
        return  np.array([
            [np.cos(theta), -np.sin(theta)*np.cos(alpha),   np.sin(theta)*np.sin(alpha),    r*np.cos(theta)],
            [np.sin(theta), np.cos(theta)*np.cos(alpha),    -np.cos(theta)*np.sin(alpha),   r*np.sin(theta)], 
            [0          ,   np.sin(alpha)              ,    np.cos(alpha),                  d],
            [0          ,   0                           ,   0                           ,   1]
        ])
    
    def rot_z(self, theta):
        return np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0,              0,             1]
        ])

    def inverse_kinematics(self, xWgrip, yWgrip, zWgrip, yawWgrip):

        # TODO: Function that calculates an elbow up 
        # inverse kinematics solution for the UR3
        
        #all lengths in meters
        #all angles in radians

        L1 = 0.152
        L2 = 0.120
        L3 = 0.244
        L4 = 0.093
        L5 = 0.213
        L6 = 0.104
        L7 = 0.083
        L8 = 0.092
        L9 = 0.0535
        L10 = 0.059

        # Step 1: find gripper position relative to the base of UR3,
        # and set theta_5 equal to -pi/2
        
        xgrip = xWgrip + 0.15
        ygrip = yWgrip - 0.15
        zgrip = zWgrip - 0.01

        yawgrip = yawWgrip*np.pi/180

        theta_5 = -np.pi/2

        
        # Step 2: find x_cen, y_cen, z_cen
        

        z_cen = zgrip
        x_cen = xgrip - L9*np.cos(yawgrip)
        y_cen = ygrip - L9*np.sin(yawgrip)


        # Step 3: find theta_1
        
        #beta is the angle from the x axis to the vector (x_cen, y_cen)        
        beta = np.arctan2(y_cen, x_cen)

        dy = L2 - L4 + L6
        r = np.sqrt(x_cen**2 + y_cen**2)

        #alpha is the angle from the x axis to vector when theta_1 is 0
        alpha = np.arcsin(dy/r)
        print(alpha*180/np.pi)
        theta_1 = beta - alpha


        # Step 4: find theta_6 

        #yaw + theta_6 = theta_1 + pi/2

        theta_6 = theta_1 + np.pi/2 - yawgrip


        # Step 5: find x3_end, y3_end, z3_end

        R01 = self.rot_z(theta_1)

        cen1 = R01.T @ np.array([x_cen, y_cen, z_cen]).T
        
        _3_end1 = np.array([cen1[0] - L7, cen1[1] - L6 - 0.027, cen1[2] + L8 + 0.052])
        
        _3_end0 = R01 @ _3_end1.T

        x3_end = _3_end0[0]
        y3_end = _3_end0[1]
        z3_end = _3_end0[2]


        # Step 6: find theta_2, theta_3, theta_4

        # theta_3 - theta_2 = theta_4

        # theta_2, theta_3 can be found same way as two link robot in text
        # where "y" is z3_end - L1 and "x" is r = sqrt(x3_end**2 + y3_end**2)
        

        r = np.sqrt(x3_end**2+y3_end**2)
        
        s = z3_end - L1


        D = (r**2 + s**2 - L3**2 - L5**2)/(2*L3*L5)
        theta_3 = np.arctan2(-np.sqrt(1-D**2), D)
        theta_2 = np.arctan2(s, r) - np.arctan2(L5*np.sin(theta_3), L3+L5*np.cos(theta_3))

        
        theta_3 = -theta_3
        theta_2 = -theta_2

        theta_4 = -(np.abs(theta_3) - np.abs(theta_2))
        
        # Return the set of joint angles to move the robot
        return theta_1, theta_2, theta_3, theta_4, theta_5, theta_6

def main(args=None):
    rclpy.init(args=args)

    ik_ur3e_pub = InverseKinematicsUR3e(args)
    rclpy.spin_once(ik_ur3e_pub)

    ik_ur3e_pub.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    if len(sys.argv) < 5:
        print("usage: ur3e_fk xWgrip yWgrip zWgrip yawWgrip")
    else:
        main(sys.argv)
