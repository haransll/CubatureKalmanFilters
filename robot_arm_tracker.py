"""

Author: Haran Arasaratnam
Date: Jun 15 2019

Reference:
https://github.com/zziz/kalman-filter

"""

import numpy as np
import matplotlib.pyplot as plt

from numpy import cos, sin, diag, eye, ones, zeros, dot, kron, isscalar, outer, vstack, hstack
# TODO: replace numpy Guassian with scipy
# from scipy.stats import multivariate_normal
from numpy.random import multivariate_normal
from scipy import linalg

def fx(target):

    target.x[0,0] = target.j

    if (-1)**target.count == -1:
        target.x[0,1] = target.k
    else:
        target.x[0,1] = 2*np.pi - target.k

    target.t += 1
    target.k += target.int_k
    if target.k > target.finish_k:
        target.k = np.pi/2
        target.j += target.int_j
        target.count += 1

    return target.x

def hx(target):
    '''

    :param target:
    :return:
    '''

    return np.array([target.r1 * cos(target.x[0,0]) - target.r2 * cos(target.x[0,0] + target.x[0,1]),
                     target.r1 * sin(target.x[0,0]) - target.r2 * sin(target.x[0,0] + target.x[0,1])
                     ])


def fx_approx(x_k):

    '''

    fx: process model/state transition

    x: state vector at time step k
        numpy array of dim_x

    return:  state vector at time step (k+1)

    '''

    return np.array([ x_k[0,0] + x_k[0,1],
                      x_k[0,0] * x_k[0,1]
                     ])

def hx_approx(x_k):
    '''

    :param x_k:
    :return:
    '''

    len_effector_1 = 0.8  # r1
    len_effector_2 = 0.2  # r2

    # TODO: rewrite as a matrix mux
    return np.array([len_effector_1*cos(x_k[0]) - len_effector_2*cos(x_k[0] + x_k[1]),
                     len_effector_1*sin(x_k[0]) - len_effector_2*sin(x_k[0] + x_k[1])
    ])


class Target(object):

    def __init__(self, x=None, fx=None):
        
        self.x = x

        self.fx = fx

        self.Q = diag([0.0001, 0.01])

        self.r1 = 0.8
        self.r2 = 0.2

        self.j = 0.3
        self.int_j =  0.1
        self.finish_j = 1.2

        self.k = np.pi/2
        self.int_k = 0.05
        self.finish_k = 1.5*np.pi

        self.t = 1
        self.count = 1


    def update_target_state(self, dt):

        process_noise = multivariate_normal(mean=zeros(2), cov=self.Q, size=1)
        
        return self.fx(self) + process_noise

class Sensor(object):

    def __init__(self, hx=None):

        '''

        :param target:
        :param R:
        :param h:
        '''

        # TODO: should I consider hx as a property of sensor class?
        self.hx = hx

        self.R = 0.005*eye(2)


    def get_sensor_measurement(self, target, dt):

        sensor_noise = multivariate_normal(mean=zeros(2), cov=self.R, size=1)

        return self.hx(target) + sensor_noise


class CKF(object):

    def __init__(self, x_kk=None, P_kk=None, Q_approx = None, R_approx=None):

        self.x_kk = x_kk
        self.P_kk = P_kk

        self.Q_approx = Q_approx
        self.R_approx = R_approx


        dim_x = P_kk.shape[0]

        # create a set of normalized cubature points and weights given the state vector dim.
        self.num_points = (2 * dim_x)
        self.cubature_points = np.concatenate((np.eye(dim_x), -1 * np.eye(dim_x)), axis=1)
        self.cubature_weights = kron(ones((1, self.num_points)), 1/self.num_points)


    def predict(self, x_kk, P_kk):

        # TODO: Can you use fx_approx?
        #x_kk1 = fx_approx(self.x_kk)
        x_kk1 = x_kk
        P_kk1 = P_kk + self.Q_approx

        self.x_kk = x_kk1
        self.P_kk = P_kk1

    def update(self, x_kk1, P_kk1, z_k):

        # calculate Xi_k from x_kk1 and P_kk1

        P_kk1 = 0.5*(P_kk1 + P_kk1.T)

        # TODO: Apply chol. and QR
        U, S, Vdash = linalg.svd(P_kk1, full_matrices=True)

        S_kk1 = 0.5 * dot((U + Vdash.T), np.sqrt(diag(S)))

        # matrix to hold cubature points
        Xi = kron(ones((1, self.num_points)), x_kk1) + dot(S_kk1, self.cubature_points)

        # pass Xi_k through the measurement function to obtain z_kk, Pzz_kk,and Pxz_kk
        Zi = [ hx_approx(Xi[:,i]) for i in range(self.num_points)]

        # convert a list of np.array to np.matrix
        Zi = vstack(Zi).T

        # predicted meas
        z_kk1 = vstack(Zi.sum(axis=1)/self.num_points)
           
        # W = np.diag([self.num_points]*len(self.num_points))    

        X_ = (Xi - kron(ones((1,self.num_points)), x_kk1))/(self.num_points**0.5)

        Z_ = (Zi - kron(ones((1,self.num_points)), z_kk1))/(self.num_points**0.5)

        # innovation cov
        # Generic way for cubature rule with unequal weights
        # Pzz = Z_.dot(W).dot(Z_.T)
        
        Pzz = dot(Z_, Z_.T) + self.R_approx

        # cross cov
        Pxz = dot(X_, Z_.T)

        # CKF gain
        G = dot(Pxz, linalg.pinv(Pzz))

        # update
        self.x_kk = x_kk1 + dot(G, (z_k - z_kk1))

        self.P_kk = P_kk1 - G.dot(Pzz).dot(G.T)

def run():

    # generate measurements from sensor
    dt = 1
    time_stamps = np.arange(0, 630 + dt, dt)

    x0 = np.array([[0., 0.]]) # theta_1 and theta_2
    target = Target(x=x0, fx=fx)

    sensor = Sensor(hx=hx)

    ground_truth, measurements = [], []

    for k in time_stamps:

        z_k = sensor.get_sensor_measurement(target, dt)

        target.x = target.update_target_state(dt)

        measurements.append(z_k)

        ground_truth.append(target.x)

        # print(target.t)

    measurements = vstack(measurements).T
    ground_truth = vstack(ground_truth).T

    # plt.subplot(211)
    # plt.plot(time_stamps, measurements[0,:])
    # plt.title('x')
    #
    # plt.subplot(212)
    # plt.plot(time_stamps, measurements[1,:])
    # plt.title('y')
    #
    # plt.show()


    # tracking states (theta1 and theta2)

    std_q1 = 0.01
    std_q2 = 0.1
    Q_approx = diag([std_q1 ** 2, std_q2 ** 2])

    R_approx = 0.005 * eye(2)

    x_kk = np.array([[0.7, np.pi]]).T
    P_kk = diag([0.81, 0.25])
    ckf = CKF(x_kk=x_kk, P_kk=P_kk, Q_approx=Q_approx, R_approx=R_approx)

    x_filter_array = []

    # TODO: for loop through numpy matrix
    for k in time_stamps:

        z_k = vstack(measurements[:, k])

        ckf.predict(ckf.x_kk, ckf.P_kk)

        ckf.update(ckf.x_kk, ckf.P_kk, z_k)

        x_filter_array.append(ckf.x_kk)

    # convert a list of np.vector to np.matrix
    x_filter_array = np.hstack(x_filter_array)

    plt.subplot(211)
    plt.plot(time_stamps, x_filter_array[0,:], 'b--', label='Estimate')
    plt.plot(time_stamps, ground_truth[0,:], 'r',label='ground truth')
    plt.grid()
    plt.legend()
    plt.ylabel('theta_1')

    plt.subplot(212)
    plt.plot(time_stamps, x_filter_array[1,:], 'b--')
    plt.plot(time_stamps, ground_truth[1,:],'r')
    plt.ylabel('theta_2')
    plt.grid()
    plt.show()

if __name__ == '__main__':

    run()