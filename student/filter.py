# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Kalman filter class
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# imports
import numpy as np

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import misc.params as params

class Filter:
    '''Kalman filter class'''
    def __init__(self):
        self.dim_state = params.dim_state # 6
        self.dt = params.dt
        self.q = params.q

    def F(self):
        ############
        # TODO Step 1: implement and return system matrix F
        ############
        dt = self.dt
        F = np.matrix(np.eye(6))
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt

        return F

        ############
        # END student code
        ############

    def Q(self):
        ############
        # TODO Step 1: implement and return process noise covariance Q
        ############

        dt = self.dt
        q = self.q
        q3 = q * np.power(dt, 3) / 3
        q2 = q * np.power(dt, 2) / 2
        q1 = q * dt

        return np.matrix([[q3, 0, 0, q2, 0, 0],
                          [0, q3, 0, 0, q2, 0],
                          [0, 0, q3, 0, 0, q2],
                          [q2, 0, 0, q1, 0, 0],
                          [0, q2, 0, 0, q1, 0],
                          [0, 0, q2, 0, 0, q1]])

        ############
        # END student code
        ############

    def predict(self, track):
        ############
        # TODO Step 1: predict state x and estimation error covariance P to next timestep, save x and P in track
        ############

        F = self.F()
        x = F * track.x
        P = F * track.P * F.transpose() + self.Q()

        track.set_x(x)
        track.set_P(P)

        ############
        # END student code
        ############

    def update(self, track, meas):
        ############
        # TODO Step 1: update state x and covariance P with associated measurement, save x and P in track
        ############

        # Kalman update function need result is Gaussian distribution,
        # so using Jacobian H, but calculate gamma we can just use h(x).
        gamma = self.gamma(track, meas)
        H = meas.sensor.get_H(track.x) # Jacobian H
        S = self.S(track, meas, H) # covariance of residual
        K = track.P * H.transpose() * np.linalg.inv(S) # Kalman gain

        x = track.x + K * gamma # state update
        I = np.identity(self.dim_state)
        P = (I - K * H) * track.P # covariance update

        track.set_x(x)
        track.set_P(P)

        ############
        # END student code
        ############
        track.update_attributes(meas)

    def gamma(self, track, meas):
        ############
        # TODO Step 1: calculate and return residual gamma
        ############

        return meas.z - meas.sensor.get_hx(track.x)

        ############
        # END student code
        ############

    def S(self, track, meas, H):
        ############
        # TODO Step 1: calculate and return covariance of residual S
        ############

        return H * track.P * H.transpose() + meas.R

        ############
        # END student code
        ############
