#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import time
import math
import datetime
from multiprocessing import Process
from multiprocessing import Queue
import os
import serial
import threading
import datetime
import struct
import time
import cython

__version__ = '1.02'
#2021-0409 have to make it more precision:jamin
#2021-0409 night set better factor
data_bytes = list()
read_period = 0.01
time_base = datetime.datetime.now().timestamp()
lock_data_buffer_write = threading.Lock()
lock_data_buffer_unpack = threading.Lock()
cont_read_imu = True
thread_imu_count = 0
unpack_count = 0
past_rcount = 0


class SerialPort:
    def __init__(self, port, baud):
        self.port = serial.Serial(port, baud)
        self.port.close()
        self.__is_new = False
        self.__data = []
        if not self.port.isOpen():
            self.port.open()

    def port_open(self):
        if not self.port.isOpen():
            self.port.open()

    def port_close(self):
        self.port.close()

    def port_send(self, send_data):
        self.port.write(send_data)

    def port_read(self):
        self.__is_new = False
        count = self.port.inWaiting()
        if count > 0:
            self.__data = []
            self.__data = self.port.read(count)
            self.__is_new = True
        return self.__is_new

    def reset_new(self):
        self.__is_new = False

    def get_data(self):
        return self.__data


class IMUPacket:
    def __init__(self, gx, gy, gz, ax, ay, az, pack_count, timestamp, rsvd, check):  # unit in deg and m/s2

        self.gx = gx
        self.gy = gy
        self.gz = gz
        self.ax = ax
        self.ay = ay
        self.az = az
        self.pack_count = pack_count
        self.timestamp = timestamp
        self.rsvd = rsvd
        self.check = check


    def print_data(self):
        pass

def thread_period_readimu(serclass, period, queue):
    global time_base
    global data_bytes
    global cont_read_imu
    global thread_imu_count
    is_new_data = serclass.port_read()
    thread_imu_count = thread_imu_count + 1
    if cont_read_imu:
        t = threading.Timer(period, thread_period_readimu, [serclass, period, queue])
        t.start()
    else:
        serclass.port_close()
    if is_new_data:
        serclass.reset_new()
        lock_data_buffer_write.acquire()
        try:
            x = list(serclass.get_data())
            data_bytes = data_bytes + x
        finally:
            lock_data_buffer_write.release()
        time_offset = datetime.datetime.now().timestamp() - time_base
        lock_data_buffer_write.acquire()
        data_bytes = data_unpack(time_offset, time_base, data_bytes, queue)
        lock_data_buffer_write.release()


def data_unpack(time_delta, timebase, buf, queue):
    last_head = 0
    len_data = len(buf)
    n = 2
    heads = []
    while n < len_data:
        if buf[n - 2] == 189 and buf[n - 1] == 219 and buf[n] == 10:
            heads.append(n - 2)
            n = n + 1
        else:
            n = n + 1
    for head in heads:
        if len_data - head >= 34:
            pack = buf[head:(head + 34)]
            imu_data = unpack_imu(pack, time_delta, head, len_data, timebase)
            if imu_data.check:
                queue.put(imu_data)
            last_head = head
    return buf[(last_head + 34):]


def combine_u32_2f(bl):  # bl in list 4 little endian
    a = bytearray(bl)
    return struct.unpack('<f', a)[0]


def combine_u16_2uint(bl):
    a = bytearray(bl)
    return struct.unpack('<H', a)[0]


def combine_u32_2uint(bl):
    a = bytearray(bl)
    return struct.unpack('<I', a)[0]


# transverse coordinate according to imu orientation
def trans_imu(imu_out):
    imu_trans = IMUPacket(imu_out.gx, imu_out.gy, imu_out.gz, imu_out.ax, imu_out.ay, imu_out.az,
                          imu_out.pack_count, imu_out.timestamp, imu_out.rsvd, imu_out.check)
    return imu_trans


def unpack_imu(packed, time_delta, head, len_data, timebase):
    global unpack_count
    global past_rcount

    # check xor
    n = 0
    xor_check = 0
    check = False
    while n < 33:
        xor_check = xor_check ^ packed[n]
        n = n + 1
    if xor_check == packed[33]:
        check = True
        # start unpack
        data_gx = packed[3:7]
        data_gy = packed[7:11]
        data_gz = packed[11:15]
        data_ax = packed[15:19]
        data_ay = packed[19:23]
        data_az = packed[23:27]
        data_rsvd = packed[27:31]
        pack_count = packed[31:33]
        gyro_x = combine_u32_2f(data_gx)
        gyro_y = combine_u32_2f(data_gy)
        gyro_z = combine_u32_2f(data_gz)
        acc_x = combine_u32_2f(data_ax)
        acc_y = combine_u32_2f(data_ay)
        acc_z = combine_u32_2f(data_az)
        reserved = combine_u32_2uint(data_rsvd)
        packet_count = combine_u16_2uint(pack_count)
        head_pos = len_data - head - 1
        tshift = head_pos / 34
        dt=time_delta - tshift * 0.005
        timestamp = timebase +dt
        imu_origin = IMUPacket(gyro_x, gyro_y, gyro_z, acc_x, acc_y, acc_z, packet_count, timestamp, reserved, check)
        imu_data = trans_imu(imu_origin)
        unpack_count = unpack_count + 1
    else:
        imu_data = IMUPacket(0, 0, 0, 0, 0, 0, 0, 0, 0, check)
    return imu_data
####################################################################################################
init_bias = []
deg2rad = np.pi / 180
rad2deg = 180 / np.pi
Qv = np.array([[1e-8, 0, 0, -5e-15, 0, 0],
               [0, 1e-8, 0, 0, -5e-15, 0],
               [0, 0, 1e-8, 0, 0, -5e-15],
               [-5e-15, 0, 0, 1e-10, 0, 0],
               [0, -5e-15, 0, 0, 1e-10, 0],
               [0, 0, -5e-15, 0, 0, 1e-10]], dtype=np.float64)
Ra0 = np.array([[1e-4, 0, 0],
                [0, 1e-4, 0],
                [0, 0, 1e-4]], dtype=np.float64)
dt = 0.005
queue_timeout=0.02
Q = np.array([0, 0, 0, 1], dtype=np.float64).reshape(4, 1)
r = np.array([0, 0, 0], dtype=np.float64).reshape(3, 1)
gravity = np.array([0, 0, -9.80665], dtype=np.float64).reshape(3, 1)
gravity_z = 9.80665
b = np.array([0, 0, 0], dtype=np.float64).reshape(3, 1)
state_x = np.array([0, 0, 0, 0, 0, 0], dtype=np.float64).reshape(6, 1)
P = np.zeros((6, 6), dtype=np.float64)
first_run = True
z_buffer = []
U = np.zeros((4, 4), dtype=np.float64)

def gyro_static_null(buf):
    offset_spec = 0.55
    gx_sum,gy_sum,gz_sum,ax_sum,ay_sum,az_sum = 0,0,0,0,0,0
    n = 0
    for e in buf:
        gx_sum, gy_sum, gz_sum, ax_sum, ay_sum, az_sum = gx_sum+e.gx, gy_sum+e.gy, gz_sum+e.gz, ax_sum+e.ax, ay_sum+e.ay, az_sum+e.az
        n = n + 1
    mean_ax = ax_sum / float(n)
    mean_ay = ay_sum / float(n)
    mean_az = az_sum / float(n)
    acc_general = math.sqrt(mean_ax ** 2 + mean_ay ** 2 + mean_az ** 2)
    g_bias = [gx_sum / float(n), gy_sum / float(n), gz_sum / float(n), acc_general, mean_ax, mean_ay, mean_az]

    if abs(g_bias[0]) > offset_spec or abs(g_bias[1] > offset_spec) or abs(g_bias[2]) > offset_spec:
        g_bias = [0, 0, 0, 9.80665, mean_ax, mean_ay, mean_az]
    return g_bias

class AttPacket:  # unit in rad
    def __init__(self, roll, pitch, yaw, quat, imu):
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw
        self.Q = quat
        self.imu = imu

def att_ekf(imu):
    global ekf_count,b,state_x,Qv,Ra0,dt,Q,r,gravity,gravity_z,P,first_run,z_buffer,U
    state_x[5] = 0
    b = b + state_x[3:6].reshape(3, 1)
    we = np.array([imu.gx * deg2rad - b[0], imu.gy * deg2rad - b[1], imu.gz * deg2rad - b[2]],dtype=np.float64).reshape(3, 1)
    omega = np.zeros((4, 4), dtype=np.float64)
    omega[0, 1] = we[2]
    omega[0, 2] = -we[1]
    omega[0, 3] = we[0]
    omega[1, 0] = -we[2]
    omega[1, 2] = we[0]
    omega[1, 3] = we[1]
    omega[2, 0] = we[1]
    omega[2, 1] = -we[0]
    omega[2, 3] = we[2]
    omega[3, 0] = -we[0]
    omega[3, 1] = -we[1]
    omega[3, 2] = -we[2]
    if first_run:
        first_run = False
        U = 0.5 * omega
        Q = Q + dt * np.dot(U, Q)
    else:
        Un = U
        Qn = Q
        U = 0.5 * omega
        Um = 0.5 * (U + Un)
        s1 = np.dot(Un, Qn)
        s2 = np.dot(Um, (Qn + 0.5 * dt * s1))
        s3 = np.dot(Um, (Qn + 0.5 * dt * s2))
        s4 = np.dot(U, (Qn + dt * s3))
        Q = Qn + 1 / 6 * (s1 + 2 * s2 + 2 * s3 + s4) * dt
    Q = Q / (math.sqrt(Q[0] ** 2 + Q[1] ** 2 + Q[2] ** 2 + Q[3] ** 2))
    dq = 0.5 * np.array([state_x[0], state_x[1], state_x[2]], dtype=np.float64).reshape(3, 1)
    dot_dq = np.dot(dq.reshape(1, 3), dq)
    if dot_dq <= 1:
        Qe = np.array([dq[0], dq[1], dq[2], math.sqrt(1 - dot_dq)], dtype=np.float64)
    else:
        sqdq = 1 / (math.sqrt(1 + dot_dq))
        Qe = sqdq * np.array([dq[0], dq[1], dq[2], 1])
    QeX = np.array([[Qe[3], Qe[2], -Qe[1], Qe[0]],
                    [-Qe[2], Qe[3], Qe[0], Qe[1]],
                    [Qe[1], -Qe[0], Qe[3], Qe[2]],
                    [-Qe[0], -Qe[1], -Qe[2], Qe[3]]], dtype=np.float64)
    Q = np.dot(QeX, Q)
    Q3 = np.array([Q[0], Q[1], Q[2]], dtype=np.float64).reshape(3, 1)
    Q3X = np.array([[0, -Q3[2], Q3[1]],
                    [Q3[2], 0, -Q3[0]],
                    [-Q3[1], Q3[0], 0]], dtype=np.float64)
    Cnb = (2 * Q[3] ** 2 - 1) * np.identity(3) - 2 * Q[3] * Q3X + 2 * np.dot(Q3, Q3.reshape(1, 3))
    acc = np.array([imu.ax, imu.ay, imu.az], dtype=np.float64).reshape(3, 1)
    acc = acc / gravity_z
    Ak = np.zeros((6, 6), dtype=np.float64)
    Ak[0, 1] = we[2]
    Ak[0, 2] = -we[1]
    Ak[0, 3] = -1
    Ak[1, 0] = -we[2]
    Ak[1, 2] = we[0]
    Ak[1, 4] = -1
    Ak[2, 0] = we[1]
    Ak[2, 1] = -we[0]
    Ak[2, 5] = -1
    F = np.identity(6) + dt * Ak
    ae = np.dot(Cnb, gravity / gravity_z)
    r = acc - ae
    a3 = abs(math.sqrt(acc[0] ** 2 + acc[1] ** 2 + acc[2] ** 2) - 1)
    b1 = math.sqrt(we[0] ** 2 + we[1] ** 2 + we[2] ** 2)
    r_abs = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
    if a3 < 0.08:
        za = 37.5 * a3
    else:
        za = 3
    if b1 < 0.6:
        zb = 5 * b1
    else:
        zb = 3
    if a3 < 0.0175 and b1 < 0.052:
        if r_abs < 0.2:
            zc = 6 * r_abs
        else:
            zc = 1.2
    else:
        if r_abs < 0.05:
            zc = 24 * r_abs
        else:
            zc = 1.2
    z = max(za, zb, zc)
    z_buffer.append(z)
    if len(z_buffer) > 50:
        z_buffer = z_buffer[1:]
    zmax = max(z_buffer)
    Ra_coef = 100 ** zmax
    Racc = Ra_coef * Ra0
    Ha = np.zeros((3, 6), dtype=np.float64)
    Ha[0, 1] = -ae[2]
    Ha[0, 2] = ae[1]
    Ha[1, 0] = ae[2]
    Ha[1, 2] = -ae[0]
    Ha[2, 0] = -ae[1]
    Ha[2, 1] = ae[0]
    tmp1 = np.dot(F, P)
    P = np.dot(tmp1, F.T) + Qv
    tmp2 = np.dot(P, Ha.T)
    tmp3 = np.dot(Ha, P)
    tmp8 = np.dot(tmp3, Ha.T) + Racc
    tmp4 = np.linalg.inv(tmp8)
    K = np.dot(tmp2, tmp4)
    tmp4 = np.identity(6) - np.dot(K, Ha)
    tmp5 = np.dot(tmp4, P)
    tmp6 = np.dot(tmp5, tmp4.T)
    tmp7 = np.dot(K, Racc)
    P = tmp6 + np.dot(tmp7, K.T)
    state_x = np.dot(K, r)
    Cbn = Cnb.T
    T11 = Cbn[0, 0]
    T21 = Cbn[1, 0]
    T31 = Cbn[2, 0]
    T32 = Cbn[2, 1]
    T33 = Cbn[2, 2]
    pitch = np.arcsin(-T31)
    roll = 0  # as temporary occupying value
    yaw = 0  # as temporary occupying value
    if T33 != 0:
        roll = np.arctan(T32 / T33)
    if T11 != 0:
        yaw = np.arctan(T21 / T11)
    if T33 < 0:
        if roll < 0:
            roll = roll + np.pi
        else:
            roll = roll - np.pi
    if T11 < 0:
        if T21 > 0:
            yaw = yaw + np.pi
        else:
            yaw = yaw - np.pi
    attitude = AttPacket(roll, pitch, yaw, [Q[0], Q[1], Q[2], Q[3]], imu)
    return attitude


def init_states(bias):
    norm_g = bias[3]
    ay = bias[5]
    ax = bias[4]
    az = bias[6]
    iroll = -np.arcsin(ay / norm_g)
    ipitch = np.arcsin(ax / norm_g)
    iyaw = 0
    if az > 0:
        if ay > 0:
            iroll = -iroll - np.pi
        else:
            iroll = -iroll + np.pi
    q = np.array([0, 0, 0, 1], dtype=np.float64).T
    cr = np.cos(iroll / 2)
    sr = np.sin(iroll / 2)
    cp = np.cos(ipitch / 2)
    sp = np.sin(ipitch / 2)
    cy = np.cos(iyaw / 2)
    sy = np.sin(iyaw / 2)
    q[3] = cr * cp * cy + sr * sp * sy
    q[0] = sr * cp * cy - cr * sp * sy
    q[1] = cr * sp * cy + sr * cp * sy
    q[2] = cr * cp * sy - sr * sp * cy
    return q


def imu_att_proc(log, queue, imu_queue):
    global init_bias
    global Q
    global b
    mSerial = SerialPort('/dev/ttyUSB0', 115200)
    thread_period_readimu(mSerial, read_period, imu_queue)
    num = 800
    ibuf = []
    while not imu_queue.empty():
        ibuf.append(imu_queue.get(block=True, timeout=queue_timeout))
    lb = len(ibuf)
    while lb - num < 0:
        time.sleep((num - lb) * dt + 0.1)
        while not imu_queue.empty():
            ibuf.append(imu_queue.get(block=True, timeout=queue_timeout))
        lb = len(ibuf)
    init_bias = gyro_static_null(ibuf)
    Q = init_states(init_bias)
    b = np.array([x*deg2rad for x in init_bias[0:3]], dtype=np.float64).reshape(3, 1)
    time.sleep(queue_timeout)
    while True:
        if imu_queue.qsize() > 4:
            buf = []
            for n in range(4):
                buf.append(imu_queue.get(block=True, timeout=queue_timeout))
            for imu in buf:
                att = att_ekf(imu)
                queue.put(att, block=True, timeout=queue_timeout)
                if queue.qsize() > 2:
                    queue.get(True)
        else:
            time.sleep(0.002)


if __name__ == '__main__':
    imu_att_queue = Queue()
    imu_data_queue = Queue()
    p_imu_att = Process(target=imu_att_proc, args=(False, imu_att_queue, imu_data_queue))
    p_imu_att.start()
    start_time=time.time()
    while True:
        while not imu_att_queue.empty():
            d = imu_att_queue.get(True)
            print('timestamp:%.3f   R: %6.2f P :%6.2f Y: %6.2f' % (time.time()-start_time, d.roll * rad2deg, d.pitch * rad2deg, d.yaw * rad2deg))
            print(imu_att_queue.qsize(), imu_data_queue.qsize())
            time.sleep(1)
