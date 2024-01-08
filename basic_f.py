# import matplotlib.pyplot as plt
import numpy as np
import os
import string
import math
from collections import deque

def Dist(x0, y0, x1, y1):
	return math.sqrt((x1 - x0) * (x1 - x0) +
									 (y1 - y0) * (y1 - y0))

def Add(list1, list2):
	list3 = []
	for i in range(len(list1)):
		list3.append(list1[i] + list2[i])
	return list3

def Multi(list0, num):
	list1 = []
	for i in range(len(list0)):
		list1.append(list0[i] * num)
	return list1

def InnerProd(list1, list2):
	list3 = []
	for i in range(len(list1)):
		list3.append(list1[i] * list2[i])
	return list3

def CreateUnitVec2d(phi):
	return [math.cos(phi), math.sin(phi)]

def PoseDirection(x, y, phi, length):
	root = [x, y]
	leaf = Add(root, Multi(CreateUnitVec2d(phi), length))
	return root + leaf

def RearToCenter(x, y, phi, length):
	root = [x, y]
	return Add(root, Multi(CreateUnitVec2d(phi), length))



def Box2d(center, heading, length, width):
	corners_x = []
	corners_y = []
	half_length = length * 0.5
	half_width = width * 0.5
	dx1 = math.cos(heading) * half_length
	dy1 = math.sin(heading) * half_length
	dx2 = math.sin(heading) * half_width
	dy2 = -math.cos(heading) * half_width

	corners_x.append(center[0] + dx1 + dx2)
	corners_x.append(center[0] + dx1 - dx2)
	corners_x.append(center[0] - dx1 - dx2)
	corners_x.append(center[0] - dx1 + dx2)
	corners_y.append(center[1] + dy1 + dy2)
	corners_y.append(center[1] + dy1 - dy2)
	corners_y.append(center[1] - dy1 - dy2)
	corners_y.append(center[1] - dy1 + dy2)
	corners_x.append(corners_x[0])
	corners_y.append(corners_y[0])

	corners = np.concatenate((np.array(corners_x).reshape(-1,1),np.array(corners_y).reshape(-1,1)),axis = 1)

	return corners

front_edge_to_center = 3.795
back_edge_to_center = 0.985
vehicle_length = 4.780
vehicle_width = 1.890
wheel_base = 2.92
length_rear_to_center = front_edge_to_center - wheel_base * 0.5

extend_ratio = 0.5
extend_length = 0.3 * extend_ratio
extend_width = 0.185 * extend_ratio

# center: vehicle rear x, y
def VehicleBox2d(center, heading, expansion):
	if not expansion:
		# print("empty expansion input")
		expansion.append(0.0)
		expansion.append(0.0)

	d_length = vehicle_length * 0.5 - back_edge_to_center
	geo_center = \
			Add(center, Multi(CreateUnitVec2d(heading), d_length))

	return Box2d(geo_center, heading, \
			vehicle_length + 2.0 * expansion[1], \
			vehicle_width + 2.0 * expansion[0])
