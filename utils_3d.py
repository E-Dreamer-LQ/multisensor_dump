import math
import numpy as np
import configparser
from vispy.visuals.transforms import MatrixTransform
from vispy.visuals.filters import TextureFilter
import json
from vispy import app, visuals, scene

from vispy import io
from PIL import Image
import os 
from basic_f import * 

def get_polygon_center(polygon):
    sum_x=0
    sum_y=0
    sum_z=0
    for step,item in enumerate(polygon.point):
        sum_x=item.x+sum_x
        sum_y=item.x+sum_y
        sum_z=item.x+sum_z
    ave_x=sum_x/(step+1)
    ave_y=sum_y/(step+1)
    ave_z=sum_z/(step+1)
    return ave_x,ave_y,ave_z


def text_to_line_3d(text):
    last_stack_point = 0  # point already put in stack
    last_stack_connect = 0  # connect already put in stack
    pts_total_num = text.count(',')
    text_line_list = text.split('\n')
    connect_total_num=0
    single_pt_num=0
    for step, line_text in enumerate(text_line_list):
            cur_line_pts=line_text.count(',')
            if cur_line_pts>1:
                connect_total_num = cur_line_pts-1+connect_total_num
            else:
                single_pt_num+=1
    pts_np = np.zeros((pts_total_num, 3))  # point stack
    single_pts_np = np.zeros((single_pt_num, 3))  # point stack
    single_pts_index=0
    connect = np.zeros((connect_total_num, 2), dtype=int)  # point stack
    for step, line_text in enumerate(text_line_list):
            cur_line_pts = line_text.count(',')
            if cur_line_pts > 1:
                coords = [[float(coord) for coord in xy.split(",")[0:2]] for xy in line_text.split(" ")]
                pts_num = len(coords)
                for i in range(pts_num):
                    pts_np[i + last_stack_point][0] = coords[i][0]
                    pts_np[i + last_stack_point][1] = coords[i][1]
                    # pts_np[i + last_stack_point][2] = points[i].z - org_pt.z
                    if i < pts_num - 1:
                        connect[i + last_stack_connect][0] = i + last_stack_point
                        connect[i + last_stack_connect][1] = i + last_stack_point + 1
                # if step==1:
                #     break
                last_stack_point = last_stack_point + pts_num
                last_stack_connect = last_stack_connect + pts_num - 1
            elif cur_line_pts>0:
                single_pts_np[single_pts_index][0]=float(line_text.split(",")[0])
                single_pts_np[single_pts_index][1]=float(line_text.split(",")[1])
                single_pts_index+=1
    return connect,pts_np,single_pts_np


#########################################################################
#  002. coord trans
#########################################################################
# set point to R AND T
def pos_trans_by_matrix(trans_rot, trans_pos, point_pos):
    T_x = pos_to_transmatrix(trans_rot, trans_pos)
    R_mat = T_x[0:3, 0:3]
    T_mat = T_x[0:3, 3]

    new_point_pos = R_mat.dot(point_pos.T).T + T_mat
    return new_point_pos

def pos_2d_trans_by_pos(point_pos, trans_pos):
    new_point_pos = np.zeros(3)
    new_point_pos[0] =trans_pos[0]+math.cos(trans_pos[2])*point_pos[0]-math.sin(trans_pos[2])*point_pos[1]
    new_point_pos[1] =trans_pos[1]+math.sin(trans_pos[2])*point_pos[0]+math.cos(trans_pos[2])*point_pos[1]
    new_point_pos[2] =trans_pos[2]+point_pos[2]
    return new_point_pos
def points_2d_trans_by_pos(point_pos, trans_pos):
    new_points_pos = np.zeros_like(point_pos)
    new_points_pos[:,0] =trans_pos[0]+math.cos(trans_pos[2])*point_pos[:,0]-math.sin(trans_pos[2])*point_pos[:,1]
    new_points_pos[:,1] =trans_pos[1]+math.sin(trans_pos[2])*point_pos[:,0]+math.cos(trans_pos[2])*point_pos[:,1]
    return new_points_pos

def pos_to_transmatrix(R_ang, T):
    obj_transform = MatrixTransform()
    obj_transform.rotate(R_ang[0], (1, 0, 0))
    obj_transform.rotate(R_ang[1], (0, 1, 0))
    obj_transform.rotate(R_ang[2], (0, 0, 1))
    obj_transform.translate(T)
    return obj_transform.matrix.T


# hmi_data[0]:T  hmi_data[1]:R
def hmi_data_to_transform(hmi_data):
    obj_transform = MatrixTransform()
    obj_transform.rotate(hmi_data[1][0], (1, 0, 0))
    obj_transform.rotate(hmi_data[1][1], (0, 1, 0))
    obj_transform.rotate(hmi_data[1][2], (0, 0, 1))
    obj_transform.translate(hmi_data[0])
    # print(hmi_data[0])
    return obj_transform


## when ds act on trans_rot direction
def cal_ds_to_pos(cur_ds, trans_pos, trans_rot):
    new_pos = np.zeros(3)
    deg_to_rad = math.pi / 180
    new_pos[0] = trans_pos[0] + cur_ds * np.cos(trans_rot[2] * deg_to_rad)
    new_pos[1] = trans_pos[1] + cur_ds * np.sin(trans_rot[2] * deg_to_rad)
    new_pos[2] = trans_pos[2] + cur_ds * np.sin(trans_rot[1] * deg_to_rad)
    return new_pos

def cal_ds_to_pos_2d(cur_ds, x_y_theta_deg):
    new_pos = np.zeros(2)
    deg_to_rad = math.pi / 180
    new_pos[0] = x_y_theta_deg[0] + cur_ds * np.cos(x_y_theta_deg[2] * deg_to_rad)
    new_pos[1] = x_y_theta_deg[1] + cur_ds * np.sin(x_y_theta_deg[2] * deg_to_rad)
    return new_pos

##calculate the distance between point to point, point to line
def cal_dist_pos(cur_pos, last_pos):
    d_pos = cur_pos - last_pos
    dist = np.sqrt(d_pos[0] * d_pos[0] + d_pos[1] * d_pos[1] + d_pos[2] * d_pos[2])
    return dist


def sign(input):
    if input > 0.00000001:
        output = 1
    elif input < -0.00000001:
        output = -1
    else:
        output = 0
    return output


def Round_PI(theta):
    if (math.fabs(theta) > math.pi):
        theta_in = theta - sign(theta) * 2 * math.pi * (1 + math.floor((math.fabs(theta) - math.pi) / (2 * math.pi)))
    else:
        theta_in = theta
    return theta_in


def GetAverAngle(Theta1, Theta2):
    Theta1 = Round_PI(Theta1)
    Theta2 = Round_PI(Theta2)
    if (math.fabs(Theta1 - Theta2) > math.pi):
        if (Theta1 < 0):
            Theta1 = 2 * math.pi + Theta1
        else:
            Theta2 = 2 * math.pi + Theta2
    Angle = Round_PI((Theta1 + Theta2) / 2.0)
    return Angle


def Convert_Line_to_Pos(line):  # line (x0,y0,x1,y1) pos (x,y,theta)
    pos = np.zeros(3)
    pos[0] = line.pt1.x
    pos[1] = line.pt1.y
    pos[2] = Round_PI(math.atan2((line[3] - line[1]), (line[2] - line[0])))
    return pos


## CoordinadteTransfer
##INTPUT:	cd0 -  x0 y0 theta0
##			cd1 -  x1 y1 theta1
##OUTPUT:	cd -  x y theta : projection of cd1 under cd0
def CoordTransfer_2d(cd0, cd1):
    cd = np.zeros(3)
    dx = cd1[0] - cd0[0]
    dy = cd1[1] - cd0[1]
    c = math.cos(cd0[2])
    s = math.sin(cd0[2])
    theta = cd1[2] - cd0[2]
    cd[0] = dx * c + dy * s
    cd[1] = dy * c - dx * s
    theta = Round_PI(theta)
    cd[2] = theta
    return cd


def Get_Circle_EndPos(Path):  # Path (x,y,theta,ds,theta)
    x0 = Path[0], y0 = Path[1], theta0 = Path[2], ds = Path[3], Rr = Path[4]
    ftheta = 0
    EndPos = np.zeros(3)
    if (math.absf(Rr) < 1):
        EndPos[0] = math.cos(theta0) * ds + x0
        EndPos[1] = math.sin(theta0) * ds + y0
        EndPos[2] = theta0
    else:
        ftheta = -sign(Rr) * math.pi / 2 + theta0
        EndPos[0] = math.cos(ds / Rr + ftheta) * math.absf(Rr) + x0 - math.absf(Rr) * math.cos(ftheta)
        EndPos[1] = math.sin(ds / Rr + ftheta) * math.absf(Rr) + y0 - math.absf(Rr) * math.sin(ftheta)
        EndPos[2] = theta0 + ds / Rr


def PK_PointToLineDist(line, pt):  # line (x0,y0,x1,y1) pt (x,y)
    # float A,B,C;
    # float x1,y1,x2,y2;
    # float d,x,y,dist_min;
    x1 = line[0]
    y1 = line[1]
    x2 = line[2]
    y2 = line[3]
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2

    d = math.absf(A * pt[0] + B * pt[1] + C) / math.sqrt(A * A + B * B)
    x = (B * B * pt[0] - A * B * pt[1] - A * C) / (A * A + B * B)
    y = (-A * B * pt[0] + A * A * pt[1] - B * C) / (A * A + B * B)

    if ((x - x1) * (x2 - x) + (y - y1) * (y2 - y) > 0):
        dist_min = d
    else:
        dist_min = min(math.sqrt((x1 - pt[0]) * (x1 - pt[0]) + (y1 - pt[1]) * (y1 - pt[1])),
                       math.sqrt((x2 - pt[0]) * (x2 - pt[0]) + (y2 - pt[1]) * (y2 - pt[1])))
    return dist_min


#########################################################################
#  003. geo calculation
#########################################################################
def getlineCross(p1x, p1y, p2x, p2y, px, py):
    cross_state = (p2x - p1x) * (py - p1y) - (px - p1x) * (p2y - p1y)
    return cross_state


def check_in_quad(quad_pointx, quad_pointy, px, py):
    [p1x, p2x, p3x, p4x, t] = quad_pointx
    [p1y, p2y, p3y, p4y, t] = quad_pointy
    a = getlineCross(p1x, p1y, p2x, p2y, px, py) * getlineCross(p3x, p3y, p4x, p4y, px, py)
    b = a > 0
    c = getlineCross(p2x, p2y, p3x, p3y, px, py) * getlineCross(p4x, p4y, p1x, p1y, px, py)
    d = c > 0
    # print(b,'b')
    # print(d,'d')
    result = np.logical_and(b, d)
    return result


def check_in_triangle(triangle_pointx, triangle_pointy, px, py):
    [p1x, p2x, p3x] = triangle_pointx
    [p1y, p2y, p3y] = triangle_pointy
    a = getlineCross(p1x, p1y, p2x, p2y, px, py) * getlineCross(p2x, p2y, p3x, p3y, px, py)
    result = a > 0
    return result


def check_out_invalid(invalid_points, pinit_cloud, eff_range):
    ir_num = invalid_points.shape[0]
    result = np.ones((pinit_cloud.shape[0]), dtype=bool)
    for i in range(ir_num):
        dxyz = np.fabs(pinit_cloud - invalid_points[i, :])
        dist = dxyz[:, 0] + dxyz[:, 1] + dxyz[:, 2]
        cur_b = dist > eff_range
        result = np.logical_and(result, cur_b)
    return result


#########################################################################
#  004. ini read
#########################################################################

def get_transPos_from_ini(path):
    config = configparser.ConfigParser()
    config.read(path)

    trans_x = float(config['trans']['trans_x'])
    trans_y = float(config['trans']['trans_y'])
    trans_z = float(config['trans']['trans_z'])

    return trans_x, trans_y, trans_z


def get_transRot_from_ini(path):
    config = configparser.ConfigParser()
    config.read(path)

    trans_rot_x = float(config['trans_rot']['rot_x'])
    trans_rot_y = float(config['trans_rot']['rot_y'])
    trans_rot_z = float(config['trans_rot']['rot_z'])
    return trans_rot_x, trans_rot_y, trans_rot_z


def get_fold_from_ini(path):
    config = configparser.ConfigParser()
    config.read(path)
    result = config['data_path']['path']
    return result

def load_json_plan(path,scene_3d,global_mesh):
    with open(path,'r') as f:
        json_obj=json.loads(f.read())
    #第一个就是自车
    max_step=0
    total_pos_arr={}
    xy_range_dict = {}
    time_step =json_obj['world_para']['time_step']# 0.1s
    for item in json_obj['objects']['object_arr']:
        [x0, y0, theta_deg] = item['star_pos']
        new_local_data = np.zeros((2, 3))
        new_local_data[0, 0] = x0
        new_local_data[0, 1] = y0
        new_local_data[1, 2] = theta_deg #deg
        path='./my_lib/'+item['type']+'.obj'
        car_mesh,xy_range = add_vis_obj_2022(path, new_local_data)
        scene_3d.add(car_mesh)
        global_mesh.append(car_mesh)
        theta0 = theta_deg * math.pi / 180
        object_pos_arr=[]
        add_last_ds=0

        for step,res in enumerate(item['ds_arr']) :
            ds   =item['ds_arr'][step]
            rou  =item['rou_arr'][step]
            speed  =item['speed_arr'][step]
            Position_arr=Draw_trajectory(x0, y0, theta0, ds, rou ,add_last_ds,speed,time_step)
            add_last_ds=add_last_ds+ds
            if step==0:
                object_pos_arr=Position_arr
            else:
                object_pos_arr=np.vstack((object_pos_arr,Position_arr))
            #######display traj
            [x0, y0, theta0] = [Position_arr[-1][0],Position_arr[-1][1],Position_arr[-1][2]]
            line_arr = Position_arr[:, 0:2]
            line_disp = scene.visuals.LinePlot(data=line_arr,marker_size=1)
            scene_3d.add(line_disp)


            # object_color=item['object_color']
            # point_len=len(item['object_x'])
            # point_arr=np.zeros((point_len,3))
            # point_arr[:,0]=item['object_x']
            # point_arr[:,1]=item['object_y']
            # vis_mesh = scene.visuals.Polygon(pos=point_arr, color=object_color)
            # scene_3d.add(vis_mesh)
            max_step=max(len(object_pos_arr),max_step)
        total_pos_arr[item['object_id']]=object_pos_arr
        xy_range_dict[item['object_id']]=xy_range
    return json_obj,total_pos_arr,xy_range_dict,max_step

def load_log_data(path,scene_3d,global_mesh,cur_frame_num):
    points_arr_dict = {}
    is_planner_ready = False
    is_frame_ready = False

    if(os.path.exists(path)):
        log_file = open(path,'r')
        log_file_line = log_file.readline() 
        max_frame_num = 1000

        origin_slot_corners_x = []
        origin_slot_corners_y = []
        slot_corners_x = []
        slot_corners_y = []

        target_pose_x = []
        target_pose_y = []
        target_pose_phi = []

        start_pose_x = []
        start_pose_y = []
        start_pose_phi = []

        sw_pose_x = []
        sw_pose_y = []
        sw_pose_theat = []

        T_lines_x = []
        T_lines_y = []

        obs_point_x = []
        obs_point_y = []
        

        obs_line_x = []
        obs_line_y = []

        path_x = []
        path_y = []
        path_phi = []
        
        odom_x = []
        odom_y = []
        odom_phi = []

        while log_file_line:
            if log_file_line.find("========== frame_num: " + str(cur_frame_num)) != -1:
                is_frame_ready = True
                log_file_line = log_file.readline()
                continue
            if log_file_line.find("ENTER Perpendicular PLAN") != -1:
                is_planner_ready = True
                log_file_line = log_file.readline()
                continue
            if is_frame_ready and is_planner_ready:
                if log_file_line.find("odo_slot_corners:") != -1:
                    line_vec = log_file_line.split()
                    odo_slot_corners_x.append(float(line_vec[-4]))
                    odo_slot_corners_y.append(float(line_vec[-2]))

                if log_file_line.find("odo_bottom_slot_corners:") != -1:
                    line_vec = log_file_line.split()
                    odo_bottom_slot_corners_x.append(float(line_vec[-4]))
                    odo_bottom_slot_corners_y.append(float(line_vec[-2]))

                if log_file_line.find("original_slot_corners:") != -1:
                    line_vec = log_file_line.split()
                    origin_slot_corners_x.append(float(line_vec[-4]))
                    origin_slot_corners_y.append(float(line_vec[-2]))

                if log_file_line.find("[PERP] slot_corners:") != -1:
                    line_vec = log_file_line.split()
                    slot_corners_x.append(float(line_vec[-4]))
                    slot_corners_y.append(float(line_vec[-2]))

                if log_file_line.find("PERP: global_target_pose") != -1:
                    line_vec = log_file_line.split()
                    target_pose_x.append(float(line_vec[-4][:-2]))
                    target_pose_y.append(float(line_vec[-2][:-8]))
                    target_pose_phi.append(float(line_vec[-1]))

                if log_file_line.find("safe_sweet_pose:") != -1:
                    line_vec = log_file_line.split()
                    sw_pose_x.append(float(line_vec[-6]))
                    sw_pose_y.append(float(line_vec[-4]))
                    sw_pose_theat.append(float(line_vec[-2]))

                if log_file_line.find("[PERP] T_lines:") != -1:
                    line_vec = log_file_line.split()
                    T_lines_x.append(float(line_vec[-4]))
                    T_lines_y.append(float(line_vec[-2]))
                    T_lines_x.append(float(line_vec[-10]))
                    T_lines_y.append(float(line_vec[-8]))

                if log_file_line.find("obs_point:") != -1:
                    line_vec = log_file_line.split()
                    obs_point_x.append(float(line_vec[-4]))
                    obs_point_y.append(float(line_vec[-2]))

                if log_file_line.find("obs_line:") != -1:
                    line_vec = log_file_line.split()
                    obs_line_x.append(float(line_vec[-4]))
                    obs_line_y.append(float(line_vec[-2]))
                    obs_line_x.append(float(line_vec[-10]))
                    obs_line_y.append(float(line_vec[-8]))

                if log_file_line.find("path_point:") != -1:
                    line_vec = log_file_line.split()
                    path_x.append(float(line_vec[-6]))
                    path_y.append(float(line_vec[-4]))
                    path_phi.append(float(line_vec[-2]))
                
                if log_file_line.find("receive odom data") != -1:
                    line_vec = log_file_line.split()
                    odom_x.append(float(line_vec[-6][:-1]))
                    odom_y.append(float(line_vec[-3][:-1]))
                    odom_phi.append(float(line_vec[-1]))

                if log_file_line.find("PERP: init_pose") != -1:
                    line_vec = log_file_line.split(":")
                    start_pose_x.append(float(line_vec[-3].split(",")[0]))
                    start_pose_y.append(float(line_vec[-2].split(",")[0]))
                    start_pose_phi.append(float(line_vec[-1]))

            log_file_line = log_file.readline()

        log_file.close()

        points_arr_dict["obs_point"] = np.concatenate((np.array(obs_point_x).reshape(-1,1),np.array(obs_point_y).reshape(-1,1),np.zeros(len(obs_point_x)).reshape(-1,1)),axis = 1)
        points_arr_dict["path_point"] = np.concatenate((np.array(path_x).reshape(-1,1),np.array(path_y).reshape(-1,1), np.zeros(len(path_x)).reshape(-1,1)),axis = 1)

        points_arr_dict["odom"] = np.concatenate((np.array(odom_x).reshape(-1,1),np.array(odom_y).reshape(-1,1), np.array(odom_phi).reshape(-1,1)),axis = 1)
        
        points_arr_dict["T_line"] = np.concatenate((np.array(T_lines_x).reshape(-1,1),np.array(T_lines_y).reshape(-1,1)),axis = 1)

        points_arr_dict["obs_line"] = np.concatenate((np.array(obs_line_x).reshape(-1,1),np.array(obs_line_y).reshape(-1,1)),axis = 1)

        points_arr_dict["slot_corners"] = np.concatenate((np.array(slot_corners_x).reshape(-1,1),np.array(slot_corners_y).reshape(-1,1)),axis = 1)

        points_arr_dict["original_slot_corners"] = np.concatenate((np.array(origin_slot_corners_x).reshape(-1,1),np.array(origin_slot_corners_y).reshape(-1,1)),axis = 1)

        points_arr_dict["global_target_pose"] = np.concatenate((np.array(target_pose_x).reshape(-1,1),np.array(target_pose_y).reshape(-1,1),np.array(target_pose_phi).reshape(-1,1)),axis = 1)
        

    return points_arr_dict


def Add(list1, list2):
	list3 = []
	for i in range(len(list1)):
		list3.append(list1[i] + list2[i])
	return list3
        

def load_json_world(path,scene_3d,global_mesh):
    with open(path,'r') as f:
        json_obj=json.loads(f.read())
    [x0, y0, theta0] = json_obj['ego_car']['star_pos']
    total_pos_arr=[]
    for step,item in enumerate(json_obj['ego_car']['ds_arr']) :
        ds            =json_obj['ego_car']['ds_arr'][step]
        rou            =json_obj['ego_car']['rou_arr'][step]
        Position_arr=Draw_trajectory(x0, y0, theta0, ds, rou)
        if step==0:
            total_pos_arr=Position_arr
        else:
            total_pos_arr=np.vstack((total_pos_arr,Position_arr))
        #######display traj
        [x0, y0, theta0] = [Position_arr[-1][0],Position_arr[-1][1],Position_arr[-1][2]]
        line_arr = Position_arr[:, 0:2]
        line_disp = scene.visuals.LinePlot(data=line_arr)
        scene_3d.add(line_disp)
    new_local_data = np.zeros((2, 3))
    new_local_data[0, 0] = total_pos_arr[0,0]
    new_local_data[0, 1] = total_pos_arr[0,1]
    new_local_data[1, 2] = total_pos_arr[0,2] * 180 / math.pi #to deg
    path='./my_lib/small_car_1108.obj'
    car_mesh = add_vis_obj_2022(path, new_local_data)
    scene_3d.add(car_mesh)
    global_mesh.append(car_mesh)

    for item in json_obj['sensor']['camera_arr']:
        camera_name=item['camera_name']
        star_pos = item['star_pos']
        FOV = item['FOV']
        cam_line=draw_2d_camera(star_pos,FOV)
        cam_line_disp = scene.visuals.Polygon(pos=cam_line,color='white')
        global_mesh.append(cam_line_disp)
        scene_3d.add(cam_line_disp)
        #print(star_pos,'star_pos')
        # pos=[0,0,0]
        # end_pos=cal_ds_to_pos_2d(10,pos)
        # ray_arr = np.zeros((2,2))
        # ray_arr[0]=pos[0:2]
        # ray_arr[1]=end_pos[0:2]
        # print('ray_arr',ray_arr)
        # ray_disp = scene.visuals.Line(pos=ray_arr,color='white')
        # scene_3d.add(ray_disp)
        # global_mesh.append(ray_disp) #test ray hit polygon

    for item in json_obj['objects']['object_arr']:
        object_color=item['object_color']
        point_len=len(item['object_x'])
        point_arr=np.zeros((point_len,3))
        point_arr[:,0]=item['object_x']
        point_arr[:,1]=item['object_y']
        vis_mesh = scene.visuals.Polygon(pos=point_arr, color=object_color)
        scene_3d.add(vis_mesh)
    return json_obj,total_pos_arr

def Get_traj_point_num_ds(ds,dsmin):
    #dsmin=0.1
    max_point_num=500
    floor_ds = int(abs(ds/dsmin))
    remian_ds = ds - floor_ds * dsmin
    if (remian_ds < 0.005):
        point_num = floor_ds
    else:
        point_num = 1 + floor_ds
    if (abs(ds) <= abs(dsmin)):
        point_num = 1;
    if (point_num > max_point_num):
        point_num = max_point_num;
        dsmin = abs(ds / max_point_num)
    return point_num,dsmin

def Draw_trajectory(x0,y0,theta0,ds,rou,add_last_ds,speed,time_step):
    dsmin=speed*time_step
    point_num, dsmin=Get_traj_point_num_ds(ds,dsmin)
    #print(dsmin,point_num)

    Position_arr=np.zeros((point_num,4))
    if (abs(rou) < 0.01):
        for i in range(point_num-1):
            Position_arr[i][0]=(i+1) * math.cos(theta0) * dsmin * sign(ds)+x0;
            Position_arr[i][1]=(i+1) * math.sin(theta0) * dsmin * sign(ds)+y0;
            Position_arr[i][2]=theta0;
            Position_arr[i][3]=(i+1) * dsmin * sign(ds)+add_last_ds;
        Position_arr[point_num-1][0]=math.cos(theta0) * ds+x0;
        Position_arr[point_num-1][1]=math.sin(theta0) * ds+y0;
        Position_arr[point_num-1][2]=theta0;
        Position_arr[point_num-1][3]=ds+add_last_ds;
    else:
        # ftheta, 为x,y相对Xr，Yr的角度, dth > 0:left
        ftheta =   -sign(rou)*math.pi / 2 + theta0
        Xr = x0 -  math.cos(ftheta)/abs(rou)
        Yr = y0 -  math. sin(ftheta)/abs(rou)
        #print(ftheta,Xr,Yr)
        step_th = abs(dsmin *rou) * sign(ds * rou);
        for i in range(point_num-1):
            Position_arr[i][0]=math.cos((i+1) * step_th+ftheta) / abs(rou)+Xr;
            Position_arr[i][1]=math.sin((i+1) * step_th+ftheta) / abs(rou)+Yr;
            Position_arr[i][2]=(i+1) * step_th+theta0;
            Position_arr[i][3]=(i+1) * dsmin * sign(ds)+add_last_ds;
        Position_arr[point_num-1][0]=math.cos(ds*rou+ftheta)/abs(rou)+Xr;
        Position_arr[point_num-1][1]=math.sin(ds*rou+ftheta)/abs(rou)+Yr;
        Position_arr[point_num-1][2]=theta0+ds*rou;
        Position_arr[point_num-1][3]=ds+add_last_ds;
    #np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
    #print(Position_arr)
    return Position_arr


#########################################################################
#  007.  read mesh in fold
#########################################################################

dict_vert = {}
dict_face = {}
dict_texure={}
dict_texcoords={}
dict_pil_texure={}
def get_mesh_path(mesh_path):
    global dict_vert,dict_face,dict_texure,dict_texcoords
    file_name_type = mesh_path.rsplit('/', 1)[-1]
    #file_type = file_name_type.rsplit('.', 1)[1]
    obj_name= file_name_type.rsplit('.', 1)[0]
    #obj_name=
    if obj_name in dict_vert:# if already load mesh
        vertices      = dict_vert[obj_name]
        faces      = dict_face[obj_name]
        texcoords = dict_texcoords[obj_name]
        texture   = dict_texure[obj_name]
        pil_texure = dict_pil_texure[obj_name]
    else:
        mesh_path= mesh_path
        texture_path=mesh_path.split(".obj")[0]+'.jpg'
        #print(mesh_path,texture_path)
        vertices, faces, normals, texcoords = io.read_mesh(mesh_path)
        texture = np.flipud(io.imread(texture_path))
        pil_texure = Image.open(texture_path)
        dict_vert[obj_name]=vertices
        dict_face[obj_name]=faces
        dict_texcoords[obj_name] =texcoords
        dict_texure[obj_name] =texture
        dict_pil_texure[obj_name]=pil_texure

    return vertices,faces,texcoords,texture,pil_texure

def check_mesh_in_path(cur_pos,xy_range,ego_path,ego_index):
    cur_danger_dist=-1
    for step,ego_pos in enumerate(ego_path):
        if step>=ego_index:

            rx,ry=get_rxrx_from_point(cur_pos,ego_pos) #将自车点投影到障碍物mesh 上
            #print(step, ego_index, rx, ry,ego_pos)
            if xy_range[0]<rx<xy_range[1] and xy_range[2]<ry<xy_range[3]:
                cur_danger_dist=ego_pos[3]
                #print(cur_danger_dist,rx,ry)
                break
    return cur_danger_dist


def add_vis_obj_2022(mesh_path,hmi_data):
    vertices,faces,texcoords,texture,img_pil=get_mesh_path(mesh_path)
    print(type(vertices))
    xrange_max=max(vertices[:,0])
    xrange_min=min(vertices[:,0])
    yrange_max=max(vertices[:,1])
    yrange_min=min(vertices[:,1])
    xy_range=[xrange_min,xrange_max,yrange_min,yrange_max]
    print(xrange_max,'xrange_max')
    print(yrange_max,'yrange_max')
    #############################################################
    vis_mesh = scene.Mesh(vertices, faces, color='white')
    obj_transform = MatrixTransform()
    obj_transform.rotate(hmi_data[1][0], (1, 0, 0))
    obj_transform.rotate(hmi_data[1][1], (0, 1, 0))
    obj_transform.rotate(hmi_data[1][2], (0, 0, 1))
    obj_transform.translate(hmi_data[0])
    vis_mesh.transform=obj_transform
    texture_filter = TextureFilter(texture, texcoords)
    vis_mesh.attach(texture_filter)
    return vis_mesh,xy_range

def draw_2d_camera(star_pos,FOV):
    cam_line = np.zeros((3,3))
    cam_line[0,:2]=star_pos[0:2]
    new_pos=np.zeros(3)
    new_pos=star_pos.copy()#!!!!!!!!!!! deep copy
    new_pos[2]=star_pos[2]+FOV/2
    #print(star_pos)
    cam_line[1,:2]=cal_ds_to_pos_2d(1,new_pos)
 #   new_pos = star_pos.copy()
    new_pos[2] = star_pos[2] - FOV/2
    #print(star_pos)
    cam_line[2,:2]=cal_ds_to_pos_2d(1,new_pos)
    return cam_line
def get_four_camera_pose(car_pos_rad,json_obj):
    sensor_num     = len(json_obj['sensor']['camera_arr'])
    camera_pos_arr_rad = np.zeros((sensor_num,3))
    FOV_arr=np.zeros(sensor_num)
    DIM_arr=np.zeros(sensor_num)
    for index,item in enumerate(json_obj['sensor']['camera_arr']):
        camera_name=item['camera_name']
        cam_pos_rad = np.array(item['star_pos'])
        cam_pos_rad[2]=cam_pos_rad[2]*math.pi/180
        FOV_arr[index] = item['FOV']
        DIM_arr[index] = item['DIM']
        camera_pos_arr_rad[index]=pos_2d_trans_by_pos(cam_pos_rad,car_pos_rad)
    return camera_pos_arr_rad,FOV_arr,DIM_arr
def get_1d_img_from_camera(camera_pos_rad,FOV,DIM,json_obj):#pos rad fov deg
    image_1d     = np.zeros(DIM)
    #print(star_pos,'star_pos')
    cur_cam_pos    = camera_pos_rad.copy()
    base_theta = cur_cam_pos[2]+FOV/2*math.pi/180
    theta_slice=FOV / DIM * math.pi / 180
    for pix_index in range(DIM):
        hit_flag=0
        hit_dist=1000
        hit_color=0
        cur_cam_pos[2]=base_theta-theta_slice*pix_index
        for item in json_obj['objects']['object_arr']:
            color_id=item['color_id']
            point_len=len(item['object_x'])
            point_arr=np.zeros((point_len,3))
            point_arr[:,0]=item['object_x']
            point_arr[:,1]=item['object_y']
            temp_hit_flag,temp_hit_dist=line_hit_polygon(cur_cam_pos,point_arr)
            if temp_hit_flag>0:
                hit_flag=1
                if temp_hit_dist<hit_dist:
                    hit_dist=temp_hit_dist
                    hit_color=color_id
        image_1d[pix_index]=hit_color
    return image_1d

def get_all_cam_image(star_pos,json_obj):
    camera_pos_arr_rad,FOV_arr,DIM_arr=get_four_camera_pose(star_pos,json_obj)
    sensor_num =int(len(FOV_arr))
    DIM=int(DIM_arr[0])
    all_image_1d = np.zeros((sensor_num,DIM))
    cam_index=1
    for cam_index in range(sensor_num):
        all_image_1d[cam_index]=get_1d_img_from_camera(camera_pos_arr_rad[cam_index],FOV_arr[cam_index],DIM,json_obj)
    return all_image_1d

def get_rxrx_from_point(pos,point_arr):#one point
    dx = point_arr[0] - pos[0]
    dy = point_arr[1] - pos[1]
    theta=pos[2]#
    c = math.cos(theta)
    s = math.sin(theta)
    rx=dx * c + dy * s
    ry=dy * c - dx * s
    return rx,ry

def get_rxrx_from_points(pos,point_arr):#points
    dx = point_arr[:,0] - pos[0]
    dy = point_arr[:,1] - pos[1]
    theta=pos[2]#
    c = math.cos(theta)
    s = math.sin(theta)
    rx=dx * c + dy * s
    ry=dy * c - dx * s
    return rx,ry
def project_2d_points(pos,points_arr):#points
    data_len=len(points_arr)
    points_2d=np.zeros((data_len,2))
    dx = points_arr[:,0] - pos[0]
    dy = points_arr[:,1] - pos[1]
    theta=pos[2]#
    c = math.cos(theta)
    s = math.sin(theta)
    points_2d[:,0]=dx * c + dy * s
    points_2d[:,1]=dy * c - dx * s
    return points_2d

def line_hit_polygon(pos,point_arr):
    rx,ry=get_rxrx_from_points(pos,point_arr)
    hit_flag=0 # if ray hit polygon
    hit_dist=0
    dist_to_pos=rx.min()
    # print(rx, ry, 'rx,ry')
    # print(pos, 'pos')
    # print( point_arr, ' point_arr,')
    if dist_to_pos>0:

        if ry.min()*ry.max()<0:
            hit_flag=1
            hit_dist=dist_to_pos
    return hit_flag,hit_dist

def get_image_from_nparr(image_mov1,all_image_1d,cam_index,data_index):
    #color_dict=dict({0:(0,0,0),1:(0,0,255),2:(0,255,0),3:(255,0,0),4:(0,255,255)})
    color_dict=dict({0:(0,0,0),1:(255,0,0),2:(0,255,0),3:(0,0,255),4:(255,255,0)})
    #color_dict=dict({1:255,2:255,3:0,4:1})

    arr_org=all_image_1d[cam_index]
    arr_after=np.array(np.vectorize(color_dict.get)(arr_org)).T
    image_mov1[data_index]=arr_after
    #print(arr_after,'arr_after')
    #"color_id": 1, "object_color": "red",
    #"color_id": 2, "object_color": "green",
    #"color_id": 3, "object_color": "blue",
    #"color_id": 4, "object_color": "yellow",
    pass
if __name__ == '__main__':
    # json_path='world_2d_1st.json'
    # load_json_world(json_path)
    # pos=[0,0,0]
    # new_pos=cal_ds_to_pos_2d(50,pos)
    # print(new_pos)
    pos = [ -0.48834333 , -0.22460437 ,181.1349257 ]
    point_arr=np.array([[ 1., -3. , 0.]])
    rx,ry=get_rxrx_from_points(pos,point_arr)
    print(rx,ry,'rx,ry')
    pass