import sys
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import QApplication, QWidget,QStyle, QHBoxLayout, QVBoxLayout, QLineEdit,QMainWindow,QTabWidget,QFileSystemModel,QTreeView,QPushButton,QComboBox,QTextEdit,QLabel,QSlider,QListView
from vispy import scene
from PyQt5.QtCore import QStringListModel
from vispy import app, visuals, scene
from utils_3d import *
import cv2,os,time
import numpy as np
import pyqtgraph as pg
# import matplotlib.pyplot as plt
#############################################################################
# UI
#############################################################################

def get_image_disp_canvas(layout_image,xpix,ypix):
    canvas_2d = scene.SceneCanvas(keys='interactive', show=True)
    view_2d = canvas_2d.central_widget.add_view()
    view_2d.camera = scene.PanZoomCamera(aspect=1)
    view_2d.camera.set_range((0, xpix), (0, ypix))
    # view_2d.camera.flip = (0, 1, 0)
    camimage1 = scene.Image()
    view_2d.add(camimage1)
    layout_image.addWidget(canvas_2d.native)
    return camimage1

class button_2d(QVBoxLayout):
    def __init__(self, data_thread):
        super(button_2d, self).__init__()
        self.btn_1 = QPushButton("1.show    ")
        self.btn_2 = QPushButton("2.egobox      ")
        self.btn_3 = QPushButton("3.clear     ")
        self.btn_4 = QPushButton("4.next      ")
        self.btn_5 = QPushButton("4.last      ")
        self.line_input = QLineEdit()
        self.line_input.editingFinished.connect(self.update_file_name)
        up_layout = QHBoxLayout()
        down_layout = QHBoxLayout()
        up_layout.addWidget(self.btn_1)
        up_layout.addWidget(self.btn_2)
        up_layout.addWidget(self.btn_3)
        down_layout.addWidget(self.btn_4)
        down_layout.addWidget(self.btn_5)
        # down_layout.addWidget(self.line_input)
        self.addLayout(up_layout, stretch=1)
        self.addLayout(down_layout, stretch=1)
        self.data_thread = data_thread
        self.btn_1.clicked.connect(self.action1)
        self.btn_2.clicked.connect(self.action2)
        self.btn_3.clicked.connect(self.action3)
        self.btn_4.clicked.connect(self.action4)
        self.btn_5.clicked.connect(self.action5)
    def action1(self): self.data_thread.from_uisig_button_com.emit(1)  # save commond
    def action2(self): self.data_thread.from_uisig_button_com.emit(2)  # save commond
    def action3(self): self.data_thread.from_uisig_button_com.emit(3)  # save commond
    def action4(self): self.data_thread.from_uisig_button_com.emit(4)  # save commond
    def action5(self): self.data_thread.from_uisig_button_com.emit(5)  # save commond
    def update_file_name(self):
        self.data_thread.from_uisig_line_input_disp.emit(self.line_input.text())

class list_objs(QHBoxLayout):
    def __init__(self, data_thread):
        super(list_objs, self).__init__()
        listView = QListView()  # 创建一个listview对象
        self.slm = QStringListModel();  # 创建mode
        # self.qList = ['Item 1', 'Item 2', 'Item 3', 'Item 4']  # 添加的数组数据
        # (self.qList)  # 将数据设置到model
        listView.setModel(self.slm)  ##绑定 listView 和 model
        listView.clicked.connect(self.clickedlist)  # listview 的点击事件
        self.data_thread=data_thread
        self.data_thread.to_uisig_list_update.connect(self.update_list)
        self.addWidget(listView)
    def clickedlist(self, qModelIndex):
        #QMessageBox.information(self, "QListView", "你选择了: " + self.qList[qModelIndex.row()])
        print("点击的是：" + str(qModelIndex.row()))
        self.data_thread.from_uisig_list_select.emit(qModelIndex.row())
    def update_list(self,list):
        self.slm.setStringList(list)

class slider_2d(QSlider):
    def __init__(self, data_thread):
        super(slider_2d, self).__init__()
        self.setOrientation(Qt.Horizontal)
        self.valueChanged.connect(lambda: self.on_change_func())
        self.setTickPosition(QSlider.TicksBelow)
        self.data_thread = data_thread
        self.setStyleSheet(
            "QSlider::groove:horizontal {background:#C0C0C0;border: 1px solid; height: 5px;margin: 0px;\n}""QSlider::handle:horizontal {background-color: #33CCCC; border: 1px solid;height: 25px;width: 5px;margin: -15px -1px;}")
        # data_thread.slider_length_signal.connect(self.setMaximum)
        self.setMaximum(150)  # 最大值1ms
        self.setMinimum(0)  # 最大值1ms
        self.setValue(0)
        self.data_thread.to_uisig_slide_index.connect(self.play_mode_slider)
        self.data_thread.to_uisig_slide_range.connect(self.range_set_bydata)

    def play_mode_slider(self, position):
        self.setValue(position)

    def mousePressEvent(self, event):
        self.setValue(QStyle.sliderValueFromPosition(self.minimum(), self.maximum(), event.x(), self.width()))

    def mouseMoveEvent(self, event):
        self.setValue(QStyle.sliderValueFromPosition(self.minimum(), self.maximum(), event.x(), self.width()))

    def on_change_func(self):
        self.data_thread.from_uisig_slide_index.emit(self.value())
    def range_set_bydata(self,max_range):
        self.setMaximum(max_range)  # 最大值1ms

class FileTreeView(QVBoxLayout):
    def __init__(self, data_thread):
        super().__init__()
        # File System View
        self.model = QFileSystemModel()
        self.tree = QTreeView()
        self.tree.setModel(self.model)
        self.model.setNameFilterDisables(False)
        self.model.setNameFilters(['*.xodr', '*.xosc', '*.obj', '*.json', '*.xmap', '*.log'])
        self.model.setRootPath(os.path.expanduser('~'))
        self.tree.setRootIndex(self.model.index('./'))
        self.tree.doubleClicked.connect(self.on_file_selected)
        self.addWidget(self.tree)
        self.data_thread = data_thread
        for i in range(1, self.tree.model().columnCount()):
            self.tree.header().hideSection(i)

    def on_file_selected(self, index):
        fs_model = self.sender().model()
        file_path = fs_model.filePath(index)
        self.data_thread.from_uisig_any_filepath.emit(file_path)
class InputView(QVBoxLayout):
    def __init__(self, data_thread):
        super().__init__()
        filetree_disp = FileTreeView(data_thread)
        self.combox1 = QComboBox()
        self.combox2 = QComboBox()
        self.combox1.addItems(['3d_view', '2d_view','3d_fly'])
        self.combox2.addItems(['0.05m', '0.1m', '0.2m', '1.0m'])
        self.combox1.currentIndexChanged[int].connect(self.update_comb1)
        # self.combox2.currentIndexChanged[int].connect(self.update_comb2)
        #########################################################################
        self.text_show=QLabel()
        self.text_infor=QLabel()
        self.button_2d=button_2d(data_thread)
        self.text_show.setText('please input:frame num')
        self.text_infor.setText('init state')
        self.text_editor=QTextEdit()
        self.list_disp=list_objs(data_thread)
        self.slider=slider_2d(data_thread)
        #self.text_editor.setText('#example data:2.00,2.11 3.20,4.00')
        self.text_editor.textChanged.connect(self.text_change)
        comblayout=QVBoxLayout()
        comblayout.addLayout(filetree_disp,stretch=10)
        comblayout.addWidget(self.combox1)
        # comblayout.addWidget(self.combox2)
        comblayout.addWidget(self.text_show)
        comblayout.addWidget(self.text_editor,stretch=1)
        comblayout.addWidget(self.slider)
        comblayout.addLayout(self.button_2d)
        comblayout.addWidget(self.text_infor)
        self.addLayout(comblayout)
        self.data_thread = data_thread
        self.data_thread.to_uisig_text_infor.connect(self.updata_infor)
    def update_comb1(self, index):
        self.data_thread.from_uisig_combox1_com.emit(index)
    def update_comb2(self, index):
        self.data_thread.from_uisig_combox2_com.emit(index)
    def updata_infor(self, infor):
        self.text_infor.setText(infor)
    def text_change(self):
        input_text=self.text_editor.toPlainText()
        pts_total_num = input_text.count(',')
        changeline_total_num = input_text.count('\n')
        # print("input_text",input_text)
        # if pts_total_num==2 and changeline_total_num==1:
        if len(input_text) > 2 and input_text.count('\n'):
            self.data_thread.from_uisig_textinput_com.emit(input_text)



class data_disp(QWidget):
    from_uisig_any_filepath      = pyqtSignal(str)  # all path input from ui
    from_uisig_combox1_com       = pyqtSignal(int)
    from_uisig_combox2_com       = pyqtSignal(int)
    from_uisig_textinput_com    = pyqtSignal(str)
    # from_uisig_textinput2_com    = pyqtSignal(str)
    to_uisig_slide_index         = pyqtSignal(int)
    to_uisig_slide_range         = pyqtSignal(int)
    from_uisig_slide_index       = pyqtSignal(int)
    to_uisig_list_update         = pyqtSignal(list)
    to_uisig_text_infor          = pyqtSignal(str)
    from_uisig_line_input_disp   = pyqtSignal(str) # slide input from ui
    from_uisig_button_com        = pyqtSignal(int)  # button input from ui
    from_uisig_hmi_table_data    = pyqtSignal(np.ndarray)  # direct sendz
    def __init__(self):
        super().__init__()
        canvas = scene.SceneCanvas(keys='interactive', title='plot3d', show=True, bgcolor='papayawhip')
        self.view_3d = canvas.central_widget.add_view(camera='turntable')#3d:,fly'panzoom'
        self.view_3d.camera.distance = 25
        #self.view.camera.center = np.zeros(3)
        widgets_layout = QVBoxLayout()
        widgets_layout.addWidget(canvas.native,stretch=2)
        chart_layout = QHBoxLayout()
        img_layout = QHBoxLayout()
        both_layout= QHBoxLayout()
        both_layout.addLayout(chart_layout,stretch=2)
        both_layout.addLayout(img_layout,stretch=1)
        widgets_layout.addLayout(both_layout,stretch=1)
        self.setLayout(widgets_layout)
        self.image_pod1=get_image_disp_canvas(img_layout,400,400)

        ########### 1 chart
        win_charts = pg.GraphicsLayoutWidget(show=True)
        win_charts.setBackground('w')
        self.plot_1 = win_charts.addPlot(title="remain_distance")
        # self.plot_1.setLabel(axis='left', text='danger_dist / m')  # 靠左
        # self.plot_1.setLabel(axis='bottom', text='step')
        self.plot_1.setRange(xRange=[0,100], yRange=[0,12], padding=0)
        self.plot_2 = win_charts.addPlot(title="ego_speed")
        chart_layout.addWidget(win_charts, stretch=1)
        ##################################
        self.lane_vis = scene.visuals.Line(parent=self.view_3d.scene)
        self.lane_centerline = scene.visuals.Line(parent=self.view_3d.scene)
        self.from_uisig_any_filepath.connect(self.update_filepath)
        self.from_uisig_combox1_com.connect(self.change_viewtype)
        self.from_uisig_slide_index.connect(self.user_change_slide)
        self.from_uisig_button_com.connect(self.user_press_botton)
        self.from_uisig_textinput_com.connect(self.up_data_hmi_data)
        self.data_hmi_data = np.zeros((3, 3))
        self.init_world()
        self.cur_frame_num = 44 
        self.data_path = ''

    def init_world(self):
        tt = self.view_3d.children[0]
        [tt._remove_child(c) for c in tt.children] 
        self.data_hmi_data = np.zeros((3, 3))
        scene.visuals.XYZAxis(width=2,parent=self.view_3d.scene)
        scene.visuals.GridLines(scale=(1, 1),parent=self.view_3d.scene)
        self.global_mesh = []
        self.points_arr_dict = {}
        self.xy_range_dict={}
        self.cur_pos_dict={}
        self.slide_range=0
        self.slide_index = 0 
        self.world_2d_data= None
        self.path_div = 0
        self.dange_dist = 10
        self.to_uisig_slide_range.emit(0)
        self.plot1_x_arr=[0]
        self.plot1_y_arr=[0]
        self.plot_1.clearPlots()
        self.plot_1.clearPlots()
        self.image_mov1=np.ones((400,400,3),dtype=np.uint8)
        self.image_pod1.set_data(self.image_mov1)
        self.image_pod1.update()
        self.scatter_pt1 = scene.visuals.Markers(parent=self.view_3d.scene)
        self.scatter_pt2 = scene.visuals.Markers(parent=self.view_3d.scene)
        self.scatter_pt3 = scene.visuals.Markers(parent=self.view_3d.scene)
        self.slot_vis = scene.visuals.Line(parent=self.view_3d.scene)
        self.origin_slot_vis = scene.visuals.Line(parent=self.view_3d.scene)
        self.target_pose_vis = scene.visuals.Line(parent=self.view_3d.scene)
        self.toconnect = np.array([[0,1], [1,2], [2,3], [3,0]])

        self.grid_map_size=20#meter
        self.map_image_fac=0.05#meter
        self.map_img_size=int(self.grid_map_size/self.map_image_fac)#
        
    def up_data_hmi_data(self,text_data):
        # data_list = text_data.split(',')
        # self.data_hmi_data[0, 0] = float(data_list[0])
        # self.data_hmi_data[0, 1] = float(data_list[1])
        # self.data_hmi_data[1, 2] = float(data_list[2])
        self.cur_frame_num = int(text_data)
        self.to_uisig_text_infor.emit('frame num:' + text_data)
    def convert_ground_to_img(self,pt_arr):#pt [[x0,y0,z0],[x1,y1,z1],...]
        self.map_x0=0
        self.map_y0=0
        pt_img=np.zeros_like(pt_arr,dtype=int)
        x_m=pt_arr[:,0]-self.map_x0
        y_m=pt_arr[:,1]-self.map_y0
        pt_img[:,0]=(-y_m/self.map_image_fac +self.map_img_size/2).astype("int32")
        pt_img[:,1]=(x_m/self.map_image_fac +self.map_img_size/2).astype("int32")
        return pt_img

    def user_change_slide(self,slide_index):
        self.cur_frame_num = slide_index
        print("cur_frame_num: ",self.cur_frame_num)
        self.points_arr_dict = load_log_data(self.data_path,self.view_3d,self.global_mesh,self.cur_frame_num) 
        if len(self.points_arr_dict)>0:#data valid
            self.to_uisig_text_infor.emit('slide move to:' + str(slide_index))
            obs_point = self.points_arr_dict["obs_point"]
            self.scatter_pt1.set_data(obs_point, edge_color=(1, 0, 1, .7), face_color=(1, 1, 0, .7), size=4)
            path_point = self.points_arr_dict["path_point"]
            self.scatter_pt2.set_data(path_point, edge_color=(0, 0, 1, .7), face_color=(1, 1, 0, .7), size=4)

            odom = self.points_arr_dict["odom"]
            self.scatter_pt3.set_data(odom, edge_color=(0, 1, 1, .7), face_color=(1, 1, 0, .7), size=4)

            slot_corners = self.points_arr_dict["slot_corners"]
            self.slot_vis.set_data(pos=slot_corners, connect=self.toconnect)

            original_slot_corners = self.points_arr_dict["original_slot_corners"]
            self.origin_slot_vis.set_data(pos=original_slot_corners, connect=self.toconnect,color=(0.5,0.8,0.3))

            ####  target pose show 
            global_target_pose = self.points_arr_dict["global_target_pose"]
            if global_target_pose.shape[0] != 0:
                target_pose_arr = VehicleBox2d([global_target_pose[0][0], global_target_pose[0][1]],global_target_pose[0][2],list())
                self.target_pose_vis.set_data(pos=target_pose_arr, connect=self.toconnect,color=(1,0,0))
        else:
            self.to_uisig_text_infor.emit('not load valid log' )

    def user_press_botton(self,new_state):
        if new_state == 3:  ## clear world
            self.init_world()
            pass
        elif new_state == 4:  ## save image
            # line_arr = np.array([[1,1],[2,3]])
            # line_disp = scene.visuals.LinePlot(data=line_arr, color='green')
            # self.view_3d.add(line_disp)
            # self.global_mesh.append(line_disp)
            self.cur_frame_num += 1 
            self.user_change_slide(self.cur_frame_num)
            pass 
        elif new_state == 5: 
            self.cur_frame_num -= 1 
            self.user_change_slide(self.cur_frame_num)

        elif new_state == 2: 
            path_point = self.points_arr_dict["path_point"]
            for i in range(path_point.shape[0]): 
                self.ego_box_vis = scene.visuals.Line(parent=self.view_3d.scene)
                egobox_arr = VehicleBox2d([path_point[i][0], path_point[i][1]],path_point[i][2],list())
                self.ego_box_vis.set_data(pos=egobox_arr, connect=self.toconnect,color=(0.8,0.8,0.3))

        elif new_state == 1: 
            pass 


    def update_filepath(self, path):
        if os.path.isdir(path):
            print('fold',path)
        else:
            file_name_type = path.rsplit('/', 1)[-1]
            file_type = file_name_type.rsplit('.', 1)[1]
            # print('file_name_type',file_name_type)
            # print('file_type',file_type)
            if(file_type == 'obj'):
                vis_mesh = add_vis_obj_2022(path, self.data_hmi_data)
                self.view_3d.add(vis_mesh)
                #self.global_mesh.append(vis_mesh) #是否加入可操控状态
                pass
                #######################
            elif (file_type == 'json'):
                self.init_world()
                self.world_2d_data,self.points_arr_dict,self.xy_range_dict,max_step=load_json_plan(path,self.view_3d,self.global_mesh)
                self.slide_range=max_step-1
                self.to_uisig_slide_range.emit(self.slide_range)
                pass
            elif (file_type == 'log'):
                self.init_world() 
                self.data_path = path 
                # self.points_arr_dict = load_log_data(path,self.view_3d,self.global_mesh,self.cur_frame_num) 
                self.slide_range = 1000
                self.to_uisig_slide_range.emit(self.slide_range)        
                pass 
     
    def change_viewtype(self,type):
        if type==0:
            self.view_3d.camera = 'turntable'
        elif type==1:
            self.view_3d.camera = 'panzoom'
        else:
            self.view_3d.camera = 'fly'

class MainWindow(QMainWindow):
    def __init__(self, *args):
        # 1.init the windows
        super(MainWindow, self).__init__(*args)
        self.setWindowTitle("3D_develop_env")
        self.desktop = QApplication.desktop()
        self.resize(1400, 900)
        main_layout = QHBoxLayout()
        data_proc = data_disp()
        main_layout.addWidget(data_proc, stretch=3)
        ######################################################
        tabs_layout = QTabWidget()
        tab1, tab2, tab3 = QWidget(), QWidget(), QWidget()
        tabs_layout.addTab(tab1, "act_board"), tabs_layout.addTab(tab2, "addon"), tabs_layout.addTab(tab3, "settings")

        input_disp = InputView(data_proc)
        #tab1.setLayout(filetree_disp)
        tab1.setLayout(input_disp)
        main_layout.addWidget(tabs_layout, stretch=1)
        main_widget = QWidget()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    demo = MainWindow()
    demo.show()
    # sys.exit(vispy.app.run())
    sys.exit(app.exec_())
