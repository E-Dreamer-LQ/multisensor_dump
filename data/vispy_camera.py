import sys
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import QApplication, QWidget,QStyle, QHBoxLayout, QVBoxLayout, QLineEdit,QMainWindow,QTabWidget,QFileSystemModel,QTreeView,QPushButton,QComboBox,QTextEdit,QLabel,QSlider,QListView
from vispy import scene
from PyQt5.QtCore import QStringListModel
from vispy import app, visuals, scene
from utils_3d import *
import cv2,os,time
import numpy as np
# import matplotlib.pyplot as plt
#############################################################################
# UI
#############################################################################
class button_2d(QVBoxLayout):
    def __init__(self, data_thread):
        super(button_2d, self).__init__()
        self.btn_1 = QPushButton("1.run/stop  ")
        self.btn_2 = QPushButton("2.func      ")
        self.btn_3 = QPushButton("3.clear     ")
        self.btn_4 = QPushButton("4.save      ")
        self.line_input = QLineEdit()
        self.line_input.editingFinished.connect(self.update_file_name)
        up_layout = QHBoxLayout()
        down_layout = QHBoxLayout()
        up_layout.addWidget(self.btn_1)
        up_layout.addWidget(self.btn_2)
        up_layout.addWidget(self.btn_3)
        down_layout.addWidget(self.btn_4)
        down_layout.addWidget(self.line_input)
        self.addLayout(up_layout, stretch=1)
        self.addLayout(down_layout, stretch=1)
        self.data_thread = data_thread
        self.btn_1.clicked.connect(self.action1)
        self.btn_2.clicked.connect(self.action2)
        self.btn_3.clicked.connect(self.action3)
        self.btn_4.clicked.connect(self.action4)
    def action1(self): self.data_thread.from_uisig_button_com.emit(1)  # save commond
    def action2(self): self.data_thread.from_uisig_button_com.emit(2)  # save commond
    def action3(self): self.data_thread.from_uisig_button_com.emit(3)  # save commond
    def action4(self): self.data_thread.from_uisig_button_com.emit(4)  # save commond
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
        self.model.setNameFilters(['*.xodr', '*.xosc', '*.obj', '*.json', '*.xmap'])
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
        self.combox1.addItems(['2d_view', '3d_view','3d_fly'])
        self.combox2.addItems(['1.0m', '1.5m', '2.0m', '3.0m', '5m', '10m', '20m', '50m', '100m'])
        self.combox1.currentIndexChanged[int].connect(self.update_comb1)
        self.combox2.currentIndexChanged[int].connect(self.update_comb2)
        #########################################################################
        self.text_show=QLabel()
        self.text_infor=QLabel()
        self.button_2d=button_2d(data_thread)
        self.text_show.setText('please input:x(m),y(m),theta(deg)')
        self.text_infor.setText('init state')
        self.text_editor=QTextEdit()
        self.list_disp=list_objs(data_thread)
        self.slider=slider_2d(data_thread)
        #self.text_editor.setText('#example data:2.00,2.11 3.20,4.00')
        self.text_editor.textChanged.connect(self.text_change)
        comblayout=QVBoxLayout()
        comblayout.addLayout(filetree_disp,stretch=10)
        comblayout.addWidget(self.combox1)
        comblayout.addWidget(self.combox2)
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
        # print(input_text)
        if pts_total_num==2 and changeline_total_num==1:
            self.data_thread.from_uisig_textinput_com.emit(input_text)

def get_image_disp_canvas(layout_image):
        canvas_2d = scene.SceneCanvas(keys='interactive', show=True)
        view_2d = canvas_2d.central_widget.add_view()
        view_2d.camera = scene.PanZoomCamera(aspect=1)
        view_2d.camera.set_range((0, 512), (0, 160))
        #view_2d.camera.flip = (0, 1, 0)
        camimage1 = scene.Image()
        view_2d.add(camimage1)
        layout_image.addWidget(canvas_2d.native)
        return camimage1

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
    from_uisig_hmi_table_data    = pyqtSignal(np.ndarray)  # direct send
    def __init__(self):
        super().__init__()
        canvas = scene.SceneCanvas(keys='interactive', title='plot3d', show=True)
        self.view_3d = canvas.central_widget.add_view(camera='panzoom')#3d:turntable,fly
        #self.view.camera.center = np.zeros(3)
        widgets_layout = QVBoxLayout()
        widgets_layout.addWidget(canvas.native,stretch=3)
        layout_image = QHBoxLayout()
        widgets_layout.addLayout(layout_image,stretch=1)
        self.setLayout(widgets_layout)
        self.image_pod1=get_image_disp_canvas(layout_image)
        self.image_pod2=get_image_disp_canvas(layout_image)
        self.image_pod3=get_image_disp_canvas(layout_image)
        self.image_pod4=get_image_disp_canvas(layout_image)

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

    def init_world(self):
        self.data_hmi_data = np.zeros((3, 3))
        scene.visuals.XYZAxis(width=2,parent=self.view_3d.scene)
        scene.visuals.GridLines(scale=(1, 1),parent=self.view_3d.scene)
        self.global_mesh = []
        self.ponint_arr = []
        self.world_2d_data= None
        self.image_mov1=np.zeros((160,512,3),dtype=np.uint8)
        self.image_mov2=np.zeros((160,512,3),dtype=np.uint8)
        self.image_mov3=np.zeros((160,512,3),dtype=np.uint8)
        self.image_mov4=np.zeros((160,512,3),dtype=np.uint8)
        self.image_pod1.set_data(self.image_mov1)
        self.image_pod1.update()
        self.image_pod2.set_data(self.image_mov2)
        self.image_pod2.update()
        self.image_pod3.set_data(self.image_mov3)
        self.image_pod3.update()
        self.image_pod4.set_data(self.image_mov4)
        self.image_pod4.update()
        self.scatter_pt1 = scene.visuals.Markers(parent=self.view_3d.scene)

        #self.object_arr=
    def up_data_hmi_data(self,text_data):
        data_list = text_data.split(',')
        self.data_hmi_data[0, 0] = float(data_list[0])
        self.data_hmi_data[0, 1] = float(data_list[1])
        self.data_hmi_data[1, 2] = float(data_list[2])
        self.to_uisig_text_infor.emit('put position:' + text_data)

    def user_change_slide(self,data_index):
        self.to_uisig_text_infor.emit('slide move to:'+str(data_index))
        if data_index<len(self.ponint_arr):
            cur_pos=self.ponint_arr[data_index]# in theta
            new_local_data=np.zeros((2,3))
            new_local_data[0,0]=cur_pos[0]
            new_local_data[0,1]=cur_pos[1]
            new_local_data[1,2]=cur_pos[2]*180/math.pi
            obj_transform = hmi_data_to_transform(new_local_data)
            #self.global_mesh[1].transform = obj_transform
            t1=time.time()
            all_image_1d=get_all_cam_image(cur_pos,self.world_2d_data)
            get_image_from_nparr(self.image_mov1,all_image_1d,0,data_index)
            get_image_from_nparr(self.image_mov2,all_image_1d,1,data_index)
            get_image_from_nparr(self.image_mov3,all_image_1d,2,data_index)
            get_image_from_nparr(self.image_mov4,all_image_1d,3,data_index)

            self.image_pod1.set_data(self.image_mov1)
            self.image_pod1.update()
            self.image_pod2.set_data(self.image_mov2)
            self.image_pod2.update()
            self.image_pod3.set_data(self.image_mov3)
            self.image_pod3.update()
            self.image_pod4.set_data(self.image_mov4)
            self.image_pod4.update()
            dt = time.time() - t1
            #print('hit_dt',dt)
            for item in self.global_mesh:
                item.transform = obj_transform
        pass
    def user_press_botton(self,new_state):
        if new_state == 3:  ## clear world
            tt = self.view_3d.children[0]
            [tt._remove_child(c) for c in tt.children]
            self.init_world()
            pass
        elif new_state == 4:  ## save image
            line_arr = np.array([[1,1],[2,3]])
            line_disp = scene.visuals.LinePlot(data=line_arr, color='green')
            self.view_3d.add(line_disp)
            self.global_mesh.append(line_disp)
            cv2.imwrite('front.png',self.image_mov1)
            cv2.imwrite('rear.png',self.image_mov2)
            cv2.imwrite('left.png',self.image_mov3)
            cv2.imwrite('right.png',self.image_mov4)

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
                self.world_2d_data,self.ponint_arr=load_json_world(path,self.view_3d,self.global_mesh)
                slide_range=len(self.ponint_arr)-1
                #print(slide_range)
                self.to_uisig_slide_range.emit(slide_range)
                pass
    def change_viewtype(self,type):
        if type==0:
            self.view_3d.camera='panzoom'
        elif type==1:
            self.view_3d.camera='turntable'
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