from vispy import app, visuals, scene
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import QApplication, QWidget,QStyle, QHBoxLayout, QVBoxLayout, QLineEdit,QMainWindow,QTabWidget,QFileSystemModel,QTreeView,QPushButton,QComboBox,QTextEdit,QLabel,QSlider,QListView
from PyQt5.QtCore import QStringListModel
import os,sys

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



class FileTreeView(QVBoxLayout):
    def __init__(self, data_thread):
        super().__init__()
        # File System View
        self.model = QFileSystemModel()
        self.tree = QTreeView()
        self.tree.setModel(self.model)
        self.model.setNameFilterDisables(False)
        self.model.setNameFilters(['*.xodr', '*.xosc', '*.obj', '*.json', '*.npz'])
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


def get_image_disp_canvas(layout_image):
    canvas_2d = scene.SceneCanvas(keys='interactive', show=True)
    view_2d = canvas_2d.central_widget.add_view(camera='panzoom')
    view_2d.camera.set_range((0, 480), (0, 320))
    # view_2d.camera.flip = (0, 1, 0)
    camimage = scene.Image()
    view_2d.add(camimage)
    layout_image.addWidget(canvas_2d.native)
    return camimage

def get_3d_disp_canvas(layout_3d):
    canvas = scene.SceneCanvas(keys='interactive', show=True)
    view_3d = canvas.central_widget.add_view(camera='turntable')
    layout_3d.addWidget(canvas.native)
    return view_3d

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
        if pts_total_num==2 and changeline_total_num==1:
            self.data_thread.from_uisig_textinput_com.emit(input_text)