
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils_ui import *


def sample_rays_np(H, W, f, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-W*.5+.5)/f, -(j-H*.5+.5)/f, -np.ones_like(i)], -1)
    rays_d = np.sum(dirs[..., None, :] * c2w[:3,:3], -1)
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d



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
    def __init__(self):
        super().__init__()
        main_layout = QVBoxLayout()
        self.view_3d=get_3d_disp_canvas(main_layout)
        layout_image = QHBoxLayout()
        self.image_pod1=get_image_disp_canvas(layout_image)
        self.image_pod2=get_image_disp_canvas(layout_image)
        main_layout.addLayout(layout_image, stretch=1)
        self.setLayout(main_layout)
        self.init_world()
    def init_world(self):
        scene.visuals.XYZAxis(width=2,parent=self.view_3d.scene)
        scene.visuals.GridLines(scale=(1, 1),parent=self.view_3d.scene)
        self.image_mov1=np.ones((320,480,3),dtype=np.uint8)*255
        self.image_mov2=np.ones((320,480,3),dtype=np.uint8)*255
        self.image_pod1.set_data(self.image_mov1)
        self.image_pod1.update()
        self.image_pod2.set_data(self.image_mov2)
        self.image_pod2.update()

class MainWindow(QMainWindow):
    def __init__(self, *args):
        # 1.init the windows
        super(MainWindow, self).__init__(*args)
        self.setWindowTitle("3D_develop_env")
        self.desktop = QApplication.desktop()
        self.resize(1400, 900)
        main_layout = QHBoxLayout()
        data_proc = data_disp()
        input_disp = InputView(data_proc)
        main_layout.addWidget(data_proc, stretch=3),main_layout.addLayout(input_disp, stretch=1)
        main_widget = QWidget()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    demo = MainWindow()
    demo.show()
    sys.exit(app.exec_())

#     H=320
#     W=480
#     f=1
#     c2w=np.array(([0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,1]))
#     rays_o, rays_d=sample_rays_np(H, W, f, c2w)
# canvas = scene.SceneCanvas(keys='interactive', title='plot3d', show=True)
# self.view_3d = canvas.central_widget.add_view(camera='panzoom')