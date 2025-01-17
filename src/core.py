import numpy as np
import dearpygui.dearpygui as dpg
import callbacks
import torch
import os
import torch.nn as nn
from PIL import ImageOps
import utils
import tags
class bio_image_vgg_classification_net(nn.Module):
    def __init__(self, class_num: int = 5, dropout_ratio: float = 0.1):
        super(bio_image_vgg_classification_net, self).__init__()
        self.class_num = 5 # 5 types of bird droplet + 1 background type
        self.conv1 = nn.Conv2d(
            in_channels=6, out_channels=96, kernel_size=7, stride=2, padding=0
        )
        self.pool1 = nn.MaxPool2d(kernel_size=3, ceil_mode=True)
        self.conv2 = nn.Conv2d(
            in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=1
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, ceil_mode=True)
        self.conv3 = nn.Conv2d(
            in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1
        )
        self.conv4 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
        )
        self.fc1 = nn.Linear(6 * 6 * 512, 4096, bias=True)
        self.fc2 = nn.Linear(4096, 4096, bias=True)
        self.fc3 = nn.Linear(4096, class_num, bias=True)
        self.dropout_layer = nn.Dropout(p=dropout_ratio)

    def forward(self, x):
        layer0_x = self.pool1(nn.functional.relu(self.conv1(x)))
        layer1_x = self.pool2(nn.functional.relu(self.conv2(layer0_x)))
        layer2_x = nn.functional.relu(self.conv3(layer1_x))
        layer3_x = nn.functional.relu(self.conv4(layer2_x))
        layer4_x = self.pool1(nn.functional.relu(self.conv4(layer3_x)))

        layer4_x = layer4_x.view(-1, 6 * 6 * 512)
        layer5_x = nn.functional.relu(self.dropout_layer(self.fc1(layer4_x)))
        layer6_x = nn.functional.relu(self.dropout_layer(self.fc2(layer5_x)))
        layer7_x = self.dropout_layer(self.fc3(layer6_x))
        pred = torch.sigmoid(layer7_x)
        return pred

    @torch.no_grad()
    def get_all_preds(model, loader):
        all_preds = torch.tensor([])
        for batch in loader:
            images, _ = batch
            preds = model(images)
            all_preds = torch.cat((all_preds, preds), dim=0)
        utils.pred_info(all_preds)


class SlidingWindowDataset(torch.utils.data.Dataset):
    def __init__(self, *, images, padding: int, win_size, stride, transform):
        e_image, bf_image = images
        img_h, img_w = e_image.size
        self.padded_e_image = ImageOps.expand(
            e_image, (padding, padding, padding, padding)
        )
        self.padded_bf_image = ImageOps.expand(
            bf_image, (padding, padding, padding, padding)
        )
        self.MAP_H, self.MAP_W = (
            (np.array([img_h, img_w]) + 2 * padding - win_size) / stride + 1
        ).astype(int)

        self.transform = transform
        self.padding = padding
        self.win_size = win_size
        self.stride = stride
        self.cell_image_num = self.MAP_W * self.MAP_H

    def __len__(self):
        return self.cell_image_num

    def __getitem__(self, idx):
        h = int(idx / self.MAP_W)
        w = idx % self.MAP_W
        top = h * self.stride
        bottom = top + self.win_size
        left = w * self.stride
        right = left + self.win_size
        e_slided = self.padded_e_image.crop((left, top, right, bottom))
        bf_slided = self.padded_bf_image.crop((left, top, right, bottom))
        e_transformed = self.transform(e_slided)
        bf_transformed = self.transform(bf_slided)
        image_mat = torch.cat((e_transformed, bf_transformed), 0)
        return image_mat, 0


class app:
    def __init__(self) -> None:
        self.img_pair = callbacks.ImgPathPair(bright=None, blue=None)
        self.models = []
        self.gallery = []
        self.names = ("Type One", "Type Two", "Type Three", "Type Four", "Type Five")
        self.type = "Type One"
        self.texture_tags = ["Bright_Field", "Blue_Field", "Heatmap"]
        self.image_spacing = 20
        self.xaxis = None
        self.yaxis = None
        self.blue_offset = np.array([0, 0])
        self.legend = None
        self.batch_size=64
        self.winsize = 10
        self.padding = 7
        self.stride = 2
        self.target_type = 0
        self.target_device = torch.device("cpu")
        self.image_loaded = False
        self.size = 6
        self.item_dict = {}
        self.droplet_dict_locs = {"Type One":None, "Type Two":None, "Type Three":None, "Type Four":None, "Type Five":None}
        self.droplet_dict_colors = {"Type One":"red", "Type Two":"white", "Type Three":"green", "Type Four":"yellow", "Type Five":"blue"}
        self.default_font = None
        dpg.create_context()
        dpg.create_viewport(title='Custom Title', width=1920, height=1080)

    def __load_models(self):
        for i in range(5):
            model = torch.load(
                os.path.join(os.getcwd(), "models/mt{t}".format(t=i)),
                map_location=torch.device("cpu"), 
            )
            model.eval()
            self.models.append(model)

    def _create_file_selector(self):
        with dpg.file_dialog(
            directory_selector=False,
            show=False,
            tag=tags.item_tags.image_file_dialog,
            file_count=2,
            callback=callbacks.file_selector_callback,
            user_data=self,
        ):
            dpg.add_file_extension(".*", color=(255, 255, 255, 255))
            dpg.add_file_extension(".png", color=(0, 255, 0, 255))
            dpg.add_file_extension(".tiff", color=(0, 255, 0, 255))

    def __set_font(self):
        # add a font registry
        with dpg.font_registry():
            # add font (set as default for entire app)
            self.default_font = dpg.add_font("Retron2000.ttf", 20)
        dpg.bind_font(self.default_font)
    def __create_ui(self):
        self._create_file_selector()
        self.__create_main_panel()

    def __create_main_panel(self):
        with dpg.window(label="Main", width=1920, height=1080, id="main_panel_id"):
            # add a buttom to enable file dialog as image selector
            self.item_dict["image_selector"] = dpg.add_button(
                label="Image Selector", callback=lambda: dpg.show_item(tags.item_tags.image_file_dialog)
            )
            # add two text to displaye loaded image name
            dpg.add_text(
                "bright image: {img_name}".format(
                    img_name="not loaded"
                    if self.img_pair.bright is None
                    else self.img_pair.bright.split("/")[-1],
                ),
                tag= tags.item_tags.text_bright_image_name,
            )

            dpg.add_text(
                "blue image: {img_name}".format(
                    img_name="not loaded"
                    if self.img_pair.blue is None
                    else self.img_pair.blue.split("/")[-1],
                ),
                tag= tags.item_tags.text_blue_image_name,
            )
            
            self.item_dict["offset_slider"] = dpg.add_slider_intx(
                label="blue image offset",
                size=2,
                callback=callbacks.update_blue_offset,
                user_data=self,
                enabled=False,
            )
            self.item_dict["device_selector"] = dpg.add_radio_button(
                ("cpu", "gpu"),
                default_value="cpu",
                horizontal=True,
                callback=callbacks.set_device,
                user_data=app,
                enabled=False,
            )

            self.item_dict["num_threads"] = dpg.add_input_int(
                label="num threads",
                width=400,
                min_value=1,
                default_value=1,
                user_data=self,
                enabled=False,
            )
            self.item_dict["detect"] = dpg.add_button(
                label="detect", callback=callbacks.detect, user_data=self, enabled=False
            )
            
            self.item_dict["type_radio"] = dpg.add_radio_button(
                ("Type One", "Type Two", "Type Three", "Type Four", "Type Five"),
                horizontal=True,
                callback=callbacks.swtich_target_type,
                user_data=self,
                enabled=False,
            )
            self.item_dict["texture_radio"] = dpg.add_radio_button(
                ("Bright Field", "Blue Field", "Heatmap"),
                horizontal=True,
                user_data=self,
                callback=callbacks.switch_texture,
                enabled=False,
            )
            self.item_dict["padding"] = dpg.add_input_int(
                label="Padding", width=400, default_value=7, enabled=False
            )
            self.item_dict["stride"] = dpg.add_input_int(
                label="Stride", width=400, default_value=2, enabled=False
            )
            self.item_dict["winsize"] = dpg.add_input_int(
                label="Window Size", width=400, default_value=10, enabled=False
            )
            with dpg.tree_node(label="Image Series", show=True, default_open=True):
                with dpg.plot(
                    label="Image Plot",
                    height=-1,
                    width=-1,
                    tag=tags.item_tags.image_plot,
                    equal_aspects=True,
                    crosshairs=True,
                ):
                    pass
            
                    # xaxis = dpg.add_plot_axis(dpg.mvXAxis, label="x")
                    # yaxis_id = dpg.add_plot_axis(dpg.mvYAxis, label="y axis")

    # def __create_working_space(self):
    #     with dpg.window(label="working space", pos=(500, 500),id = "working_space_id"):
    #         pass
            # dpg.add_button(label='set',callback = "button_window")

<<<<<<< HEAD
            with dpg.window(tag="button_window",width=400,height=60,show=False):
=======
            with dpg.window(id="button_window",width=400,height=600,show=False):
>>>>>>> 9c03e181ae1911f72308d1424a9feb04f4a28644
                self.item_dict["Add"] = dpg.add_button(label="Add",callback = callbacks.Add,user_data=self)
                self.item_dict["Delete"] = dpg.add_button(label="Delete",callback = callbacks.Delete,user_data=self)
                self.item_dict["Size"] = dpg.add_slider_int(label="Size",default_value=6,min_value= 4,max_value=10 ,callback=callbacks.Size,user_data=self)
                self.item_dict["color"] = dpg.add_color_picker(label="Color",no_side_preview=True,display_hex=False,callback=callbacks.Color,user_data=self)
            with dpg.window(tag="Droplet",label="Droplet",width=350,height=350,show=False):
                self.Type_One = dpg.add_text("Type One : {d}".format(d=None))    
                self.Type_Two = dpg.add_text("Type Two : {d}".format(d=None))
                self.Type_Three = dpg.add_text("Type Three : {d}".format(d=None))
                self.Type_Four = dpg.add_text("Type Four : {d}".format(d=None))
                self.Type_Five = dpg.add_text("Type Five : {d}".format(d=None))             
                
        
    def launch(self):
        self.__load_models()
        print("load models")
        self.__set_font()
        print("set font")
        self.__create_ui()
        # self.__create_working_space()
        # dpg.setup_viewport()
        # dpg.set_viewport_title(title="Oil Droplet Detection")
        # dpg.set_viewport_pos([500, 500])
        # dpg.set_viewport_width(1000)
        # dpg.set_viewport_height(1000)
        # dpg.show_font_manager()
        dpg.setup_dearpygui()
        dpg.set_primary_window("main_panel_id", True)

        dpg.show_viewport()
        dpg.start_dearpygui()
        dpg.destroy_context()

 
if __name__ == "__main__":
    app = app()
    app.launch()
    
