import dearpygui.dearpygui as dpg
from collections import namedtuple
import core
from matplotlib.pyplot import semilogx, show
import torch
import dpg_utils
import numpy as np
import utils
import os
from PIL import Image
from tags import *

ImgPathPair = namedtuple("ImgPair", ["bright", "blue"])
import utils


class cell_info:
    def __init__(self, tex_id, size, loc, image_id=None):
        self.texture_id = tex_id
        self.image_series_id = image_id
        self.size = size
        self.loc = loc
        self.ref = None

    def update_info(self, tex_id, image_id, size, loc):
        self.texture_id = tex_id
        self.image_series_id = image_id
        self.size = size
        self.loc = loc

    def bottom_right(self):
        return [self.loc[0] + self.size[0], self.loc[1] + self.size[1]]


def file_selector_callback(sender, app_data, app):
    img_path, img_pair = dpg_utils.parse_file_selector_data(app_data)
    print("img_path:", img_path)
    # set new image pair path
    app.img_pair = img_pair
    # update text in main panel
    dpg.set_value(
        item_tags.text_bright_image_name,
        value="bright image: {img_name}".format(
            img_name=img_pair.bright.split("/")[-1],
        ),
    )
    dpg.set_value(
        item_tags.text_blue_image_name,
        value="blue image: {img_name}".format(img_name=img_pair.blue.split("/")[-1],),
    )
    # clear previous images
    dpg_utils.clear_drawlist(app.texture_tags)
    # add images to working space
    img_br = os.path.join(os.getcwd(), img_path[0])
    img_bl = os.path.join(os.getcwd(), img_path[1])
    print("img br, img bl:", img_br, img_bl)
    br_w, br_h, channels, data_br = dpg.load_image(img_br)
    bl_w, bl_h, channels, data_bl = dpg.load_image(img_bl)
    app.img_width = br_w
    app.img_height = br_h
    # print("file_selector_callback")
    # print(app.img_width)
    # print(app.img_height)
    # with dpg.texture_registry() as reg_id:
    #     dpg.add_static_texture(width, height, data, id="image_id",parent= reg_id)
    if app.xaxis is None:
        app.xaxis = dpg.add_plot_axis(
            dpg.mvXAxis, label="x axis", parent=item_tags.image_plot
        )
    if app.yaxis is None:
        app.yaxis = dpg.add_plot_axis(
            dpg.mvYAxis, label="y axis", parent=item_tags.image_plot
        )
    # print(app.yaxis)
    dpg_utils.add_images_texture(br_w, br_h, data_br, app.texture_tags[0])
    br_size = np.array([br_w, br_h])
    br_loc = np.array([0, 0])
    br_cell = cell_info(app.texture_tags[0], br_size, br_loc)
    br_tex_ref = dpg.add_image_series(
        app.texture_tags[0],
        br_cell.loc,
        br_cell.bottom_right(),
        label=app.texture_tags[0],
        parent=app.yaxis,
    )
    br_cell.ref = br_tex_ref
    dpg_utils.add_images_texture(bl_w, bl_h, data_bl, app.texture_tags[1])
    bl_size = np.array([bl_w, bl_h])
    bl_loc = np.array([0, 0])
    bl_cell = cell_info(app.texture_tags[1], bl_size, bl_loc)
    bl_tex_ref = dpg.add_image_series(
        app.texture_tags[1],
        bl_cell.loc,
        bl_cell.bottom_right(),
        show=False,
        label=app.texture_tags[1],
        parent=app.yaxis,
    )
    bl_cell.ref = bl_tex_ref
    dpg_utils.add_heatmap_image(br_w, br_h, app.texture_tags[2])
    hm_cell = cell_info(app.texture_tags[2], br_size, br_loc)
    hm_tex_ref = dpg.add_image_series(
        app.texture_tags[2],
        br_cell.loc,
        br_cell.bottom_right(),
        show=False,
        label=app.texture_tags[2],
        parent=app.yaxis,
    )
    hm_cell.ref = hm_tex_ref
    dpg.fit_axis_data(app.xaxis)
    dpg.fit_axis_data(app.yaxis)
    # add cell info to app
    app.gallery.append(br_cell)
    app.gallery.append(bl_cell)
    app.gallery.append(hm_cell)
    # inform app that the image is loaed
    app.image_loaded = True
    enable_all_items(app)
    # w = br_w
    # h = br_h
    # if os.path.exists('empty.png') == False:
    img_bg = Image.new(mode="RGBA", size=(br_h, br_w))
    img_bg.save("empty.png")
    img_detect_path = "empty.png"
    img_detect = os.path.join(os.getcwd(), img_detect_path)
    w, h, channels, data = dpg.load_image(img_detect)
    # print("load_image",w,h)
    names = ("Type One", "Type Two", "Type Three", "Type Four", "Type Five")
    dpg_utils.add_images_texture(w, h, data, names[0] + "texture")
    dpg_utils.add_images_series(names[0], w, h)
    dpg_utils.add_images_texture(w, h, data, names[1] + "texture")
    dpg_utils.add_images_series(names[1], w, h)
    dpg_utils.add_images_texture(w, h, data, names[2] + "texture")
    dpg_utils.add_images_series(names[2], w, h)
    dpg_utils.add_images_texture(w, h, data, names[3] + "texture")
    dpg_utils.add_images_series(names[3], w, h)
    dpg_utils.add_images_texture(w, h, data, names[4] + "texture")
    dpg_utils.add_images_series(names[4], w, h)


def check_image_loaded(app):
    if not app.image_loaded:
        print("images are not loaded")
        return False
    return True


def detect(sender, app_data, app):
    if not check_image_loaded(app):
        return
    print("start detection: tpye{d}".format(d=app.target_device))
    droplet_num, predicted_map, predicted_heatmap = utils.binary_droplet_detection(
        app.img_pair.blue,
        app.img_pair.bright,
        app.batch_size,
        app.padding,
        app.stride,
        app.winsize,
        threshold=0.7,
        erosion_iter=1,
        model=app.models[app.target_type],
        device=app.target_device,
        verbose=True,
    )
    # type droplet_num
    print("end detection: {d}".format(d=droplet_num))
    news_locs = utils.droplet_loc(predicted_map)
    app.droplet_dict_locs[app.type] = utils.clean_similar(news_locs)
    # print("news_locs",news_locs)
    utils.pic_rectangle(
        app.type,
        app.droplet_dict_locs[app.type],
        outline=app.droplet_dict_colors[app.type],
    )
    dpg.show_item("button_window")


def Add():
    with dpg.handler_registry():
        print("Add :")
        while 1:
            if dpg.is_item_hovered("image_plot") == True:
                if dpg.is_item_left_clicked("image_plot") == True:
                    loc = dpg.get_plot_mouse_pos()
                    # locs = droplet_loc(predicted_map)
                    # locs.append([round(loc) for loc in loc])
                    locs = [round(loc) for loc in loc]
                    core.app.droplet_locs.append(locs)
                    utils.pic_rectangle(
                        core.app.droplet_locs, core.app.size, update=True
                    )
                    break
    pass


def Delete():
    with dpg.handler_registry():
        print("Delete :")
        while 1:
            if dpg.is_item_hovered("image_plot") == True:
                if dpg.is_item_left_clicked("image_plot") == True:
                    loc = dpg.get_plot_mouse_pos()
                    locs = [[round(loc) for loc in loc]]
                    # print(locs[0])
                    # [140, 141]
                    try_locs = utils.find_rectangle(locs[0], locs, size=core.app.size)
                    for try_loc in try_locs:
                        if try_loc in core.app.droplet_locs:
                            core.app.droplet_locs.remove(try_loc)
                    print("droplet_locs", core.app.droplet_locs)
                    utils.pic_rectangle(
                        core.app.droplet_locs, core.app.size, update=True
                    )

                    break

    pass


def Size(sender, app_data, app):
    app.size = app_data
    utils.pic_rectangle(
        app.type,
        app.droplet_dict_locs[app.type],
        app.size,
        app.droplet_dict_colors[app.type],
    )
    return app.size


def Color(sender, app_data, app):
    # app.droplet_dict_colors = app_data
    new_color = dpg.get_value(sender)
    color = []
    for i in new_color:
        color.append(int(i))
    color_tuple = tuple(color)
    print("color_tuple", color_tuple)
    utils.pic_rectangle(
        app.type, app.droplet_dict_locs[app.type], app.size, color_tuple
    )


def update_blue_offset(sender, app_data, app):
    if not check_image_loaded(app):
        return
    app.blue_offset[0] = app_data[0]
    app.blue_offset[1] = app_data[1]
    dpg.configure_item(
        app.gallery[1].ref,
        bounds_min=app.gallery[1].loc + app.blue_offset[0],
        bounds_max=app.gallery[1].bottom_right() + app.blue_offset[1],
    )


def switch_texture(sender, app_data, app):
    if not check_image_loaded(app):
        return
    if app_data == "Bright Field":
        dpg.configure_item(app.gallery[0].ref, show=True)
        dpg.configure_item(app.gallery[1].ref, show=False)
        dpg.configure_item(app.gallery[2].ref, show=False)
    elif app_data == "Blue Field":
        dpg.configure_item(app.gallery[0].ref, show=False)
        dpg.configure_item(app.gallery[1].ref, show=True)
        dpg.configure_item(app.gallery[2].ref, show=False)
    elif app_data == "Heatmap":
        dpg.configure_item(app.gallery[0].ref, show=False)
        dpg.configure_item(app.gallery[1].ref, show=False)
        dpg.configure_item(app.gallery[2].ref, show=True)


def update_padding(sender, app_data, app):
    app.padding = app_data


def update_stride(sender, app_data, app):
    app.stride = app_data


def update_win_size(sender, app_data, app):
    app.winsize = app_data


def swtich_target_type(sender, app_data, app):
    names = ("Type One", "Type Two", "Type Three", "Type Four", "Type Five")
    target_type = names.index(app_data)
    app.target_type = target_type
    print("ctarget type: {d}".format(d=names[app.target_type]))


def set_device(sender, app_data, app):
    if app_data == "cpu":
        app.target_device = torch.device("cpu")
    elif app_data == "gpu":
        print("cuda available: {d}".format(d=torch.cuda.is_available()))
        if torch.cuda.is_available():
            app.target_device = torch.device("cuda")
    print("current device: {d}".format(d=app.target_device))


def enable_all_items(app):
    for key, val in app.item_dict.items():
        dpg.enable_item(val)
