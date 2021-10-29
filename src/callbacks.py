import dearpygui.dearpygui as dpg
from collections import namedtuple

from matplotlib.pyplot import semilogx, show
from numpy import core
import torch
import dpg_utils
import numpy as np
import utils
from tags import *
from core import app

ImgPathPair = namedtuple("ImgPair", ["bright", "blue"])


def image_selector_callback(sender, app_data, app: app):
    # clear previous images
    dpg_utils.clear_drawlist(app.texture_ids)
    img_keys = []
    img_types = []
    img_path = []
    for img_name in app_data["selections"].keys():
        img_path.append(app_data["selections"][img_name])
        name_features = str(img_name).split(".")[0].split("_")
        img_types.append(name_features[-1].lower())
        img_keys.append("_".join(name_features[:-1]))
    if img_keys.count(img_keys[0]) != len(img_keys):
        print("two images does not have the same key: {keys}".format(img_keys))
        return
    if not ("bf" in img_types and "e" in img_types):
        print(
            "the type of the two images should be 'bf' or 'e': {types}".format(
                types=img_types
            )
        )
        return
    if img_types[0] == "bf":
        img_path_pair = ImgPathPair(bright=img_path[0], blue=img_path[1])
    elif img_types[0] == "e":
        img_path_pair = ImgPathPair(bright=img_path[1], blue=img_path[0])

    print(
        "record target image paths: \n\t1)[bright] {br_img_name}\n\t2)[blue] {bl_img_name}".format(
            br_img_name=img_path_pair.bright, bl_img_name=img_path_pair.blue
        )
    )

    # set new image pair path
    app.img_pair = img_path_pair
    # update text in main panel
    dpg.set_value(
        "main_panel_bright_img_id",
        value="bright image: {img_name}".format(
            img_name=img_path_pair.bright.split("/")[-1],
        ),
    )
    dpg.set_value(
        "main_panel_blue_img_id",
        value="blue image: {img_name}".format(
            img_name=img_path_pair.blue.split("/")[-1],
        ),
    )
    # bright image cell
    br_img_cell = dpg_utils.add_texture_to_workspace(
        img_path_pair.bright, app.texture_ids[0], app.yaxis, True
    )
    # blue image cell
    bl_img_cell = dpg_utils.add_texture_to_workspace(
        img_path_pair.blue, app.texture_ids[1], app.yaxis, False
    )
    # heatmap image cell
    hm_img_cell = dpg_utils.add_image_buff_to_workspace(
        br_img_cell.size, app.texture_ids[2], app.yaxis, False
    )

    dpg.fit_axis_data(app.xaxis)
    dpg.fit_axis_data(app.yaxis)
    # add cell info to app
    app.gallery.append(br_img_cell)
    app.gallery.append(bl_img_cell)
    app.gallery.append(hm_img_cell)
    # inform app that the image is loaed
    app.image_loaded = True
    enable_all_items(app)
    # print(dpg.get_item_configuration(app.legend))


def check_image_loaded(app):
    if not app.image_loaded:
        print("images are not loaded")
        return False
    return True


def detect_droplets(sender, app_data, app):
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
    print("end detection: {d}".format(d=app.droplet_num))


def update_blue_offset(sender, app_data, app):
    if not check_image_loaded(app):
        return
    app.blue_offset[0] = app_data[0]
    app.blue_offset[1] = app_data[1]
    # print(app.blue_offset)
    dpg.configure_item(
        app.gallery[1].image_series_tag,
        bounds_min=app.gallery[1].loc + app.blue_offset,
        bounds_max=app.gallery[1].bottom_right() + app.blue_offset,
    )


def switch_texture(sender, app_data, app):
    if not check_image_loaded(app):
        return
    if app_data == "Bright Field":
        dpg.configure_item(app.gallery[0].image_series_tag, show=True)
        dpg.configure_item(app.gallery[1].image_series_tag, show=False)
        dpg.configure_item(app.gallery[2].image_series_tag, show=False)
    elif app_data == "Blue Field":
        dpg.configure_item(app.gallery[0].image_series_tag, show=False)
        dpg.configure_item(app.gallery[1].image_series_tag, show=True)
        dpg.configure_item(app.gallery[2].image_series_tag, show=False)
    elif app_data == "Heatmap":
        dpg.configure_item(app.gallery[0].image_series_tag, show=False)
        dpg.configure_item(app.gallery[1].image_series_tag, show=False)
        dpg.configure_item(app.gallery[2].image_series_tag, show=True)


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
    for key, val in app.item_tag_dict.items():
        dpg.enable_item(val)


def add_droplet_manually(sender, app_data, app: app):
    if dpg.is_item_hovered(item_tags.image_plot_workspace):
        mouse_pos = np.array(dpg.get_plot_mouse_pos(), dtype=np.integer)
        app.detection_data[app.target_type].append(mouse_pos)
        print(app.detection_data)
