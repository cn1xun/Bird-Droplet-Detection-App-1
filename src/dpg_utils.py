import dearpygui.dearpygui as dpg
import numpy as np
import matplotlib.pyplot as plt
import core

def add_images_texture(width, height ,data ,id):
    with dpg.texture_registry():
        dpg.add_dynamic_texture(width, height, data, id=id)

def add_images_series(type,width,height,parent=76):
    dpg.add_image_series(
        type+"texture",
        (0,height),
        (width,0),
        uv_min=(0, 0),
        uv_max=(1, 1),
        show =True,
        parent=parent,
        label=type)

def add_heatmap_image(w, h, id):
    texture_buffer = np.ones((w, h, 4))
    texture_buffer[:, :, -1] = 1
    with dpg.texture_registry():
        dpg.add_dynamic_texture(w, h, texture_buffer.flatten(), id=id)


def clear_drawlist(img_ids):
    for img_id in img_ids:
        if dpg.does_item_exist(img_id):
            dpg.delete_item(img_id)
        if dpg.does_item_exist(img_id + "_tex"):
            dpg.delete_item(img_id + "_tex")
