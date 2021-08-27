import dearpygui.dearpygui as dpg
import numpy as np
import matplotlib.pyplot as plt


def add_image(imgae_path, id):
    im = plt.imread(imgae_path) / 255.0
    height, width, channels = im.shape
    im = np.append(im, np.ones((height, width, 1)), 2)
    with dpg.texture_registry():
        dpg.add_dynamic_texture(width, height, im, id=id)
    return width, height


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
