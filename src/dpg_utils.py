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

def parse_image_selector_data(app_data):
    img_keys = []
    img_types = []
    img_path = []
    for img_name in app_data["selections"].keys():
        img_path.append(app_data["selections"][img_name])
        name_features = str(img_name).split(".")[0].split("_")
        img_types.append(name_features[-1].lower())
        img_keys.append("_".join(name_features[:-1]))
    if img_keys.count(img_keys[0]) != len(img_keys):
        print(
            "two images does not have the same key: {keys}".format(img_keys)
        )
        return
    if not ("bf" in img_types and "e" in img_types):
        print(
            "the type of the two images should be 'bf' or 'e': {types}".format(
                types=img_types
            )
        )
        return
    if img_types[0] == "bf":
        img_pair = ImgPathPair(bright=img_path[0], blue=img_path[1])
    elif img_types[0] == "e":
        img_pair = ImgPathPair(bright=img_path[1], blue=img_path[0])
