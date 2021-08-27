import dearpygui.dearpygui as dpg
import numpy as np

tex_buffer = np.ones((500, 500, 4))
tex_buffer[:, :, 0] = 0.7
tex_buffer[10:50, 10:50, 1] = 0.5
texture_data = []
for i in range(0, 100 * 100):
    texture_data.append(255 / 255)
    texture_data.append(0.5)
    texture_data.append(255 / 255)
    texture_data.append(255 / 255)

with dpg.texture_registry():
    dpg.add_dynamic_texture(100, 100, texture_data, id="texture_id")
    dpg.add_dynamic_texture(500, 500, tex_buffer.flatten(), id="texture_id0")
    dpg.add_dynamic_texture(100, 100, texture_data, id="texture_id1")


def _update_dynamic_textures(sender, app_data, user_data):

    new_color = dpg.get_value(sender)
    new_color[0] = new_color[0] / 255
    new_color[1] = new_color[1] / 255
    new_color[2] = new_color[2] / 255
    new_color[3] = new_color[3] / 255

    new_texture_data = []
    for i in range(0, 100 * 100):
        new_texture_data.append(new_color[0])
        new_texture_data.append(new_color[1])
        new_texture_data.append(new_color[2])
        new_texture_data.append(new_color[3])

    dpg.set_value("texture_id", new_texture_data)


with dpg.window(label="Tutorial"):
    dpg.add_image("texture_id")
    dpg.add_image("texture_id0")
    dpg.add_color_picker(
        (255, 0, 255, 255),
        label="Texture",
        no_side_preview=True,
        alpha_bar=True,
        width=200,
        callback=_update_dynamic_textures,
    )

dpg.start_dearpygui()
