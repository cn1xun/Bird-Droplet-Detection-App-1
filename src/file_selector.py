import dearpygui.dearpygui as dpg

def callback(sender, app_data):
    print("Sender: ", sender)
    print("App Data: ", app_data)


with dpg.file_dialog(directory_selector=False, show=False, callback=callback, id="file_dialog_id",file_count = 2):
    dpg.add_file_extension(".*", color=(255, 255, 255, 255))
    dpg.add_file_extension("Source files (*.cpp *.h *.hpp){.cpp,.h,.hpp}", color=(0, 255, 255, 255))
    dpg.add_file_extension(".cpp", color=(255, 255, 0, 255))
    dpg.add_file_extension(".h", color=(255, 0, 255, 255), custom_text="header")
    dpg.add_file_extension("Python(.py){.py}", color=(0, 255, 0, 255))
    dpg.add_file_extension(".png", color=(0, 255, 0, 255))

with dpg.window(label="Main", width=1200, height=300):
    dpg.add_button(label="Image Selector", callback=lambda: dpg.show_item("file_dialog_id"))
# add a font registry
with dpg.font_registry():
    # add font (set as default for entire app)
    dpg.add_font("Retron2000.ttf", 40, default_font=True)

dpg.start_dearpygui()