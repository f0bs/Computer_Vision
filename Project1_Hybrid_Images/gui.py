import Tkinter as tk
import argparse
import sys
sys.path.append('/Users/kb/bin/opencv-3.1.0/build/lib/')
import cv2
import hybrid
import json
import numpy as np
import os
sys.path.append('./pyuiutils/')
import pyuiutils.uiutils as uiutils
import tkFileDialog
import threading
import ttk


class ImageAlignmentFrame(uiutils.BaseFrame):
    def __init__(self, parent, root, template_file=None):
        uiutils.BaseFrame.__init__(self, parent, root, 4, 5)
        tk.Button(self,
                  text='Load First Image',
                  command=self.load_first).grid(row=0,
                                                column=0,
                                                sticky=tk.W + tk.E)
        tk.Button(self,
                  text='Load Second Image',
                  command=self.load_second).grid(row=0,
                                                 column=1,
                                                 sticky=tk.W + tk.E)
        tk.Button(self,
                  text='Undo',
                  command=self.undo).grid(row=0,
                                          column=2,
                                          sticky=tk.W + tk.E)
        tk.Button(self,
                  text='Redo',
                  command=self.redo).grid(row=0,
                                          column=3,
                                          sticky=tk.W + tk.E)
        tk.Button(self,
                  text='View Hybrid',
                  command=self.process_compute).grid(row=0,
                                                     column=4,
                                                     sticky=tk.W + tk.E)
        tk.Button(self,
                  text='Save Correspondances',
                  command=self.save_corr).grid(row=1,
                                               column=0,
                                               sticky=tk.W + tk.E)
        tk.Button(self,
                  text='Load Correspondances',
                  command=self.load_corr).grid(row=1,
                                               column=1,
                                               sticky=tk.W + tk.E)

        self.left_image_widget = uiutils.ClickableImageWidget(self)
        self.left_image_widget.grid(row=2,
                                    column=0,
                                    columnspan=2,
                                    sticky=tk.NSEW)
        self.right_image_widget = uiutils.ClickableImageWidget(self)
        self.right_image_widget.grid(row=2,
                                     column=3,
                                     columnspan=2,
                                     sticky=tk.NSEW)
        self.left_image_name = None
        self.right_image_name = None
        self.left_redo_queue = []
        self.right_redo_queue = []
        self.grid_rowconfigure(2, weight=1)
        self.image_receiver = None

        self.template_file = template_file

    def  process_template(self):
        if self.template_file is not None:
            def load_template_and_compute():
                self.load_corr(self.template_file)
                self.process_compute()

            def load_template_local():
                #self.wait_visibility()
                self.after(0, load_template_and_compute)

            threading.Thread(target=load_template_local).start()

    def load_first(self, img_name=None):
        img_name, img = self.ask_for_image(img_name)
        if img is not None:
            self.left_image_widget.draw_new_image(img)
            self.left_image_name = img_name

    def load_second(self, img_name=None):
        img_name, img = self.ask_for_image(img_name)
        if img is not None:
            self.right_image_widget.draw_new_image(img)
            self.right_image_name = img_name

    def load_corr(self, filename=None):
        if filename is None:
            filename = tkFileDialog.askopenfilename(
                parent=self,
                filetypes=[('JSON File', '*.json')])
        if filename is not None and os.path.isfile(filename):
            with open(filename, 'r') as infile:
                conf = json.load(infile, 'utf-8')
                self.load_first(conf['first_image'])
                self.load_second(conf['second_image'])
                for c in conf['first_image_points']:
                    self.left_image_widget.push_click_image_coordinates(
                            int(c[0]), int(c[1]))
                for c in conf['second_image_points']:
                    self.right_image_widget.push_click_image_coordinates(
                            int(c[0]), int(c[1]))
                self.set_status('Loaded from template ' + filename)

    def save_corr(self):
        filename = tkFileDialog.asksaveasfilename(
            parent=self,
            filetypes=[('JSON File', '*.json')])
        if filename is not None:
            conf = dict()
            conf['first_image'] = self.left_image_name
            conf['second_image'] = self.right_image_name
            conf[
                'first_image_points'
            ] = self.left_image_widget.get_clicked_points_in_image_coordinates()
            conf[
                'second_image_points'
            ] = self.right_image_widget.get_clicked_points_in_image_coordinates(
                    )
            with open(filename, 'w') as outfile:
                json.dump(conf, outfile, indent=2)
                self.set_status('Saved to template ' + filename)

    def undo(self):
        action = self.left_image_widget.pop_click()
        if action is not None:
            self.left_redo_queue.append(action)
        action = self.right_image_widget.pop_click()
        if action is not None:
            self.right_redo_queue.append(action)

    def redo(self):
        if len(self.left_redo_queue) > 0:
            action = self.left_redo_queue.pop()
            self.left_image_widget.push_click(action[0], action[1])
        if len(self.right_redo_queue) > 0:
            action = self.right_redo_queue.pop()
            self.right_image_widget.push_click(action[0], action[1])

    def get_mapping(self):
        if not (self.left_image_widget.has_image() and
                    self.right_image_widget.has_image()):
            return None
        left = self.left_image_widget.get_clicked_points_in_image_coordinates()
        right = self.right_image_widget.get_clicked_points_in_image_coordinates(
        )
        num_points = min(len(left), len(right))
        if num_points != 3:
            uiutils.error(
                'Please click on at exactly three corresponding points.')
            return None
        left = left[:num_points]
        right = right[:num_points]
        left = np.array([[x, y] for y, x in left], np.float32)
        right = np.array([[x, y] for y, x in right], np.float32)

        at = cv2.getAffineTransform(right, left)
        return at

    def set_receiver(self, receiver):
        assert receiver is not None
        self.image_receiver = receiver

    def process_compute(self):
        mapping = self.get_mapping()
        # assert mapping is not None and mapping.shape == (2, 3)
        # assert self.image_receiver is not None
        if mapping is not None and self.image_receiver is not None:
            self.image_receiver(self.left_image_widget.get_image(),
                                self.right_image_widget.get_image(), mapping)


class HybridImageFrame(uiutils.BaseFrame):
    def __init__(self, parent, root, receiver, tab_num, config_file=None):
        uiutils.BaseFrame.__init__(self, parent, root, 7, 4)

        tk.Label(self,
                 text='Left Image Sigma:').grid(row=0,
                                                column=0,
                                                sticky=tk.E)
        self.left_sigma_slider = tk.Scale(self,
                                          from_=0.1,
                                          to=10,
                                          resolution=0.1,
                                          orient=tk.HORIZONTAL)
        self.left_sigma_slider.grid(row=0, column=1, sticky=tk.E + tk.W)
        self.left_sigma_slider.bind('<ButtonRelease-1>', self.update_hybrid)

        tk.Label(self,
                 text='Right Image Sigma:').grid(row=0,
                                                 column=2,
                                                 sticky=tk.E)
        self.right_sigma_slider = tk.Scale(self,
                                           from_=0.1,
                                           to=10,
                                           resolution=0.1,
                                           orient=tk.HORIZONTAL)
        self.right_sigma_slider.grid(row=0, column=3, sticky=tk.E + tk.W)
        self.right_sigma_slider.bind('<ButtonRelease-1>', self.update_hybrid)

        tk.Label(self,
                 text='Left Image Kernel Size:').grid(row=1,
                                                      column=0,
                                                      sticky=tk.E)
        self.left_size_slider = tk.Scale(self,
                                         from_=3,
                                         to=25,
                                         resolution=1,
                                         orient=tk.HORIZONTAL)
        self.left_size_slider.grid(row=1, column=1, sticky=tk.E + tk.W)
        self.left_size_slider.bind('<ButtonRelease-1>', self.update_hybrid)

        tk.Label(self,
                 text='Right Image Kernel Size:').grid(row=1,
                                                       column=2,
                                                       sticky=tk.E)
        self.right_size_slider = tk.Scale(self,
                                          from_=3,
                                          to=25,
                                          resolution=1,
                                          orient=tk.HORIZONTAL)
        self.right_size_slider.grid(row=1, column=3, sticky=tk.E + tk.W)
        self.right_size_slider.bind('<ButtonRelease-1>', self.update_hybrid)

        self.left_high_low_indicator = tk.StringVar()
        self.left_high_low_indicator.set('low')
        tk.Radiobutton(self,
                       text='High Pass',
                       variable=self.left_high_low_indicator,
                       value='high').grid(row=2,
                                          column=0,
                                          sticky=tk.W)
        tk.Radiobutton(self,
                       text='Low Pass',
                       variable=self.left_high_low_indicator,
                       value='low').grid(row=2,
                                         column=1,
                                         sticky=tk.W)
        self.left_high_low_indicator.trace('w', self.update_hybrid)

        self.right_high_low_indicator = tk.StringVar()
        self.right_high_low_indicator.set('high')
        tk.Radiobutton(self,
                       text='High Pass',
                       variable=self.right_high_low_indicator,
                       value='high').grid(row=2,
                                          column=2,
                                          sticky=tk.W)
        tk.Radiobutton(self,
                       text='Low Pass',
                       variable=self.right_high_low_indicator,
                       value='low').grid(row=2,
                                         column=3,
                                         sticky=tk.W)
        self.right_high_low_indicator.trace('w', self.update_hybrid)

        tk.Label(self,
                 text='Mix-in Ratio (0=left, 1=right):').grid(row=3,
                                                              column=0,
                                                              sticky=tk.E)
        self.mixin_slider = tk.Scale(self,
                                     from_=0.0,
                                     to=1.0,
                                     resolution=0.05,
                                     orient=tk.HORIZONTAL)
        self.mixin_slider.grid(row=3, column=1, sticky=tk.E + tk.W)
        self.mixin_slider.set(0.5)
        self.mixin_slider.bind('<ButtonRelease-1>', self.update_hybrid)

        self.view_grayscale = tk.IntVar()
        tk.Checkbutton(self, text='View Result in Grayscale', variable=self.view_grayscale).grid(row=3, column=2, sticky=tk.E)
        self.view_grayscale.trace('w', self.change_view_color_space)

        tk.Label(self,
                 text='Scale factor (1=left, 5=right):').grid(row=4,
                                                              column=0,
                                                              sticky=tk.E)
        self.scale_slider = tk.Scale(self,
                                     from_=1.0,
                                     to=5.0,
                                     resolution=0.2,
                                     orient=tk.HORIZONTAL)
        self.scale_slider.grid(row=4, column=1, sticky=tk.E + tk.W)
        self.scale_slider.set(2.0)
        self.scale_slider.bind('<ButtonRelease-1>', self.update_hybrid)

        tk.Button(self,
                  text='Save Configuration',
                  command=self.save_conf).grid(row=5,
                                               column=0,
                                               sticky=tk.W + tk.E)
        tk.Button(self,
                  text='Load Configuration',
                  command=self.load_conf).grid(row=5,
                                               column=1,
                                               sticky=tk.W + tk.E)
        self.save_grayscale = tk.IntVar()
        tk.Checkbutton(self, text='Save Result in Grayscale', variable=self.save_grayscale).grid(row=4, column=2, sticky=tk.E)

        tk.Button(self,
                  text='Save Hybrid Image', command=self.save_image
                  ).grid(row=4,
                                               column=3,
                                               sticky=tk.W + tk.E)

        self.image_widget = uiutils.ImageWidget(self)
        self.image_widget.grid(row=6, column=0, columnspan=4, sticky=tk.NSEW)
        self.grid_rowconfigure(6, weight=1)
        self.left_image = None
        self.right_image = None
        self.tab_num = tab_num
        if config_file is not None:
            self.load_conf(config_file)
        receiver.set_receiver(self.set_images_and_mapping)

    def set_images_and_mapping(self, img1, img2, mapping):
        assert img1 is not None and img2 is not None and mapping is not None
        assert mapping.shape == (2, 3)
        self.left_image = img1
        h, w = img1.shape[:2]
        self.right_image = cv2.warpAffine(img2, mapping, (w, h),
                                          borderMode=cv2.BORDER_REFLECT)
        if self.tab_num >= 0:
            self.parent.tab(self.tab_num, state=tk.NORMAL)
            self.parent.select(self.tab_num)
        self.update_hybrid()

    def update_hybrid(self, *args):
        if self.left_image is not None and self.right_image is not None:
            left_kernel_size = int(self.left_size_slider.get() / 2) * 2 + 1
            right_kernel_size = int(self.right_size_slider.get() / 2) * 2 + 1
            hybrid_image = hybrid.create_hybrid_image(
                self.left_image, self.right_image,
                self.left_sigma_slider.get(), left_kernel_size,
                self.left_high_low_indicator.get(),
                self.right_sigma_slider.get(), right_kernel_size,
                self.right_high_low_indicator.get(), self.mixin_slider.get(), self.scale_slider.get())
            self.image_widget.draw_cv_image(hybrid_image)

    def change_view_color_space(self, *args):
        self.image_widget.set_grayscale(self.view_grayscale.get() == 1)

    def load_conf(self, filename=None):
        if filename is None:
            filename = tkFileDialog.askopenfilename(parent=self,
                    filetypes=[('JSON file', '*.json')])
        if filename is not None:
            with open(filename, 'r') as infile:
                conf = json.load(infile, 'utf-8')
                self.left_sigma_slider.set(conf['left_sigma'])
                self.left_size_slider.set(conf['left_size'])
                self.left_high_low_indicator.set(conf['left_mode'].lower())
                self.right_sigma_slider.set(conf['right_sigma'])
                self.right_size_slider.set(conf['right_size'])
                self.right_high_low_indicator.set(conf['right_mode'].lower())
                self.mixin_slider.set(conf['mixin_ratio'])
                self.scale_slider.set(conf['scale_factor'])
                self.view_grayscale.set(conf['view_grayscale'])
                self.save_grayscale.set(conf['save_grayscale'])
                self.set_status('Loaded config from ' + filename)


    def save_conf(self):
        filename = tkFileDialog.asksaveasfilename(
            parent=self,
            filetypes=[('JSON File', '*.json')])
        if filename is not None:
            conf = dict()
            conf['left_sigma'] = self.left_sigma_slider.get()
            conf['left_size'] = self.left_size_slider.get()
            conf['left_mode'] = self.left_high_low_indicator.get()
            conf['right_sigma'] = self.right_sigma_slider.get()
            conf['right_size'] = self.right_size_slider.get()
            conf['right_mode'] = self.right_high_low_indicator.get()
            conf['mixin_ratio'] = self.mixin_slider.get()
            conf['scale_factor'] = self.scale_slider.get()
            conf['view_grayscale'] = self.view_grayscale.get()
            conf['save_grayscale'] = self.save_grayscale.get()
            with open(filename, 'w') as outfile:
                json.dump(conf, outfile, indent=2)
                self.set_status('Saved config to ' + filename)

    def save_image(self):
        f = uiutils.ask_for_image_path_to_save(self)
        if f is not None:
            self.image_widget.write_to_file(f, self.save_grayscale.get() == 1)


class HybridImagesUIFrame(tk.Frame):
    def __init__(self, parent, root, template_file=None, config_file=None):
        tk.Frame.__init__(self, parent)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        notebook = ttk.Notebook(self)
        notebook.grid(row=0, sticky=tk.NSEW)
        alignment_frame = ImageAlignmentFrame(notebook, root, template_file)
        notebook.add(alignment_frame, text='Align Images')
        hybrid_frame = HybridImageFrame(notebook, root, alignment_frame, 1,
                config_file)
        notebook.add(hybrid_frame, text='View Hybrid')
        notebook.tab(1, state=tk.DISABLED) # Will be enabled after alignment
        # prevent  conflict between  threads on windows
        alignment_frame.process_template()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Run the Hybrid Images GUI.')
    parser.add_argument('--template', '-t',
                        help='A template file.',
                        default=None)
    parser.add_argument('--config', '-c', help='Configuration for generating ' +
            'the hybrid image.', default=None)
    args = parser.parse_args()
    root = tk.Tk()
    root.title('Cornell CS 5670 - Hybrid Images Project')
    w, h = root.winfo_screenwidth(), root.winfo_screenheight() - 50
    root.geometry('{}x{}+0+0'.format(w, h))
    root.grid_columnconfigure(0, weight=1)
    root.grid_rowconfigure(0, weight=1)
    app = HybridImagesUIFrame(root, root, args.template, args.config)
    app.grid(row=0, sticky=tk.NSEW)
    root.mainloop()
