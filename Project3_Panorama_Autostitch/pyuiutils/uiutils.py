from PIL import Image, ImageTk, ImageDraw
import cv2
import logging
import numpy as np
import os
import tkinter.filedialog as tkFileDialog
import tkinter as tk
from tkinter import ttk
import tkinter.messagebox as tkMessageBox

# Logger for this module
logger = logging.getLogger(__name__)

# Supported filetypes
supportedFiletypes = [('JPEG Image', '*.jpg'), ('PNG Image', '*.png'),
                      ('PPM Image', '*.ppm')]

# Colors for points
color_list = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
              (0, 255, 255), (255, 0, 255)]


def error(msg):
    '''Display a message box with the title 'Error' and the message as body.'''
    tkMessageBox.showerror('Error', msg)


def convert_cv_to_tk(cv_image):
    '''Converts an OpenCV-loaded image to the format used by Tkinter.'''
    if len(cv_image.shape) == 3:
        img = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
    else:
        img = Image.fromarray(cv_image)
    return ImageTk.PhotoImage(img)


def get_fitted_dimension(object_height, object_width, container_height,
                         container_width):
    '''Computes the dimensions of an object if it were to be fitted into a
    container, preseving the aspect ratio. Returns a tuple of height, width,
    scale-up ratio, with height and width being integers.'''
    original_height = object_height
    ratio = object_width / float(object_height)
    if container_height < object_height:
        object_height = container_height
        object_width = int(ratio * object_height)
    if container_width < object_width:
        object_width = container_width
        object_height = int(object_width / ratio)
    return object_height, object_width, float(object_height) / original_height


def coordinates_of_top_left(object_height, object_width, container_height,
                            container_width):
    '''Computes the coordinates of an object which is to be centered in a
    container. Returns a tuple of y, x coordinates as floats.'''
    x = (container_width - object_width) / 2.0
    y = (container_height - object_height) / 2.0
    return y, x


class ImageWidget(tk.Canvas):
    '''This class represents a Canvas on which OpenCV images can be drawn.
       The canvas handles shrinking of the image if the image is too big,
       as well as writing of the image to files. '''

    def __init__(self, parent):
        '''Starts with empty canvas.'''
        tk.Canvas.__init__(self, parent)
        self.raw_image = None
        self.show_grayscale = False
        self.drawn_image_dim = (0, 0)  # height, width
        self.bind('<Configure>', self.redraw)

    def get_fitted_dimension(self, cv_image=None):
        '''Returns the height, width, scale-up as if the given image were to be
        fit on this canvas. Uses the raw_image if cv_image is None.'''
        cv_image = cv_image if cv_image is not None else self.raw_image
        if cv_image is None:
            raise ValueError('There is no image drawn on the canvas.')
        height, width = cv_image.shape[:2]
        if height == 0 or width == 0:
            raise ValueError('The CV image must have non-zero dimension.')
        if self.winfo_height() > 0 and self.winfo_width() > 0:
            return get_fitted_dimension(height, width, self.winfo_height(),
                                        self.winfo_width())
        else:
            return -1, -1

    def coordinates_of_top_left(self, cv_image=None):
        '''Returns the coordinates of the top left of the given image if it were
        to be fitted to the canvas. Uses raw_image if cv_image is None.'''
        height, width, _ = self.get_fitted_dimension(cv_image)
        return coordinates_of_top_left(height, width, self.winfo_height(),
                                       self.winfo_width())

    def fit_cv_image_to_canvas(self, cv_image):
        '''Fits a CV image to the size of the canvas, and returns a tuple of
        height, width, and the new CV image.'''
        height, width, _ = self.get_fitted_dimension(cv_image)
        dest = cv2.resize(cv_image, (width, height),
                          interpolation=cv2.INTER_LANCZOS4)
        return height, width, dest

    def draw_cv_image(self, cv_image):
        '''Draws the given OpenCV image and store a reference to it.'''
        assert cv_image is not None
        assert cv_image.shape[0] >= 1
        assert cv_image.shape[1] >= 1
        assert len(cv_image.shape) == 3 or cv_image.shape[2] == 3
        self.raw_image = cv_image  # preserve the image to be drawn
        self.redraw()

    def redraw(self, *args):
        '''Redraw the image on the canvas.'''
        # The initial container size is 1x1
        if self.raw_image is not None and self.winfo_height(
        ) > 1 and self.winfo_width() > 1:
            if self.show_grayscale and len(self.raw_image.shape) == 3:
                cv_image = cv2.cvtColor(self.raw_image, cv2.COLOR_BGR2GRAY)
            else:
                cv_image = self.raw_image
            height, width, small_cv_image = self.fit_cv_image_to_canvas(
                cv_image)
            img = convert_cv_to_tk(small_cv_image)
            self.tk_image = img  # prevent the image from being garbage collected
            self.delete("all")
            self.drawn_image_dim = (height, width)
            y, x = self.coordinates_of_top_left()
            self.create_image(x, y, anchor=tk.NW, image=self.tk_image)

    def write_to_file(self, filename, grayscale=False):
        '''Writes the original OpenCV image to the given file.'''
        if self.raw_image is not None:
            img = self.raw_image
            if grayscale:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(filename, img)

    def get_image(self):
        '''Returns the OpenCV image associated with this widget.'''
        return self.raw_image.copy() if self.raw_image is not None else None

    def has_image(self):
        '''Returns True if the canvas has an image drawn on it.'''
        return self.raw_image is not None

    def set_grayscale(self, value):
        '''If the passet value is true, draws the given raw_image as grayscale
        otherwise defaults to its color scheme.'''
        self.show_grayscale = value
        self.redraw()


class ClickableImageWidget(ImageWidget):
    '''An image-displaying widget that lets you click on the image to select
    points.'''

    def __init__(self, parent, dot_size=5):
        '''dot_size is the size of the clicked dots.'''
        ImageWidget.__init__(self, parent)
        self.dot_size = dot_size
        self.clicked_points = []
        self.plain_image = None
        self.bind('<Button-1>', self.handle_click)

    def get_clicked_points(self):
        return self.clicked_points[:]

    def get_clicked_points_in_image_coordinates(self):
        return [self.canvas_to_image_coordinates(y, x)
                for y, x in self.clicked_points]

    def in_bounds(self, y, x):
        '''Returns true if the given coordinates like within the drawn image.'''
        h, w, _ = self.get_fitted_dimension()
        img_y_offset, img_x_offset = self.coordinates_of_top_left()
        return (y - img_y_offset) >= 0 and (y - img_y_offset) < h and \
               (x - img_x_offset) >= 0 and (x - img_x_offset) < w

    def pop_click(self):
        '''Removes and returns the coordinates of the last clicked point lying
        within the drawn image.'''
        if len(self.clicked_points) > 0:
            old = self.clicked_points.pop()
            self.draw_all_points()
            return old

    def push_click(self, y, x):
        '''Draws a point if it is in bounds and adds it to the internal list.'''
        if self.in_bounds(y, x):
            self.clicked_points.append((y, x))
            self.draw_all_points()

    def push_click_image_coordinates(self, y, x):
        '''Draws a point if it is in bounds and adds it to the internal list.
        The coordinates are expressed in image coordinates.'''
        self.push_click(*self.image_to_canvas_coordinates(y, x))

    def draw_new_image(self, cv_image):
        '''Draw a new image on the canvas, clearing all the drawn points.
        Use this instead of draw_cv_image().'''
        self.plain_image = cv_image
        self.clicked_points = []
        self.draw_cv_image(cv_image)

    def canvas_to_image_coordinates(self, y, x):
        '''Converts the canvas-coordinates of a point to the original images'
        coordinates system.'''
        img_y_offset, img_x_offset = self.coordinates_of_top_left()
        original_height, original_width = self.raw_image.shape[:2]
        drawn_height, drawn_width = self.drawn_image_dim
        clicked_y = float(original_height) * (y - img_y_offset) / drawn_height
        clicked_x = float(original_width) * (x - img_x_offset) / drawn_width
        return (clicked_y, clicked_x)

    def image_to_canvas_coordinates(self, y, x):
        '''Converts the image coordinates to canvas-coordinates.'''
        img_y_offset, img_x_offset = self.coordinates_of_top_left()
        original_height, original_width = self.raw_image.shape[:2]
        drawn_height, drawn_width = self.drawn_image_dim
        res_y = y * drawn_width / float(original_width) + img_y_offset
        res_x = x * drawn_height / float(original_height) + img_x_offset
        return (res_y, res_x)

    def draw_all_points(self):
        '''Draws all the points previously selected.'''
        self.raw_image = self.plain_image.copy()
        _, _, scale = self.get_fitted_dimension()
        r = int(self.dot_size / scale)
        color_index = 0
        for y, x in self.clicked_points:
            clicked_coords = self.canvas_to_image_coordinates(y, x)
            clicked_y, clicked_x = int(clicked_coords[0]), int(
                clicked_coords[1])
            cv2.circle(self.raw_image, (clicked_x, clicked_y), r,
                       color_list[color_index % len(color_list)], -1)
            color_index += 1
        self.redraw()

    def handle_click(self, event):
        '''Adds a new clicked point to the internal list and redraws the
        image.'''
        self.push_click(event.y, event.x)

    def get_image(self):
        '''Returns the OpenCV image associated with this widget.'''
        return self.plain_image.copy(
        ) if self.plain_image is not None else None


class BaseFrame(tk.Frame):
    def __init__(self, parent, root, nrows, ncols, initial_status=''):
        '''
        Inputs:
            parent - the parent container
            root - The Tk root
            nrows - Number of rows including the status line
            ncols - Number of columns
            initial_status - The status text seen when the widget is first
                             loaded
        '''
        assert nrows >= 1 and ncols >= 1
        tk.Frame.__init__(self, parent)
        self.parent = parent
        self.root = root

        self.status = tk.Label(self, text=initial_status)
        self.status.grid(row=nrows,
                         column=0,
                         columnspan=ncols,
                         sticky=tk.W + tk.E)

        # Assign equal weight to all columns
        for i in range(ncols):
            self.grid_columnconfigure(i, weight=1)

    def set_status(self, text):
        self.status.configure(text=text)
        self.root.update()

    def ask_for_image(self, filename=None):
        if filename is None:
            filename = tkFileDialog.askopenfilename(
                parent=self,
                filetypes=supportedFiletypes)
        if filename and os.path.isfile(filename):
            image = cv2.imread(filename)
            self.set_status('Loaded ' + filename)
            return filename, image
        return None, None


def showMatrixDialog(parent, text='Ok', rows=0, columns=0, array=None):
    '''This displays a modal dialog with the specified row and columns.'''

    top = tk.Toplevel(parent)

    if rows == 0 or columns == 0:
        assert array is not None
        model = array
    else:
        assert rows > 0 and columns > 0
        model = np.zeros((rows, columns), dtype=np.float)

    cells = []

    for i in range(rows):
        r = []
        for j in range(columns):
            entry = tk.Entry(top)
            entry.insert(0, str(model[i, j]))
            entry.grid(row=i, column=j)
            r.append(entry)
        cells.append(r)

    def acceptButtonClick():
        for i in range(rows):
            for j in range(columns):
                try:
                    model[i, j] = float(cells[i][j].get())
                except:
                    cells[i][j].configure(bg='red')
                    return
        top.destroy()

    wasCancelled = {'value': False}

    def cancelButtonClick():
        model = None
        wasCancelled['value'] = True
        top.destroy()

    tk.Button(top,
              text=text,
              command=acceptButtonClick).grid(row=rows,
                                              column=columns - 1,
                                              sticky=tk.E + tk.W)

    tk.Button(top,
              text='Cancel',
              command=cancelButtonClick).grid(row=rows,
                                              column=0,
                                              sticky=tk.E + tk.W)

    parent.wait_window(top)

    return None if wasCancelled['value'] else model


def concatImages(imgs):
    # Skip Nones
    imgs = [x for x in imgs if x is not None]  # Filter out Nones
    if len(imgs) == 0:
        return None
    imgs = [img for img in imgs if img is not None]
    maxh = max([img.shape[0] for img in imgs]) if imgs else 0
    sumw = sum([img.shape[1] for img in imgs]) if imgs else 0
    vis = np.zeros((maxh, sumw, 3), np.uint8)
    vis.fill(255)
    accumw = 0
    for img in imgs:
        h, w = img.shape[:2]
        vis[:h, accumw:accumw + w, :] = img
        accumw += w
    return vis


def ask_for_image_path_to_save(parent):
    return tkFileDialog.asksaveasfilename(parent=parent,
                                      filetypes=supportedFiletypes)


if __name__ == '__main__':
    root = tk.Tk()
    frame = tk.Frame(root)

    def doClick():
        showMatrixDialog(frame, rows=3, columns=4)

    tk.Button(frame, text='Click', command=doClick).pack()
    frame.pack()
    root.mainloop()
