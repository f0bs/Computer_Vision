import subprocess
import os
import imageio


class GifWriter(object):

    def __init__(self, temp_format, dest_gif):
        self.temp_format = temp_format
        self.dest_gif = dest_gif
        self.temp_filenames = []
        self.closed = False

    def append(self, image):
        if self.closed:
            raise Exception('GifWriter is already closed')
        filename = self.temp_format % len(self.temp_filenames)
        self.temp_filenames.append(filename)
        imageio.imwrite(filename, image)

    def close(self):
        frames = []
        for filename in self.temp_filenames:
        	frames.append(imageio.imread(filename))

        imageio.mimwrite(self.dest_gif, frames, format='GIF', duration=2.0/100.0, loop=0.0)
        for filename in self.temp_filenames:
            os.unlink(filename)
        self.closed = True
