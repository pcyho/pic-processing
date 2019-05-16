import tkinter as tk
import tkinter.filedialog as tkf
from PIL import Image, ImageTk
import os


class GUI(object):
    def gui_win(self):
        window = tk.Tk()
        window.resizable = False
        window.title(self.title)
        window.geometry(self.size)
        window.resizable = False

        self.button = tk.Button(window,
                                text='button',
                                font=('Arial', 12),
                                command=self.button_listener,
                                width=30,
                                height=3)
        self.button.pack()

        self.lable = tk.Label(window, text='img here', bg='gray')
        self.lable.pack()

        window.mainloop()

    def __init__(self, title, size):
        self.title = title
        self.size = size
        self.labeltext = None
        self.filename = None
        self.button = None
        self.lable = None
        self.gui_win()

    def button_listener(self):
        filename = tkf.askopenfile()
        if filename != '':
            self.filename = filename.name.encode().decode('utf-8')
            img = Image.open(self.filename)
            img = ImageTk.PhotoImage(img)
            self.lable.config(image=img)
            self.lable.image = img


if __name__ == '__main__':
    gui = GUI('title', '1000x600')
