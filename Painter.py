from tkinter import *
from tkinter.colorchooser import askcolor
from tkinter.ttk import * 
import PIL
import cv2
from tkinter import filedialog
import sys ,os
from PIL import Image, ImageDraw, ImageGrab

class Paint(object):

    DEFAULT_PEN_SIZE = 9.0
    DEFAULT_COLOR = 'black'

    def __init__(self):
        self.root = Tk()

        style = Style() 
        style.configure('TButton', font = 
               ('calibri', 12, 'bold'), 
                    borderwidth = '8') 
  
        # Changes will be reflected 
#        by the movement of mouse. 
        style.configure('TButton', foreground='red') 
        
  
        self.pen_button = Button(self.root, text='pen', command=self.use_pen)
        self.pen_button.grid(row=0, column=0)

        self.brush_button = Button(self.root, text='brush', command=self.use_brush)
        self.brush_button.grid(row=0, column=1)


        self.eraser_button = Button(self.root, text='eraser', command=self.use_eraser)
        self.eraser_button.grid(row=0, column=2)

        self.choose_size_button = Scale(self.root, from_=7, to=11, orient=HORIZONTAL)
        self.choose_size_button.grid(row=0, column=4)
        self.eraser_button = Button(self.root, text='Detect', command=self.save_image)
        self.eraser_button.grid(row=0, column=3)

        self.load_button = Button(self.root, text='Load from File', command=self.load_file)
        self.load_button.grid(row=0, column=5)

        self.c = Canvas(self.root, bg='white', width=600, height=600)
        self.image1 = PIL.Image.new('RGB', (600, 600), 'white')
        self.draw = ImageDraw.Draw(self.image1)


        self.c.grid(row=1, columnspan=6)

        self.setup()
        self.root.mainloop()

    def setup(self):
        self.old_x = None
        self.old_y = None
        self.line_width = 9
        self.color = self.DEFAULT_COLOR
        self.eraser_on = False
        self.active_button = self.pen_button
        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)

    def save_image(self):
        filename = f'image_{1}.png'   # image_number increments by 1 at every save
        self.image1.save(filename)
        self.root.destroy()
     

    def load_file(self):
     
        datapath = filedialog.askopenfilename()
        if not os.path.isfile(datapath):
            return
        self.image_path = cv2.imread(datapath)
        cwd = os.getcwd()
        path = cwd
        filename = f'\image_{1}.png'   # image_number increments by 1 at every save
        cv2.imwrite(path + filename,self.image_path)
        #self.root.quit()
        self.root.destroy()
        
        

    def use_pen(self):
        self.activate_button(self.pen_button)

    def use_brush(self):
        self.activate_button(self.brush_button)


    def use_eraser(self):
        self.activate_button(self.eraser_button, eraser_mode=True)

    def activate_button(self, some_button, eraser_mode=False):
        #self.active_button.config(relief=RAISED)
        #some_button.config(relief=SUNKEN)
        self.active_button = some_button
        self.eraser_on = eraser_mode

    def paint(self, event):
        if(self.choose_size_button.get() <= 9):
            self.line_width = 9
        else:
            self.line_width = self.choose_size_button.get()
        paint_color = 'white' if self.eraser_on else self.color
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x, self.old_y, event.x, event.y,
                               width=int(self.line_width), fill=paint_color,
                               capstyle=ROUND, smooth=TRUE, splinesteps=36)
            self.draw.line((self.old_x, self.old_y, event.x, event.y), fill=paint_color,  width=int(self.line_width))
        self.old_x = event.x
        self.old_y = event.y

    def reset(self, event):
        self.old_x, self.old_y = None, None


#Paint()
