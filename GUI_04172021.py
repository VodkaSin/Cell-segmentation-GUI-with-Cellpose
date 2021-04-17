import wx
import wx.xrc
import numpy as np
import os.path
import cv2
import ntpath
from PIL import Image, ImageEnhance, ImageFont, ImageDraw
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from cellpose import utils
from cellpose import models
from cellpose import plot
from cellpose import io
from matplotlib.patches import Circle

def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img

def path_leaf(path, re = 'name'):
    head, tail = ntpath.split(path)
    if re == 'name':
        return tail or ntpath.basename(head)
    elif re == 'dir':
        return head or ntpath.basename(tail)

def get_locations(data,maxkey):
    locations = []
    for key in range(maxkey):
        found = []
        w = len(data[0])
        h = len(data)
        for i in range(h):
            for j in range(w):
                if data[i][j] == key:
                    found.append([j, i])
        locations.append(found)
    return locations

def get_labels(dir, region = None):
    data = np.load(dir, allow_pickle=True).item()
    mask = data['masks']
    if region is not None:
        mask = mask[region[0]:region[2],region[1]:region[3]]
    maxkey = np.amax(mask)
    return mask, maxkey

def get_centers(data,maxkey):
    locations = get_locations(data,maxkey)
    means = []
    for i in range(len(locations)):
        loc = locations[i]
        mean = ((np.max(loc, axis = 0)+np.min(loc, axis = 0))/2).astype(int)
        means.append(tuple(mean))
    print(means)
    return means



###########################################################################
## Class MyFrame1
###########################################################################

class MyFrame1 ( wx.Frame ):

    def __init__( self, parent ):
        wx.Frame.__init__ ( self, parent, id = wx.ID_ANY, title = 'Cell segmentation 4.11', pos = wx.DefaultPosition, size = wx.Size( 1050,740 ), style = wx.DEFAULT_FRAME_STYLE|wx.BORDER_SIMPLE|wx.TAB_TRAVERSAL )


        self.SetSizeHints( wx.DefaultSize, wx.DefaultSize )

        self.wxSBmp1 = wx.Bitmap(500,500)
        self.wxSBmp2 = wx.Bitmap(500,500)
        self.wxImg1 = wx.Image(500,500)
        self.wxImg2 = wx.Image(500,500)
        self.pilImg1 = Image.new(mode = "RGB", size = (500,500))
        self.pilImg2 = Image.new(mode = "RGB", size = (500,500))
        self.pilImgGr = Image.new(mode = "L", size = (500,500))
        self.number1 = 0.4
        self.number3 = 0.0
        self.pilImg1name = ''
        self.centers = []
        self.model = 0
        self.channel = 3
        self.compressed = False
        self.centerShow = False
        self.regionShow = False
        self.scanregion = None

        bSizer1 = wx.BoxSizer( wx.VERTICAL )

        self.m_panel7 = wx.Panel( self, wx.ID_ANY, wx.DefaultPosition, wx.Size( -1,40 ), wx.TAB_TRAVERSAL )
        self.m_panel7.SetMinSize( wx.Size( -1,40 ) )
        self.m_panel7.SetMaxSize( wx.Size( -1,40 ) )

        bSizer5 = wx.BoxSizer( wx.HORIZONTAL )

        self.m_panel8 = wx.Panel( self.m_panel7, wx.ID_ANY, wx.DefaultPosition, wx.Size( 500,-1 ), wx.TAB_TRAVERSAL )
        bSizer6 = wx.BoxSizer( wx.VERTICAL )

        self.m_staticText7 = wx.StaticText( self.m_panel8, wx.ID_ANY, u"Original Image", wx.DefaultPosition, wx.DefaultSize, 5 )
        self.m_staticText7.Wrap( -1 )

        self.m_staticText7.SetFont( wx.Font( 12, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL, False, "Arial" ) )

        bSizer6.Add( self.m_staticText7, 0, wx.ALIGN_CENTER|wx.ALL, 5 )


        self.m_panel8.SetSizer( bSizer6 )
        self.m_panel8.Layout()
        bSizer5.Add( self.m_panel8, 1, wx.EXPAND |wx.ALL, 0 )

        self.m_panel9 = wx.Panel( self.m_panel7, wx.ID_ANY, wx.DefaultPosition, wx.Size( 500,-1 ), wx.TAB_TRAVERSAL )
        bSizer7 = wx.BoxSizer( wx.VERTICAL )

        self.m_staticText8 = wx.StaticText( self.m_panel9, wx.ID_ANY, u"Processed Image", wx.DefaultPosition, wx.DefaultSize, 5 )
        self.m_staticText8.Wrap( -1 )

        self.m_staticText8.SetFont( wx.Font( 12, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL, False, "Arial" ) )

        bSizer7.Add( self.m_staticText8, 0, wx.ALIGN_CENTER|wx.ALL, 5 )


        self.m_panel9.SetSizer( bSizer7 )
        self.m_panel9.Layout()
        bSizer5.Add( self.m_panel9, 1, wx.EXPAND |wx.ALL, 0 )


        self.m_panel7.SetSizer( bSizer5 )
        self.m_panel7.Layout()
        bSizer1.Add( self.m_panel7, 1, wx.EXPAND |wx.ALL, 0 )

        self.m_panel10 = wx.Panel( self, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL )
        bSizer2 = wx.BoxSizer( wx.HORIZONTAL )

        self.m_panel1 = wx.Panel( self.m_panel10, wx.ID_ANY, wx.DefaultPosition, wx.Size( 500,500 ), wx.TAB_TRAVERSAL )
        bSizer9 = wx.BoxSizer( wx.VERTICAL )

        self.m_bitmap1 = wx.StaticBitmap( self.m_panel1, wx.ID_ANY, wx.NullBitmap, wx.DefaultPosition, wx.Size( 500,500 ), 0 )
        bSizer9.Add( self.m_bitmap1, 0, wx.ALL|wx.EXPAND, 5 )


        self.m_panel1.SetSizer( bSizer9 )
        self.m_panel1.Layout()
        bSizer2.Add( self.m_panel1, 1, wx.EXPAND |wx.ALL, 5 )

        self.m_panel2 = wx.Panel( self.m_panel10, wx.ID_ANY, wx.DefaultPosition, wx.Size( 500,500 ),wx.TAB_TRAVERSAL )
        bSizer10 = wx.BoxSizer( wx.VERTICAL )

        self.m_bitmap2 = wx.StaticBitmap( self.m_panel2, wx.ID_ANY, wx.NullBitmap, wx.DefaultPosition, wx.Size( 500,500 ), 0 )
        bSizer10.Add( self.m_bitmap2, 0, wx.ALL|wx.EXPAND, 5 )


        self.m_panel2.SetSizer( bSizer10 )
        self.m_panel2.Layout()
        bSizer2.Add( self.m_panel2, 1, wx.EXPAND |wx.ALL, 5 )


        self.m_panel10.SetSizer( bSizer2 )
        self.m_panel10.Layout()
        bSizer2.Fit( self.m_panel10 )
        bSizer1.Add( self.m_panel10, 1, wx.EXPAND |wx.ALL, 0 )

        self.m_panel3 = wx.Panel( self, wx.ID_ANY, wx.DefaultPosition, wx.Size( -1,100 ), wx.TAB_TRAVERSAL )
        self.m_panel3.SetMinSize( wx.Size( -1,100 ) )

        bSizer4 = wx.BoxSizer( wx.HORIZONTAL )

        self.m_panel5 = wx.Panel( self.m_panel3, wx.ID_ANY, wx.DefaultPosition, wx.Size( 500,-1 ), wx.BORDER_SIMPLE|wx.TAB_TRAVERSAL )
        gbSizer1 = wx.GridBagSizer( 0, 0 )
        gbSizer1.SetFlexibleDirection( wx.BOTH )
        gbSizer1.SetNonFlexibleGrowMode( wx.FLEX_GROWMODE_SPECIFIED )

        self.m_staticText1 = wx.StaticText( self.m_panel5, wx.ID_ANY, u"File name:", wx.DefaultPosition, wx.Size( 80,-1 ), 0 )
        self.m_staticText1.Wrap( -1 )

        gbSizer1.Add( self.m_staticText1, wx.GBPosition( 0, 0 ), wx.GBSpan( 1, 1 ), wx.ALL, 5 )

        self.m_staticText2 = wx.StaticText( self.m_panel5, wx.ID_ANY, u"Size:", wx.DefaultPosition, wx.Size( 80,-1 ), 0 )
        self.m_staticText2.Wrap( -1 )

        gbSizer1.Add( self.m_staticText2, wx.GBPosition( 1, 0 ), wx.GBSpan( 1, 1 ), wx.ALL, 5 )

        self.m_staticText3 = wx.StaticText( self.m_panel5, wx.ID_ANY, u"Count cells:", wx.DefaultPosition, wx.Size( 80,-1 ), 0 )
        self.m_staticText3.Wrap( -1 )

        gbSizer1.Add( self.m_staticText3, wx.GBPosition( 2, 0 ), wx.GBSpan( 1, 1 ), wx.ALL, 5 )

        self.m_filename = wx.StaticText( self.m_panel5, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, 0 )
        self.m_filename.Wrap( -1 )

        gbSizer1.Add( self.m_filename, wx.GBPosition( 0, 1 ), wx.GBSpan( 1, 1 ), wx.ALL, 5 )

        self.m_width = wx.StaticText( self.m_panel5, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, 0 )
        self.m_width.Wrap( -1 )

        gbSizer1.Add( self.m_width, wx.GBPosition( 1, 1 ), wx.GBSpan( 1, 1 ), wx.ALL, 5 )

        self.m_height = wx.StaticText( self.m_panel5, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, 0 )
        self.m_height.Wrap( -1 )

        gbSizer1.Add( self.m_height, wx.GBPosition( 2, 1 ), wx.GBSpan( 1, 1 ), wx.ALL, 5 )


        self.m_panel5.SetSizer( gbSizer1 )
        self.m_panel5.Layout()
        bSizer4.Add( self.m_panel5, 1, wx.EXPAND |wx.ALL, 5 )

        self.m_panel6 = wx.Panel( self.m_panel3, wx.ID_ANY, wx.DefaultPosition, wx.Size( 500,-1 ), wx.BORDER_SIMPLE|wx.TAB_TRAVERSAL )
        gbSizer2 = wx.GridBagSizer( 0, 0 )
        gbSizer2.SetFlexibleDirection( wx.BOTH )
        gbSizer2.SetNonFlexibleGrowMode( wx.FLEX_GROWMODE_SPECIFIED )

        self.m_slider1 = wx.Slider( self.m_panel6, wx.ID_ANY, 50, 0, 100, wx.DefaultPosition, wx.Size( 120,-1 ), wx.SL_HORIZONTAL|wx.BORDER_SIMPLE )
        gbSizer2.Add( self.m_slider1, wx.GBPosition( 0, 1 ), wx.GBSpan( 1, 1 ), wx.ALL, 5 )

        m_modelChoices = [ u"Cytoplasm", u"Nuclei" ]
        self.m_model = wx.Choice( self.m_panel6, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, m_modelChoices, 0 )
        self.m_model.SetSelection( 0 )
        gbSizer2.Add( self.m_model, wx.GBPosition( 0, 3 ), wx.GBSpan( 1, 1 ), wx.ALL, 5 )

        self.m_staticText9 = wx.StaticText( self.m_panel6, wx.ID_ANY, u"Flow Threshold", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.m_staticText9.Wrap( -1 )

        gbSizer2.Add( self.m_staticText9, wx.GBPosition( 0, 0 ), wx.GBSpan( 1, 1 ), wx.ALL, 5 )

        self.m_staticText10 = wx.StaticText( self.m_panel6, wx.ID_ANY, u"Model", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.m_staticText10.Wrap( -1 )

        gbSizer2.Add( self.m_staticText10, wx.GBPosition( 0, 2 ), wx.GBSpan( 1, 1 ), wx.ALL, 5 )

        self.m_checkBox1 = wx.CheckBox( self.m_panel6, wx.ID_ANY, u"Get center", wx.DefaultPosition, wx.DefaultSize, 0 )
        gbSizer2.Add( self.m_checkBox1, wx.GBPosition( 0, 5 ), wx.GBSpan( 1, 1 ), wx.ALL, 5 )

        self.m_slider3 = wx.Slider( self.m_panel6, wx.ID_ANY, 50, 0, 200, wx.DefaultPosition, wx.Size( 120,-1 ), wx.SL_HORIZONTAL|wx.BORDER_SIMPLE )
        gbSizer2.Add( self.m_slider3, wx.GBPosition( 1, 1 ), wx.GBSpan( 1, 1 ), wx.ALL, 5 )

        m_channelChoices = [ u"Grey", u"Red", u"Green", u"Blue"]
        self.m_channel = wx.Choice( self.m_panel6, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, m_channelChoices, 0 )
        self.m_channel.SetSelection( 3 )
        gbSizer2.Add( self.m_channel, wx.GBPosition( 1, 3 ), wx.GBSpan( 1, 1 ), wx.ALL, 5 )

        self.m_staticText12 = wx.StaticText( self.m_panel6, wx.ID_ANY, u"Channel", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.m_staticText12.Wrap( -1 )

        self.m_checkBox2 = wx.CheckBox( self.m_panel6, wx.ID_ANY, u"Region on", wx.DefaultPosition, wx.DefaultSize, 0 )
        gbSizer2.Add( self.m_checkBox2, wx.GBPosition( 1, 5 ), wx.GBSpan( 1, 1 ), wx.ALL, 5 )

        gbSizer2.Add( self.m_staticText12, wx.GBPosition( 1, 2 ), wx.GBSpan( 1, 1 ), wx.ALL, 5 )



        self.m_staticText11 = wx.StaticText( self.m_panel6, wx.ID_ANY, u"Cell Threshold", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.m_staticText11.Wrap( -1 )
        gbSizer2.Add( self.m_staticText11, wx.GBPosition( 1, 0 ), wx.GBSpan( 1, 1 ), wx.ALL, 5 )

        #       gbSizer2.Add( self.m_staticText11, wx.GBPosition( 1, 0 ), wx.GBSpan( 1, 1 ), wx.ALL, 5 )


        self.m_panel6.SetSizer( gbSizer2 )
        self.m_panel6.Layout()
        bSizer4.Add( self.m_panel6, 1, wx.EXPAND |wx.ALL, 5 )


        self.m_panel3.SetSizer( bSizer4 )
        self.m_panel3.Layout()
        bSizer1.Add( self.m_panel3, 1, wx.EXPAND |wx.ALL, 0 )

        self.m_panel4 = wx.Panel( self, wx.ID_ANY, wx.DefaultPosition, wx.Size( -1,50 ), wx.TAB_TRAVERSAL )
        self.m_panel4.SetMaxSize( wx.Size( -1,50 ) )

        bSizer3 = wx.BoxSizer( wx.HORIZONTAL )

        self.m_load = wx.Button( self.m_panel4, wx.ID_ANY, u"Load file", wx.DefaultPosition, wx.DefaultSize, 0 )
        bSizer3.Add( self.m_load, 0, wx.ALL, 5 )

        #        self.m_capture = wx.Button( self.m_panel4, wx.ID_ANY, u"Capture Image", wx.DefaultPosition, wx.DefaultSize, 0 )
        #        bSizer3.Add( self.m_capture, 0, wx.ALL, 5 )


        self.m_convert = wx.Button( self.m_panel4, wx.ID_ANY, u"Convert to Grey", wx.DefaultPosition, wx.DefaultSize, 0 )
        bSizer3.Add( self.m_convert, 0, wx.ALL, 5 )

        self.m_reset = wx.Button( self.m_panel4, wx.ID_ANY, u"Reset", wx.DefaultPosition, wx.DefaultSize, 0 )
        bSizer3.Add( self.m_reset, 0, wx.ALL, 5 )

        self.m_find = wx.Button( self.m_panel4, wx.ID_ANY, u"Find Cells", wx.DefaultPosition, wx.DefaultSize, 0 )
        bSizer3.Add( self.m_find, 0, wx.ALL, 5 )

        self.m_generate = wx.Button( self.m_panel4, wx.ID_ANY, u"Show Cellpose", wx.DefaultPosition, wx.DefaultSize, 0 )
        bSizer3.Add( self.m_generate, 0, wx.ALL, 5 )

        self.m_outline = wx.Button( self.m_panel4, wx.ID_ANY, u"Show Outline", wx.DefaultPosition, wx.DefaultSize, 0 )
        bSizer3.Add( self.m_outline, 0, wx.ALL, 5 )

        self.m_center = wx.Button( self.m_panel4, wx.ID_ANY, u"Show Centers", wx.DefaultPosition, wx.DefaultSize, 0 )
        bSizer3.Add( self.m_center, 0, wx.ALL, 5 )



        self.m_savePoints = wx.Button( self.m_panel4, wx.ID_ANY, u"Export Centers", wx.DefaultPosition, wx.DefaultSize, 0 )
        bSizer3.Add( self.m_savePoints, 0, wx.ALL, 5 )

        self.m_saveMap = wx.Button( self.m_panel4, wx.ID_ANY, u"Export Map", wx.DefaultPosition, wx.DefaultSize, 0 )
        bSizer3.Add( self.m_saveMap, 0, wx.ALL, 5 )

        self.m_saveImage = wx.Button( self.m_panel4, wx.ID_ANY, u"Save Image", wx.DefaultPosition, wx.DefaultSize, 0 )
        bSizer3.Add( self.m_saveImage, 0, wx.ALL, 5 )

        self.m_panel4.SetSizer( bSizer3 )
        self.m_panel4.Layout()
        bSizer1.Add( self.m_panel4, 1, wx.EXPAND |wx.ALL, 0 )


        self.SetSizer( bSizer1 )
        self.Layout()

        self.Centre( wx.BOTH )
        # Menu
        self.m_menubar1 = wx.MenuBar( 0 )
        self.m_menu1 = wx.Menu()
        self.m_menuItem1 = wx.MenuItem( self.m_menu1, wx.ID_ANY, u"Scanning region", wx.EmptyString, wx.ITEM_NORMAL )
        self.m_menu1.Append( self.m_menuItem1 )
        self.m_Multiple = wx.MenuItem( self.m_menu1, wx.ID_ANY, u"Multiple images", wx.EmptyString, wx.ITEM_NORMAL )
        self.m_menu1.Append( self.m_Multiple )

        self.m_menubar1.Append( self.m_menu1, u"Settings" )

        self.SetMenuBar( self.m_menubar1 )


        self.Centre( wx.BOTH )

        # Connect Events
        self.m_slider1.Bind( wx.EVT_SLIDER, self.OnChangeSlider1 )
        self.m_slider3.Bind( wx.EVT_SLIDER, self.OnChangeSlider3 )
        self.m_load.Bind( wx.EVT_BUTTON, self.ClickLoad )
        #        self.m_capture.Bind( wx.EVT_BUTTON, self.ClickCapture )
        self.m_convert.Bind( wx.EVT_BUTTON, self.ClickConvert )
        self.m_reset.Bind( wx.EVT_BUTTON, self.ClickReset )
        self.m_find.Bind( wx.EVT_BUTTON, self.ClickFind )
        self.m_generate.Bind( wx.EVT_BUTTON, self.ClickGenerate )
        self.m_outline.Bind( wx.EVT_BUTTON, self.ClickOutline )
        self.m_center.Bind( wx.EVT_BUTTON, self.ClickCenter )
        self.m_saveImage.Bind( wx.EVT_BUTTON, self.ClickSaveImage )
        self.m_savePoints.Bind( wx.EVT_BUTTON, self.ClickSavePoints )
        self.m_saveMap.Bind( wx.EVT_BUTTON, self.ClickSaveMap )
        self.Bind( wx.EVT_MENU, self.InputRegion, id = self.m_menuItem1.GetId() )
        self.Bind( wx.EVT_MENU, self.LoadMultiple, id = self.m_Multiple.GetId() )
        self.m_model.Bind( wx.EVT_CHOICE, self.ModelChoice )
        self.m_channel.Bind( wx.EVT_CHOICE, self.ChannelChoice )
        self.m_checkBox1.Bind( wx.EVT_CHECKBOX, self.CenterOn )
        self.m_checkBox2.Bind( wx.EVT_CHECKBOX, self.RegionOn )

    def __del__( self ):
        pass



    # Virtual event handlers, overide them in your derived class
    def ModelChoice( self, event ):
        self.model = self.m_model.GetSelection()
        print('model'+str(self.model))

    def ChannelChoice( self, event ):
        self.channel = self.m_channel.GetSelection()
        print('channel'+ str(self.channel))

    def CenterOn( self, event ):
        self.centerShow = self.m_checkBox1.GetValue()
        print(self.centerShow)


    def ShowCenter(self):
        image = self.pilImg2
        draw = ImageDraw.Draw(image)
        r = 2
        for i in range(len(self.centers)):
            x = self.centers[i][0]
            y = self.centers[i][1]
            leftup = (x-r,y-r)
            rightup = (x+r,y-r)
            rightdown = (x+r,y+r)
            leftdown = (x-r,y+r)
            twopoint1 = [leftup,rightdown]
            twopoint2 = [rightup,leftdown]
            draw.line(twopoint1, fill=(255,0,0,0))
            draw.line(twopoint2, fill=(255,0,0,0))
        new_img = image
        self.centerimage = new_img
        self.NewIm2(new_img)


    def RegionOn( self, event ):
        self.regionShow = self.m_checkBox2.GetValue()
        if self.regionShow:
            print('Region on')
            image = self.pilImg1.copy()
            draw = ImageDraw.Draw(image)
            draw.rectangle(self.scanregion,outline='purple')
            self.NewIm1(image)
            self.cropped = image.crop(self.scanregion)
            self.NewIm2(self.cropped)
        else:
            self.ClickReset()




    def InputRegion( self, event ):
        a = MyDialog1(self)
        res = a.ShowModal()
        if res == wx.ID_OK or wx.ID_CANCEL:
            self.scanregion = a.Get_value()
        a.Destroy()
        print(self.scanregion)

    def LoadMultiple( self, event ):
        wcd = 'Images (*.jpg,*.png,*.bmp,*.tif)|*.jpg;*.png;*.bmp;*.tif'
        dlg = wx.FileDialog(self, "Load files", "", "", wcd, wx.FD_MULTIPLE)
        if dlg.ShowModal() == wx.ID_OK:
            print('OK loaded')
            files = dlg.GetPaths()
            print(files)
            print('Number of files: ',len(files))
            imgs = [io.imread(f) for f in files]
            type = os.path.splitext(files[0])[1]
            self.channels = [self.channel,0]
            modeltype = ('cyto','nuclei')[self.model]
            model = models.Cellpose(gpu=False, model_type=modeltype)
            masks, flows, styles, diams = model.eval(imgs, diameter=None, flow_threshold=self.number1, cellprob_threshold=self.number3 , channels=self.channels)
            io.masks_flows_to_seg(imgs, masks, flows, diams, files, self.channels)
            for f in files:
                npy = f.replace(type,'_seg.npy')
                data, maxkey = get_labels(npy)
                dir = npy.replace('_seg.npy','_map.csv')
                np.savetxt(dir, data, delimiter=",")
                if self.centerShow:
                    centers = np.asarray(get_centers(data, maxkey))
                    dir = npy.replace('_seg.npy','_center.csv')
                    np.savetxt(dir, centers, delimiter=",")
                os.remove(npy)



    def ClickLoad( self, event ):
        #        wcd = 'JPEG files (*.jpg)|*.jpg|PNG files (*.png)|*.png|BMP files (*.bmp)|*.bmp|TIF files (*.tif)|*.tif'
        wcd = 'Images (*.jpg,*.png,*.bmp,*.tif)|*.jpg;*.png;*.bmp;*.tif'

        dlg = wx.FileDialog(self, "Load", "", "",wcd,wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)

        if dlg.ShowModal() == wx.ID_OK:
            # Image is compressed to prevent long processing time
            maxsize =512
            self.pilImg1 = Image.open(dlg.GetPath())
            self.buffer = self.pilImg1
            self.pilImg1fullpath = dlg.GetPath()
            self.pilImg1name = path_leaf(dlg.GetPath(),'name')
            self.pilImg1path = path_leaf(dlg.GetPath(),'dir')
            self.pilImg1name = path_leaf(dlg.GetPath(),'name')
            self.pilImg1type = os.path.splitext(self.pilImg1fullpath)[1]
            if max(self.pilImg1.size[0],self.pilImg1.size[1]) > maxsize:
                print("Compressing...")
                compressed_pathname = self.pilImg1fullpath.replace(self.pilImg1type,'_compressed'+self.pilImg1type)
                compress_ratio = min(maxsize/self.pilImg1.size[0],maxsize/self.pilImg1.size[1])
                print('Compress ratio:', compress_ratio)
                #self.pilImg1 = Image.open(compressed_pathname)
                newsize = (round(self.pilImg1.size[0]*compress_ratio), round(self.pilImg1.size[1]*compress_ratio))
                print(newsize)
                img = self.pilImg1.resize((newsize), Image.ANTIALIAS)
                self.pilImg1 = img
                self.pilImg1fullpath = compressed_pathname
                img.save(compressed_pathname, optimize=True,quality=75)
                print('Compressed image saved')
            self.pilImg2 = self.pilImg1
            self.wxImg1 = wx.Image(self.pilImg1.size[0],self.pilImg1.size[1])
            self.wxImg1.SetData(self.pilImg1.convert("RGB").tobytes(),self.pilImg1.size[0],self.pilImg1.size[1])

            #        wxImg1.setAlphaData(pil.convert("RGBA").tostring()[3::4])

            self.wxImg1 = self.Scale(500,self.wxImg1)
            self.wxSBmp1 = self.wxImg1.ConvertToBitmap()
            self.m_bitmap1.SetBitmap(self.wxSBmp1)
            self.Refresh()
            self.RefreshImage()

            self.m_filename.SetLabel(dlg.GetPath())
            self.m_width.SetLabel(str(self.pilImg1.size[0])+' x'+str(self.pilImg1.size[1]))



    #    def ClickCapture( self, event ):
    #        event.Skip()

    def ClickConvert( self, event ):
        self.pilImg2 = self.pilImg1.convert('L')
        self.RefreshImage()

    def ClickOutline( self, event ):
        # Show outline
        img0 = self.nppilImg1
        if img0.shape[0] < 4:
            img0 = np.transpose(img0, (1,2,0))
        if img0.shape[-1] < 3 or img0.ndim < 3:
            img0 = plot.image_to_rgb(img0, channels=self.channels)
        else:
            if img0.max()<=50.0:
                img0 = np.uint8(np.clip(img0*255, 0, 1))
                self.NewIm2(Image.fromarray(self.masks))
        outlines = utils.masks_to_outlines(self.masks)
        outX, outY = np.nonzero(outlines)
        img0[outX, outY] = np.array([255,75,75])
        self.NewIm2(Image.fromarray(img0))


    def ClickReset( self, event):
        self.NewIm1(self.buffer)
        self.NewIm2(self.buffer)
        self.m_slider1.SetValue(40)
        self.m_slider3.SetValue(50)

    def NewIm2(self, new):
        self.pilImg2 = new
        self.Refresh()
        self.SetBitmap2()

    def NewIm1(self, new):
        self.pilImg1 = new
        self.Refresh()
        self.SetBitmap1()


    def ClickFind( self, event ):
        #segmentation
        modeltype = ('cyto','nuclei')[self.model]
        print(modeltype)
        model = models.Cellpose(gpu=False, model_type=modeltype)
        self.channels = [self.channel,0]
        if self.regionShow:
            image = self.pilImg2
            pathname = self.pilImg1fullpath.replace(self.pilImg1type,'_cropped.tif')
            image.save(pathname)
            self.nppilImg1 = io.imread(pathname)
        else:
            self.nppilImg1 = io.imread(self.pilImg1fullpath)
        print(self.number1, self.number3)
        self.masks, self.flows, styles, self.diams = model.eval(self.nppilImg1, diameter=None, flow_threshold=self.number1, cellprob_threshold=self.number3 , channels=self.channels)
        io.masks_flows_to_seg(self.nppilImg1, self.masks, self.flows, self.diams, self.pilImg1fullpath, self.channels)
        dir = self.pilImg1fullpath.replace(self.pilImg1type,'_seg.npy')

        self.data, self.maxkey = get_labels(dir, region = None)
        self.m_height.SetLabel(str(self.maxkey) + '  (avg. Diameter in pixels: '+ str(round(self.diams)) + ')')
        os.remove(dir)
        if self.centerShow:
            self.centers = get_centers(self.data,self.maxkey)
        newbox=wx.MessageDialog(None,'Results succesfully saved','Notice',wx.OK)
        answer=newbox.ShowModal()
        newbox.Destroy()

    def ClickCenter( self, event):
        # Calculate center locations
        self.ShowCenter()



    def ClickGenerate( self, event ):
        # Show cellpose
        self.NewIm2(Image.fromarray(self.flows[0]))

    def ClickSaveImage( self, event ):
        pathname = self.pilImg1fullpath.replace(self.pilImg1type,'_processed.png')
        cv2.imwrite(pathname, np.asarray(self.pilImg2))
        io.save_to_png(self.nppilImg1, self.masks, self.flows, self.pilImg1fullpath)
        print('Mask successfully saved as ', pathname)

    def CenterLocations(self):
        centers = []
        for i in range(self.maxkey):
            locations = np.where(self.data == i+1)
            center = locations.mean(axis=0)
            centers.append(center)
        return zip(*centers.astype(int))

    def ClickSavePoints( self, event ):
        # Export center locations
        centers = np.asarray(self.centers)
        if self.regionShow:
            region = self.scanregion
            centers = [centers[i] for i,_ in enumerate(centers) if centers[i][0]>region[0] and centers[i][0]<region[2] and centers[i][1]>region[1] and centers[i][i]<region[3]]
        dir = self.pilImg1fullpath.replace(self.pilImg1type,'_center.csv')
        np.savetxt(dir, centers, delimiter=",")
        print('Successfully found the centers')

    def ClickSaveMap (self, event):
        dir = self.pilImg1fullpath.replace(self.pilImg1type,'_map.csv')
        data = [centers[i] for i,_ in enumerate(centers) if centers[i][0]>region[0] and centers[i][0]<region[2] and centers[i][1]>region[1] and centers[i][i]<region[3]]
        np.savetxt(dir, self.data, delimiter=",")


    def OnChangeSlider1( self, event ):
        self.number1 = self.m_slider1.GetValue()/100


    def OnChangeSlider3( self, event ):
        self.number3 = (self.m_slider3.GetValue()-50)/100

    def Scale(self, factor, wxImage):

        MaxImageSize = factor
        W = wxImage.GetWidth()
        H = wxImage.GetHeight()

        if W > H:
            NewW = MaxImageSize
            NewH = MaxImageSize * H / W
        else:
            NewH = MaxImageSize
            NewW = MaxImageSize * W / H
        wxImage = wxImage.Scale(int(NewW),int(NewH))

        return wxImage

    def RefreshImage(self):

        enhancer1 = ImageEnhance.Contrast(self.pilImg2)
        ehImg1 = enhancer1.enhance(1)
        enhancer2 = ImageEnhance.Brightness(ehImg1)
        ehImg2 = enhancer2.enhance(1)

        self.wxImg2.SetData(ehImg2.convert("RGB").tobytes(),ehImg2.size[0],ehImg2.size[1])
        self.wxImg2 = self.Scale(500,self.wxImg2)
        self.wxSBmp2 = self.wxImg2.ConvertToBitmap()
        self.m_bitmap2.SetBitmap(self.wxSBmp2)
        self.Refresh()

    def SetBitmap2(self):
        self.wxImg2 = wx.Image(self.pilImg2.size[0],self.pilImg2.size[1])
        self.wxImg2.SetData(self.pilImg2.convert("RGB").tobytes(),self.pilImg2.size[0],self.pilImg2.size[1])
        self.wxImg2 = self.Scale(500,self.wxImg2)
        self.wxSBmp2 = self.wxImg2.ConvertToBitmap()
        self.m_bitmap2.SetBitmap(self.wxSBmp2)

    def SetBitmap1(self):
        self.wxImg1 = wx.Image(self.pilImg1.size[0],self.pilImg1.size[1])
        self.wxImg1.SetData(self.pilImg1.convert("RGB").tobytes(),self.pilImg1.size[0],self.pilImg1.size[1])
        self.wxImg1 = self.Scale(500,self.wxImg1)
        self.wxSBmp1 = self.wxImg1.ConvertToBitmap()
        self.m_bitmap1.SetBitmap(self.wxSBmp1)

###########################################################################
## Class MyDialog1
###########################################################################

class MyDialog1 ( wx.Dialog ):

    def __init__( self, parent ):
        wx.Dialog.__init__ ( self, parent, id = wx.ID_ANY, title = 'Scanning region', pos = wx.DefaultPosition, size = wx.Size( 350,174 ), style = wx.DEFAULT_DIALOG_STYLE )

        self.SetSizeHints( wx.DefaultSize, wx.DefaultSize )

        self.x0 = 0
        self.x1 = 0
        self.y0 = 0
        self.y1 = 0
        self.OK = False

        bSizer1 = wx.BoxSizer( wx.VERTICAL )

        bSizer2 = wx.BoxSizer( wx.HORIZONTAL )

        self.m_staticText1 = wx.StaticText( self, wx.ID_ANY, u"x0 :", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.m_staticText1.Wrap( -1 )
        bSizer2.Add( self.m_staticText1, 0, wx.ALL, 5 )

        self.m_textCtrl1 = wx.TextCtrl( self, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, 0 )
        bSizer2.Add( self.m_textCtrl1, 0, wx.ALL, 5 )

        self.m_staticText2 = wx.StaticText( self, wx.ID_ANY, u", y0 :", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.m_staticText2.Wrap( -1 )
        bSizer2.Add( self.m_staticText2, 0, wx.ALL, 5 )

        self.m_textCtrl2 = wx.TextCtrl( self, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, 0 )
        bSizer2.Add( self.m_textCtrl2, 0, wx.ALL, 5 )


        bSizer1.Add( bSizer2, 1, wx.EXPAND, 5 )

        bSizer3 = wx.BoxSizer( wx.HORIZONTAL )

        self.m_staticText3 = wx.StaticText( self, wx.ID_ANY, u"x1 :", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.m_staticText3.Wrap( -1 )
        bSizer3.Add( self.m_staticText3, 0, wx.ALL, 5 )

        self.m_textCtrl3 = wx.TextCtrl( self, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, 0 )
        bSizer3.Add( self.m_textCtrl3, 0, wx.ALL, 5 )

        self.m_staticText4 = wx.StaticText( self, wx.ID_ANY, u", y1 :", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.m_staticText4.Wrap( -1 )
        bSizer3.Add( self.m_staticText4, 0, wx.ALL, 5 )

        self.m_textCtrl4 = wx.TextCtrl( self, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, 0 )
        bSizer3.Add( self.m_textCtrl4, 0, wx.ALL, 5 )


        bSizer1.Add( bSizer3, 1, wx.EXPAND, 5 )

        bSizer4 = wx.BoxSizer( wx.HORIZONTAL )


        self.m_button1 = wx.Button( self, wx.ID_OK, u"OK", wx.Point( -1,-1 ), wx.DefaultSize, 0 )
        bSizer4.AddStretchSpacer()
        bSizer4.Add( self.m_button1, 0, wx.CENTER )
        bSizer4.AddStretchSpacer()

        self.m_button2 = wx.Button( self, wx.ID_CANCEL, u"Cancel", wx.Point( -1,-1 ), wx.DefaultSize, 0 )
        bSizer4.Add( self.m_button2, 0, wx.CENTER )
        bSizer4.AddStretchSpacer()


        bSizer1.Add( bSizer4, 1, wx.EXPAND, 5 )



        self.SetSizer( bSizer1 )
        self.Layout()

        self.Centre( wx.BOTH )

        # Connect Events
        self.m_textCtrl1.Bind( wx.EVT_TEXT, self.leftupx )
        self.m_textCtrl2.Bind( wx.EVT_TEXT, self.leftupy )
        self.m_textCtrl3.Bind( wx.EVT_TEXT, self.rightlowx )
        self.m_textCtrl4.Bind( wx.EVT_TEXT, self.rightlowy )
        self.m_button1.Bind( wx.EVT_BUTTON, self.ClickOK )
        self.m_button2.Bind( wx.EVT_BUTTON, self.ClickCancel )



    def __del__( self ):
        pass

    def rightlowx( self, event ):
        self.x1 = self.m_textCtrl3.GetValue()

    def rightlowy( self, event ):
        self.y1 = self.m_textCtrl4.GetValue()


    def leftupx( self, event ):
        self.x0 = self.m_textCtrl1.GetValue()


    def leftupy( self, event ):
        self.y0 = self.m_textCtrl2.GetValue()


    def ClickOK( self, event ):
        #print(self.x0,self.y0,self.x1,self.y1)
        self.EndModal(wx.ID_OK)

    def ClickCancel( self, event ):
        self.x0 = 0
        self.x1 = 0
        self.y0 = 0
        self.y1 = 0
        self.EndModal(wx.ID_CANCEL)

    def Get_value(self):
        try:
            x0,y0,x1,y1 = int(self.x0), int(self.y0), int(self.x1), int(self.y1)
            return [x0,y0,x1,y1]
        except ValueError:
            warning = wx.MessageBox('Invalid entry, please retry', 'Warning', wx.OK | wx.ICON_WARNING)


if __name__ == '__main__':
    app = wx.App(False)

    frame = MyFrame1(parent=None)
    frame.Show()

    app.MainLoop()
