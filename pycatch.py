from pyautogui import position,press,screenshot,moveTo,moveRel,click
from numpy import array,sum,argmax
from random import randrange,seed,uniform
from time import time,sleep
import cv2
from imutils import resize,grab_contours
from os import system,path
import argparse
from tkinter import *


# initiate the parser
text="PyCatch - By:FJLJ, is a program that fishes smartly for you, really... It uses machine learning to find the bobber and motion detection for the bite! Run program, center mouse in detection field wait for the cast. If not using afk mode... after a cast feel free to move your mouse around. Default KeyBindings Cast=Q  (optional)Bauble=Num9"
parser = argparse.ArgumentParser(description = text)

# add long and short argument
parser.add_argument("-n","--noafkmode", help="Mouse moves only when caught.(use if not AFK)", action="store_true")
parser.add_argument("-s","--sensitivity", help="Set bite sensitivity, lower=more sensitive. default:45000")
parser.add_argument("-c", "--casts", help="Number of casts to bot. default:random 170-350")
parser.add_argument("-k","--keys", help="List supported keys", action="store_true")
parser.add_argument("-cb","--cast_binding", help="Key toq use to cast. Default:q")
parser.add_argument("-bb","--bauble_binding", help="Key to use bauble macro. Default:num9")
parser.add_argument("-d","--detection_boxes", help="Draw detection boxes over game window.", action="store_true")
#parser.add_argument("-db","--detection_boxes", help="Draw detection boxes over game window.")

args = parser.parse_args()
go = 1

if args.detection_boxes:
	root = Tk()
	root.attributes("-transparentcolor","white")
	root.attributes("-topmost", True)
	root.state('zoomed')
	root.overrideredirect(True)
	#root.withdraw()
	myCanvas = Canvas(root, bg="white", height=root.winfo_height()-60, width=root.winfo_width(),highlightthickness=0)
	boxp = myCanvas.create_rectangle(0, 0, root.winfo_width(), root.winfo_height()-60, fill="white",width=0)
	myCanvas.pack(fill = BOTH, expand = True, ipady = 0,ipadx = 0)
	box = myCanvas.create_rectangle(0, 0, 910, 460, fill="white",width=3,outline='green')
	myCanvas.pack()
	myCanvas.itemconfigure(box, state='hidden')
	#root.deiconify()
	root.update()

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

if args.keys:
  #os.system("mode con: cols=80 lines=25")
  vkeys =  ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'fn', 'up']
  vkeys2 = ['add', 'alt', 'del', 'end', 'esc', 'f10', 'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19', 'f20', 'f21', 'f22', 'f23', 'f24', 'tab', 'win', 'yen','apps', 'ctrl', 'down', 'help', 'home', 'kana', 'left', 'num0', 'num1', 'num2', 'num3', 'num4', 'num5', 'num6', 'num7', 'num8', 'num9', 'pgdn', 'pgup', 'stop', 'clear', 'enter', 'final', 'hanja', 'junja', 'kanji', 'pause']
  vkeys3 = ['print', 'prtsc', 'right', 'shift', 'sleep', 'space', 'accept', 'delete', 'divide', 'escape', 'hangul', 'insert', 'pageup', 'prtscr', 'return', 'select', 'option', 'altleft', 'convert', 'decimal', 'execute', 'hanguel', 'numlock', 'winleft', 'command', 'altright', 'capslock']
  vkeys4 = [ 'ctrlleft','multiply', 'pagedown', 'prntscrn', 'subtract', 'volumeup', 'winright', 'backspace', 'ctrlright', 'nexttrack', 'playpause', 'prevtrack', 'separator', 'shiftleft', 'launchapp1', 'launchapp2', 'launchmail', 'modechange', 'nonconvert', 'scrolllock', 'shiftright', 'volumedown', 'volumemute', 'optionleft', 'browserback', 'browserhome', 'browserstop', 'printscreen', 'optionright', 'browsersearch', 'browserforward', 'browserrefresh', 'browserfavorites', 'launchmediaselect']
  print("\nValid keys for bindings.\nhttps://pyautogui.readthedocs.io/en/latest/keyboard.html#keyboard-keys\n")
  for l in chunker(sorted(vkeys,key=len),20):
    print(",".join(l))
  for l in chunker(sorted(vkeys2,key=len),10):
    print(",".join(l))
  for l in chunker(sorted(vkeys3,key=len),7):
    print(",".join(l))
  for l in chunker(sorted(vkeys4,key=len),5):
    print(",".join(l))
  go = 0

def usleep(ms):
    sleepn = ms
    sleepitr = ms/20.0
    while sleepn > 0.0:
      if args.detection_boxes:
        root.update()
      sleepn-=sleepitr
      sleep(sleepitr)
  
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = path.abspath(".")

    return path.join(base_path, relative_path)

def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers

CHUNK = 2**11
clickc = False
itrs = 0
roundn = 0
scale = 0.01392
classes = ["Bobber"]
bobsw = resource_path("data/bobber.weights")
bobsc = resource_path("data/bobber.cfg")
net = cv2.dnn.readNet(bobsw, bobsc)

ck = args.cast_binding if args.cast_binding else "q"
sensitivity = int(args.cast_binding) if args.cast_binding else 45000
bk = args.bauble_binding if args.bauble_binding else "num9"


lastrands=[time(),randrange(170,350)]
if args.casts:
  tar = int(args.casts)
else:
  tar = lastrands[-1]

if go == 1:
  system('cls & title PyCatch - By:FJLJ')
ft = "--------------PyCatch--------------\nThe Smart WOW Fishing Bot - By:fjlj" 
if args.noafkmode:
  ft += "\nPyCatch.py -h for help.\nCast=%s, Bauble=%s\n\nWaiting 5 seconds to Begin.\n(place cursor in center of detection field)\n"
else:
  ft += "(AFK Mode)\nPyCatch.py -h for help.\nCast=%s, Bauble=%s\n\nWaiting 5 seconds to Begin.\n(place cursor in center of detection field)\n"
ot = "--------------PyCatch--------------\nThe Smart WOW Fishing Bot - By:fjlj"
if args.noafkmode:
  ot += "\nCast=%s, Bauble=%s\n\n"
else:
  ot += "(AFK Mode)\nCast=%s, Bauble=%s\n"
if go ==1:
  print(ft%(ck,bk))
  sleep(5)
  x, y = position()
while roundn < tar and go == 1:
 if(roundn != 0):
   system('cls')
   print(ot%(ck,bk))
 lastrands.append(len(lastrands))
 seed(tuple(lastrands))
 roundn += 1
 print("Cast %d/%d - RandSeeds: %d"%(roundn,tar,len(lastrands)))
 del lastrands[:]  
 press(ck)
 lastrands.append(uniform(0.75,1.86))

 if args.noafkmode == False: 
   x, y = position()
 #"t2\\%06d.jpg"%(roundn),
 clampx = (x-455 if x-455 > 0 else 0)
 clampy = (y-230 if y-230 > 0 else 0)
 #root.geometry("%dx%d+%d+%d"%(910,460,clampx,clampy))
 # myCanvas.config(width=910, height=460)
 if args.detection_boxes:
    myCanvas.coords(box,clampx,clampy,clampx+910,clampy+460)
    myCanvas.itemconfigure(box, state='normal')
    root.update()
 # myCanvas.pack()
 #root.deiconify()
 #root.update()
 usleep(round(lastrands[-1], 2))
 #root.update()
 image = screenshot(region=(x-450,y-225,900,450))
 #root.update()
 open_cv_image = array(image)
 image = open_cv_image[:, :, ::-1].copy()
 
 Width = image.shape[1]
 Height = image.shape[0]
 
 blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)
 net.setInput(blob)
 outs = net.forward(get_output_layers(net))
 center_x = 0
 center_y = 0
 for out in outs:
    if args.detection_boxes:
        root.update()
    for detection in out:
        if args.detection_boxes:
            root.update()
        scores = detection[5:]
        class_id = argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.4:
            center_x = int(detection[0] * Width)
            center_y = int(detection[1] * Height)
 if args.detection_boxes:
    root.update()
 minLoc = (round(center_x),round(center_y))
 lastrands.append(minLoc)
 print("Bobber Detected: (%d,%d)"%minLoc)
 newx = x
 newy = y
 if(minLoc != (0.0,0.0) or args.noafkmode):
   lastrands.append(randrange(445,455))
   newx += (minLoc[0]-lastrands[-1])
   lastrands.append(randrange(215,225))
   newy += (minLoc[1]-lastrands[-1])
   if args.noafkmode == False:
     print("Distance to move: (%d,%d)"%(abs(newx-x),abs(newy-y)))
   if(abs(newx-x) > 25 or abs(newy-y) > 15):
    lastrands.append(uniform(0.15,0.85))
    usleep(round(lastrands[-1], 2))
    lastrands.append(uniform(0.12,0.52))
    if args.noafkmode == False:
      moveTo(newx,newy, round(lastrands[-1], 2))
 if args.detection_boxes:
    myCanvas.itemconfigure(box, state='hidden')
    root.update()
 firstFrame = None
 if args.detection_boxes:
    doff = randrange(2,5)
    dcalc = ((root.winfo_height()-60)/(newy+0.01*0.66))
    clx = (newx-50)+(dcalc*doff) if (newx-50)+(dcalc*doff) > 0 else 0
    cly = (newy-50)+(dcalc*doff) if (newy-50)+(dcalc*doff) > 0 else 0
 while clickc == False: 
   if args.detection_boxes:
     myCanvas.coords(box,clx,cly,clx+100-(dcalc*doff*2),cly+100-(dcalc*doff*2))
     myCanvas.itemconfigure(box, state='normal')
     root.update()
   
   im2 = screenshot(region=(newx-40,newy-40,80,80))
   open_cv_im2 = array(im2)
   frame = open_cv_im2[:, :, ::-1].copy()
   frame = resize(frame, width=500)
   #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   w = array([[[ 0.95, 0.02,  1.15]]])
   #root.update()
   gray = cv2.convertScaleAbs(sum(frame*w, axis=2))
   gray = cv2.GaussianBlur(gray, (23, 23), 0)
   #root.update()
   if firstFrame is None:
     firstFrame = gray
     continue
   frameDelta = cv2.absdiff(firstFrame, gray)
   thresh = cv2.threshold(frameDelta, 23, 255, cv2.THRESH_BINARY)[1]
   #root.update()
   thresh = cv2.erode(thresh, None, iterations=19)
   thresh = cv2.dilate(thresh, None, iterations=65)
   #thresh = cv2.dilate(thresh, (3,3,3), iterations=1)
   #root.update()
   cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
   cnts = grab_contours(cnts)
   #root.update()
   if len(cnts) <= 3 and len(cnts) > 0:
     #cv2.imwrite("itrs/cast_t"+str(roundn)+"_"+str(itrs)+".jpg", thresh)
     for c in cnts:
       #root.update()
       if cv2.contourArea(c) < len(cnts)*sensitivity:
         continue
       clickc = True
       (x1, y1, w1, h1) = cv2.boundingRect(c)
       cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)
       #cv2.imwrite("cast_t"+str(roundn)+".jpg", thresh)
   itrs += 1
   #root.update()
   lastrands.append(randrange(800,1000))
   if clickc == True or itrs > lastrands[-1]:
     clickc = True
     lastrands.append(int(len(cnts)))
     print("ChangesDetected:%d"%len(cnts))
 if args.detection_boxes:
    myCanvas.itemconfigure(box, state='hidden')
    root.update()  
 clickc = False
 lastrands.append(randrange(0,100))
 itrs = lastrands[-1]
 lastrands.append(uniform(0.05,0.75)+0.15)
 if args.noafkmode == False:
   usleep(round(lastrands[-1], 2))
 else:
   moveTo(newx,newy, round(lastrands[-1], 2))
 click(button='right')
 lastrands.append(uniform(2.45,3.85))
 lastrands.append(uniform(2.45,3.85))
 if roundn%35 == 0:
  usleep(round(lastrands[-1], 2))
  press(bk)
  lastrands.append(randrange(8,12))
  usleep(round(lastrands[-1], 2))
 usleep(round(lastrands[-2], 2))
 lastrands.append(randrange(-70,50))
 lastrands.append(randrange(-25,5))
 lastrands.append(uniform(0.15,0.47))
 if args.noafkmode == False:
   moveRel(lastrands[-3],lastrands[-2],round(lastrands[-1],2))
 