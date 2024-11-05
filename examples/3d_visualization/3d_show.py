#!/usr/bin/env python3

import sys
sys.path.append("../..")
from HandTrackerRenderer import HandTrackerRenderer
from Filters import LandmarksSmoothingFilter
import argparse
import numpy as np
import cv2
from o3d_utils import Visu3D
import fcntl
import termios
import sys
import os

n_1 = 0
n_2 = 0

LINES_HAND = [[0,1],[1,2],[2,3],[3,4], 
            [0,5],[5,6],[6,7],[7,8],
            [5,9],[9,10],[10,11],[11,12],
            [9,13],[13,14],[14,15],[15,16],
            [13,17],[17,18],[18,19],[19,20],[0,17]]

class HandTracker3DRenderer:
    def __init__(self, tracker, mode_3d="image", smoothing=True):
        print(mode_3d)

        self.tracker = tracker
        self.mode_3d = mode_3d
        if self.mode_3d == "mixed" and not self.tracker.xyz:
            print("'mixed' 3d visualization needs the tracker to be in 'xyz' mode !")
            print("3d visualization falling back to 'world' mode.")
            self.mode_3d = 'world'
        if self.mode_3d == "image":
            self.vis3d = Visu3D(zoom=0.7, segment_radius=10)
            z = min(tracker.img_h, tracker.img_w)/3
            self.vis3d.create_grid([0,tracker.img_h,-z],[tracker.img_w,tracker.img_h,-z],[tracker.img_w,tracker.img_h,z],[0,tracker.img_h,z],5,2) # Floor
            self.vis3d.create_grid([0,0,z],[tracker.img_w,0,z],[tracker.img_w,tracker.img_h,z],[0,tracker.img_h,z],5,2) # Wall
            self.vis3d.init_view()
        elif "world" in self.mode_3d:
            self.vis3d = Visu3D(bg_color=(0.2, 0.2, 0.2), zoom=1.1, segment_radius=0.01)
            x_max = 0.2 if self.tracker.solo else 0.4
            y_max = 0.2
            z_max = 0.2
            self.vis3d.create_grid([-x_max,y_max,-z_max],[x_max,y_max,-z_max],[x_max,y_max,z_max],[-x_max,y_max,z_max],1 if self.tracker.solo else 2,1) # Floor
            self.vis3d.create_grid([-x_max,y_max,z_max],[x_max,y_max,z_max],[x_max,-y_max,z_max],[-x_max,-y_max,z_max],1 if self.tracker.solo else 2,1) # Wall
            self.vis3d.init_view()
        elif self.mode_3d == "mixed":
            self.vis3d = Visu3D(bg_color=(0.4, 0.4, 0.4), zoom=0.8, segment_radius=0.01)
            x_max = 0.9
            y_max = 0.6
            grid_depth = 2
            self.vis3d.create_grid([-x_max,y_max,0],[x_max,y_max,0],[x_max,y_max,grid_depth],[-x_max,y_max,grid_depth],2,grid_depth) # Floor
            self.vis3d.create_grid([-x_max,y_max,grid_depth],[x_max,y_max,grid_depth],[x_max,-y_max,grid_depth],[-x_max,-y_max,grid_depth],2,2) # Wall
            self.vis3d.create_camera()
            self.vis3d.init_view()

        self.smoothing = smoothing
        self.filter = None
        if self.smoothing:
            if tracker.solo:
                if self.mode_3d == "image":
                    self.filter = [LandmarksSmoothingFilter(min_cutoff=0.01, beta=40, derivate_cutoff=1)]
                else:
                    self.filter = [LandmarksSmoothingFilter(min_cutoff=1, beta=20, derivate_cutoff=10, disable_value_scaling=True)]
            else:
                if self.mode_3d == "image":
                    self.filter = [
                        LandmarksSmoothingFilter(min_cutoff=0.01,beta=40,derivate_cutoff=1),
                        LandmarksSmoothingFilter(min_cutoff=0.01,beta=40,derivate_cutoff=1)
                        ]
                else:
                    self.filter = [
                        LandmarksSmoothingFilter(min_cutoff=1, beta=20, derivate_cutoff=10, disable_value_scaling=True),
                        LandmarksSmoothingFilter(min_cutoff=1, beta=20, derivate_cutoff=10, disable_value_scaling=True)
                        ]

        self.nb_hands_in_previous_frame = 0

    def draw_hand(self, hand, i):
        if self.mode_3d == "image":
            # Denormalize z-component of 'norm_landmarks'
            lm_z = (hand.norm_landmarks[:,2:3] * hand.rect_w_a  / 0.4).astype(np.int32)
            # ... and concatenates with x and y components of 'landmarks'
            points = np.hstack((hand.landmarks, lm_z))
            radius = hand.rect_w_a / 30 # Thickness of segments depends on the hand size
        elif "world" in self.mode_3d:
            if self.mode_3d == "raw_world":
                points = hand.world_landmarks
            else: # "world"
                points = hand.get_rotated_world_landmarks()
            if not self.tracker.solo:
                delta_x = -0.2 if  hand.label == "right" else 0.2
                points = points + np.array([delta_x,0,0])
            radius = 0.01
        elif self.mode_3d == "mixed":
            wrist_xyz = hand.xyz / 1000.0
            # Beware that y value of (x,y,z) coordinates given by depth sensor is negative 
            # in the lower part of the image and positive in the upper part.
            wrist_xyz[1] = -wrist_xyz[1]
            points = hand.get_rotated_world_landmarks()
            points = points + wrist_xyz - points[0]
            radius = 0.01

        if self.smoothing:
            points = self.filter[i].apply(points, object_scale=hand.rect_w_a)

        for i,a_b in enumerate(LINES_HAND):
            a, b = a_b
            self.vis3d.add_segment(points[a], points[b], radius=radius, color=[1*(1-hand.handedness),hand.handedness,0]) # if hand.handedness<0.5 else [0,1,0])
                    
    def draw(self, hands):
        if self.smoothing and len(hands) != self.nb_hands_in_previous_frame:
            for f in self.filter: f.reset()
        self.vis3d.clear()
        self.vis3d.try_move()
        self.vis3d.add_geometries()
        for i, hand in enumerate(hands):
            self.draw_hand(hand, i)
        self.vis3d.render()
        self.nb_hands_in_previous_frame = len(hands)
    
    def to_pixel(self, finger, im_size, num_node):
        max_size = 2.0
        finger_pixel = np.zeros((num_node, 2), dtype=int)
        for i in range(num_node):
            x = finger[0, i] * 10 + max_size / 2
            y = finger[1, i] * 10 + max_size / 2
            finger_pixel[i, 0] = int(x / max_size * im_size)
            finger_pixel[i, 1] = int(y / max_size * im_size)
        return finger_pixel
    
    def draw_third(self, hand):
        hand_nodes = hand.world_landmarks
        num_node = 21
        im_size = 500
        key_0 = getkey()
        global n_1
        global n_2
        #print(key_0)
        if key_0 == 1792833:
            if n_1 != 9:
                n_1 += 1
        if key_0 == 1792834:
            if n_1 != -9:
                n_1 -= 1
        if key_0 == 1792835:
            n_2 += 1
            if n_2 == 19:
                n_2 = -17
        if key_0 == 1792836:
            n_2 -= 1
            if n_2 == -18:
                n_2 = 18
        Sita_1 = n_1 * np.pi / 18
        Sita_2 = n_2 * np.pi / 18
        #print(n_2)
        M_1 = np.array([[np.cos(Sita_2), 0, -np.sin(Sita_2)], [0, 1, 0], [np.sin(Sita_2), 0, np.cos(Sita_2)]])
        M_2 = np.array([[1, 0, 0], [0, np.cos(Sita_1), -np.sin(Sita_1)], [0, np.sin(Sita_1), np.cos(Sita_1)]])
        rot_finger = M_2 @ M_1 @ hand_nodes.T
        finger_pixel = self.to_pixel(rot_finger, im_size, num_node)
        return finger_pixel, im_size, num_node

def getkey():
    fno = sys.stdin.fileno()

    #stdinの端末属性を取得
    attr_old = termios.tcgetattr(fno)

    # stdinのエコー無効、カノニカルモード無効
    attr = termios.tcgetattr(fno)
    attr[3] = attr[3] & ~termios.ECHO & ~termios.ICANON # & ~termios.ISIG
    termios.tcsetattr(fno, termios.TCSADRAIN, attr)

    # stdinをNONBLOCKに設定
    fcntl_old = fcntl.fcntl(fno, fcntl.F_GETFL)
    fcntl.fcntl(fno, fcntl.F_SETFL, fcntl_old | os.O_NONBLOCK)

    chr = 0

    try:
        # キーを取得
        c = sys.stdin.read(1)
        if len(c):
            while len(c):
                chr = (chr << 8) + ord(c)
                c = sys.stdin.read(1)
                #up:1792833
                #down:1792834
                #right:1792835
                #left:1792836
    finally:
        # stdinを元に戻す
        fcntl.fcntl(fno, fcntl.F_SETFL, fcntl_old)
        termios.tcsetattr(fno, termios.TCSANOW, attr_old)

    return chr

parser = argparse.ArgumentParser()
# parser.add_argument('-e', '--edge', action="store_true",
#                     help="Use Edge mode (postprocessing runs on the device)")
parser_tracker = parser.add_argument_group("Tracker arguments")
parser_tracker.add_argument('-i', '--input', type=str, 
                    help="Path to video or image file to use as input (if not specified, use OAK color camera)")
parser_tracker.add_argument("--pd_model", type=str,
                    help="Path to a blob file for palm detection model")
parser_tracker.add_argument("--lm_model", type=str,
                    help="Path to a blob file for landmark model")
parser_tracker.add_argument('-s', '--solo', action="store_true", 
                    help="Detect one hand max")         
parser_tracker.add_argument('-f', '--internal_fps', type=int, 
                    help="Fps of internal color camera. Too high value lower NN fps (default= depends on the model)")                    
parser_tracker.add_argument('--internal_frame_height', type=int,                                                                                 
                    help="Internal color camera frame height in pixels")    
parser_tracker.add_argument('--single_hand_tolerance_thresh', type=int, default=0,
                    help="(Duo mode only) Number of frames after only one hand is detected before calling palm detection (default=%(default)s)")                

parser_renderer3d = parser.add_argument_group("3D Renderer arguments")
parser_renderer3d.add_argument('-m', '--mode_3d', nargs='?', 
                    choices=['image', 'world', 'raw_world', 'mixed'], const='image', default='mixed',
                    help="Specify the 3D coordinates used. See README for description (default=%(default)s)")
parser_renderer3d.add_argument('--no_smoothing', action="store_true", 
                    help="Disable smoothing filter (smoothing works only in solo mode)")   
args = parser.parse_args()

args.edge = True
if args.edge:
    from HandTrackerEdge import HandTracker
else:
    from HandTracker import HandTracker

dargs = vars(args)
tracker_args = {a:dargs[a] for a in ['pd_model', 'lm_model', 'internal_fps', 'internal_frame_height'] if dargs[a] is not None}

tracker = HandTracker(
        input_src=args.input, 
        use_world_landmarks=args.mode_3d != "image",
        solo=args.solo,
        xyz= args.mode_3d == "mixed",
        stats=True,
        single_hand_tolerance_thresh=args.single_hand_tolerance_thresh,
        lm_nb_threads=1,
        **tracker_args
        )

renderer3d = HandTracker3DRenderer(tracker, mode_3d=args.mode_3d, smoothing=not args.no_smoothing)
renderer2d = HandTrackerRenderer(tracker)

pause = False
hands = []

while True:
    # Run hand tracker on next frame
    if not pause:
        frame, hands, bag = tracker.next_frame()
        if frame is None: break
        # Render 2d frame
        frame = renderer2d.draw(frame, hands, bag)
        cv2.imshow("HandTracker", frame)
        for k, hand in enumerate(hands):
            if k == 0:
                finger_pixel, im_size, num_node = renderer3d.draw_third(hand)
                third_draw = np.zeros((im_size, im_size, 3), dtype=np.uint8)
                num_edge = 20
                for i in range(num_node):
                    cv2.circle(third_draw, (finger_pixel[i, 0], finger_pixel[i, 1]), 5, (0, 255, 0), -1)
                for i in range(num_edge):
                    r = i % 4
                    if r == 0:
                        j_0 = 0
                    else:
                        j_0 = i
                    j_1 = i + 1
                    cv2.line(third_draw, (finger_pixel[j_0, 0], finger_pixel[j_0, 1]), (finger_pixel[j_1, 0], finger_pixel[j_1, 1]), (0, 255, 0), 3)
                cv2.imshow('3D model', third_draw)
    key = cv2.waitKey(1)
    # Draw hands on open3d canvas
    renderer3d.draw(hands)
    if key == 27 or key == ord('q'):
        break
    elif key == 32: # space
        pause = not pause
    elif key == ord('s'):
        if renderer3d.filter:
            renderer3d.smoothing = not renderer3d.smoothing
renderer2d.exit()
tracker.exit()
