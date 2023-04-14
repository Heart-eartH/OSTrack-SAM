import os
import sys
import argparse
import cv2
import numpy as np
prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

from lib.test.evaluation import Tracker


def run_imgdir(tracker_name, tracker_param, imgdir, optional_box=None, debug=None, save_results=False,index=None):
    """Run the tracker on your webcam.
    args:
        tracker_name: Name of tracking method.
        tracker_param: Name of parameter file.
        debug: Debug level.
    """
    tracker = Tracker(tracker_name, tracker_param, "video")

    tracker.run_imgdir(videofilepath=imgdir, optional_box=optional_box, debug=debug, save_results=save_results,index=index)


def main():
    parser = argparse.ArgumentParser(description='Run the tracker on your webcam.')
    parser.add_argument('tracker_name', type=str,help='Name of tracking method.')

    parser.add_argument('--optional_box', type=float, default=None, nargs="+", help='optional_box with format x y w h.')
    parser.add_argument('--debug', type=int, default=0, help='Debug level.')
    parser.add_argument('--save_results', dest='save_results', action='store_true', help='Save bounding boxes')
    parser.set_defaults(save_results=False)

    args = parser.parse_args()

    tracker_param='vitb_384_mae_ce_32x4_ep300'
    imgdir=''
    while True:
        panduan=input('b for box, p for point:')
        if (panduan=='b' or panduan=="B" or panduan=='p' or panduan=="P"):
            break
    if (panduan=='b' or panduan=='B'):
        run_imgdir(args.tracker_name,tracker_param, imgdir, args.optional_box, args.debug, args.save_results,index=(0,128,0))
    else:
        run_imgdir(args.tracker_name, tracker_param, imgdir, False, args.debug, args.save_results,index=(0,128,0))


if __name__ == '__main__':
    main()