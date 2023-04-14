import os
import sys
import argparse
import cv2
import numpy as np
prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

from lib.test.evaluation import Tracker


def run_video(tracker_name, tracker_param, videofile, optional_box=None, debug=None, save_results=False):
    """Run the tracker on your webcam.
    args:
        tracker_name: Name of tracking method.
        tracker_param: Name of parameter file.
        debug: Debug level.
    """
    tracker = Tracker(tracker_name, tracker_param, "video")
    tracker.run_video(videofilepath=videofile, optional_box=optional_box, debug=debug, save_results=save_results)
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
    test_path='/media/zjy/4TB/data/SAM_coarse_data/box2mask2'
    names=os.listdir(test_path)
    for name in names:
        tracker_param='vitb_384_mae_ce_32x4_ep300'
        imgdir='/media/zjy/4TB/data/DAVIS2017/DAVIS-2017-trainval-480p/JPEGImages/480p/'+name

        color_list=[(0,0,128),(0,128,0),(0,128,128),(128,0,0),(128,0,128)]
        for i in range(len(color_list)):
            gtmask = cv2.imread(
                '/media/zjy/4TB/data/DAVIS2017/DAVIS-2017-trainval-480p/Annotations/480p/' + name + '/00000.png')
            gtmask = (gtmask == color_list[i])
            # 顺序第一种mask：0,0,128
            # 顺序第二种mask：0,128,0
            # 顺序第三种mask：0,128,128
            # 顺序第四种mask：128,0,0
            # 顺序第五种mask：128,0,128
            gtmask = gtmask[:, :, 0] & gtmask[:, :, 1] & gtmask[:, :, 2]
            if (True in gtmask):
                index = np.where(gtmask == True)
                xx=index[0].min()
                yy=index[1].min()
                ww=index[0].max()-xx
                hh=index[1].max()-yy
                run_imgdir(args.tracker_name,tracker_param, imgdir, [yy,xx,hh,ww], args.debug, args.save_results,index=[name,i,color_list[i]])
            else:
                print('no this color!')

if __name__ == '__main__':
    main()
