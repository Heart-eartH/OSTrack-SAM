import importlib
import os
from collections import OrderedDict
from lib.test.evaluation.environment import env_settings
import time
import cv2 as cv

from lib.utils.lmdb_utils import decode_img
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import build_sam, SamPredictor,sam_model_registry

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)



def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)



def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

def trackerlist(name: str, parameter_name: str, dataset_name: str, run_ids = None, display_name: str = None,
                result_only=False):
    """Generate list of trackers.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_ids: A single or list of run_ids.
        display_name: Name to be displayed in the result plots.
    """
    if run_ids is None or isinstance(run_ids, int):
        run_ids = [run_ids]
    return [Tracker(name, parameter_name, dataset_name, run_id, display_name, result_only) for run_id in run_ids]

def to_int(string):
    return int(os.path.basename(string)[:-4])

class Tracker:
    """Wraps the tracker for evaluation and running purposes.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_id: The run id.
        display_name: Name to be displayed in the result plots.
    """

    def __init__(self, name: str, parameter_name: str, dataset_name: str, run_id: int = None, display_name: str = None,
                 result_only=False):
        assert run_id is None or isinstance(run_id, int)

        self.name = name
        self.parameter_name = parameter_name
        self.dataset_name = dataset_name
        self.run_id = run_id
        self.display_name = display_name
        self.predictor = SamPredictor(sam_model_registry["vit_h"](checkpoint="").to(device="cuda"))

        env = env_settings()
        if self.run_id is None:
            self.results_dir = '{}/{}/{}'.format(env.results_path, self.name, self.parameter_name)
        else:
            self.results_dir = '{}/{}/{}_{:03d}'.format(env.results_path, self.name, self.parameter_name, self.run_id)
        if result_only:
            self.results_dir = '{}/{}'.format(env.results_path, self.name)

        tracker_module_abspath = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                              '..', 'tracker', '%s.py' % self.name))
        if os.path.isfile(tracker_module_abspath):
            tracker_module = importlib.import_module('lib.test.tracker.{}'.format(self.name))
            self.tracker_class = tracker_module.get_tracker_class()
        else:
            self.tracker_class = None

    def create_tracker(self, params):
        tracker = self.tracker_class(params, self.dataset_name)
        return tracker

    def run_sequence(self, seq, debug=None):
        """Run tracker on sequence.
        args:
            seq: Sequence to run the tracker on.
            visualization: Set visualization flag (None means default value specified in the parameters).
            debug: Set debug level (None means default value specified in the parameters).
            multiobj_mode: Which mode to use for multiple objects.
        """
        params = self.get_parameters()

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)

        params.debug = debug_

        init_info = seq.init_info()

        tracker = self.create_tracker(params)

        output = self._track_sequence(tracker, seq, init_info)
        return output

    def _track_sequence(self, tracker, seq, init_info):

        output = {'target_bbox': [],
                  'time': []}
        if tracker.params.save_all_boxes:
            output['all_boxes'] = []
            output['all_scores'] = []

        def _store_outputs(tracker_out: dict, defaults=None):
            defaults = {} if defaults is None else defaults
            for key in output.keys():
                val = tracker_out.get(key, defaults.get(key, None))
                if key in tracker_out or val is not None:
                    output[key].append(val)

        image = self._read_image(seq.frames[0])

        start_time = time.time()
        out = tracker.initialize(image, init_info)
        if out is None:
            out = {}

        prev_output = OrderedDict(out)
        init_default = {'target_bbox': init_info.get('init_bbox'),
                        'time': time.time() - start_time}
        if tracker.params.save_all_boxes:
            init_default['all_boxes'] = out['all_boxes']
            init_default['all_scores'] = out['all_scores']

        _store_outputs(out, init_default)

        for frame_num, frame_path in enumerate(seq.frames[1:], start=1):
            image = self._read_image(frame_path)

            start_time = time.time()

            info = seq.frame_info(frame_num)
            info['previous_output'] = prev_output

            if len(seq.ground_truth_rect) > 1:
                info['gt_bbox'] = seq.ground_truth_rect[frame_num]
            out = tracker.track(image, info)
            prev_output = OrderedDict(out)
            _store_outputs(out, {'time': time.time() - start_time})

        for key in ['target_bbox', 'all_boxes', 'all_scores']:
            if key in output and len(output[key]) <= 1:
                output.pop(key)

        return output

    def mouse_handler(self,event, x, y, flags,param):
        if event == cv.EVENT_LBUTTONDOWN and flags ==cv.EVENT_FLAG_CTRLKEY + cv.EVENT_LBUTTONDOWN:

            self.input_point.append([x,y])
            self.input_lbael.append(int(0))
            self.samflag=True
        elif event == cv.EVENT_LBUTTONDOWN:

            self.input_point.append([x,y])
            self.input_lbael.append(int(1))
            self.samflag=True

    def run_imgdir(self, videofilepath, optional_box=None, debug=None, visdom_info=None, save_results=False,index=None):
        """Run the tracker with the vieofile.
        args:
            debug: Debug level.
        """
        img_list=os.listdir(videofilepath)
        img_list.sort(key=to_int)
        params = self.get_parameters()

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)
        params.debug = debug_

        params.tracker_name = self.name
        params.param_name = self.parameter_name

        multiobj_mode = getattr(params, 'multiobj_mode', getattr(self.tracker_class, 'multiobj_mode', 'default'))

        if multiobj_mode == 'default':
            tracker = self.create_tracker(params)

        else:
            raise ValueError('Unknown multi object mode {}'.format(multiobj_mode))

        output_boxes = []

        def _build_init_info(box):
            return {'init_bbox': box}

        display_name = 'Display: ' + tracker.params.tracker_name
        if optional_box is not None:
            frame = cv.imread(videofilepath + '/' + img_list[0])
            frame_disp = frame.copy()
            color_seg = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
            self.predictor.set_image(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
            self.input_point=[]
            self.input_lbael=[]
            self.samflag=False

            while True:
                cv.imshow(display_name,frame_disp*(1-color_seg))
                cv.setMouseCallback(display_name, self.mouse_handler)

                k=cv.waitKey(1)
                if self.samflag==True:
                    color_seg = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
                    masks, scores, logits = self.predictor.predict(
                        point_coords=np.array(self.input_point),
                        point_labels=np.array(self.input_lbael),
                        multimask_output=True,)
                    mask = masks[np.argmax(scores)]
                    inde = np.where(mask == True)
                    xx = inde[0].min()
                    yy = inde[1].min()
                    ww = inde[0].max() - xx
                    hh = inde[1].max() - yy
                    cv.rectangle(color_seg, [yy,xx,hh,ww],(1,1,1), 3)
                    if self.input_lbael[-1]==0:
                        cv.circle(frame_disp, (self.input_point[-1][0], self.input_point[-1][1]), 5, (0, 128, 0), -1)
                    elif self.input_lbael[-1]==1:
                        cv.circle(frame_disp, (self.input_point[-1][0], self.input_point[-1][1]), 5, (0, 0, 128), -1)
                    self.samflag=False
                if k==ord('f'):
                    break
                elif k==ord('r'):
                    frame_disp = frame.copy()
                    self.input_point = []
                    self.input_lbael = []
                    color_seg = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
            init_state = [yy,xx, hh, ww]
            tracker.initialize(frame, _build_init_info(init_state))
            output_boxes.append(init_state)

        else:

            cv.namedWindow(display_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
            cv.resizeWindow(display_name, 960, 720)

            frame = cv.imread(videofilepath + '/' + img_list[0])
            cv.imshow(display_name, frame)
            while True:

                frame_disp = frame.copy()

                cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL,
                           1.5, (0, 0, 0), 1)

                x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
                init_state = [x, y, w, h]
                tracker.initialize(frame, _build_init_info(init_state))
                output_boxes.append(init_state)
                break

        for img in img_list:
            frame = cv.imread(videofilepath+'/'+img)

            if frame is None:
                break

            img_tosave = frame.copy()

            out = tracker.track(frame)
            state = [int(s) for s in out['target_bbox']]
            output_boxes.append(state)

            cv.rectangle(img_tosave, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
                         index, 5)
            self.predictor.set_image(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
            masks, scores, logits = self.predictor.predict(box=np.array([state[0],state[1],state[2] + state[0],state[3] + state[1]]), multimask_output=True, )
            mask = masks[np.argmax(scores)]
            color_seg = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
            color_seg[mask == True, :] = (0, 0, 255)

            cv.imshow(display_name, (img_tosave*0.5+color_seg*0.5)/255)
            key = cv.waitKey(1)

        cv.destroyAllWindows()


    def get_parameters(self):
        """Get parameters."""
        param_module = importlib.import_module('lib.test.parameter.{}'.format(self.name))
        params = param_module.parameters(self.parameter_name)
        return params

    def _read_image(self, image_file: str):
        if isinstance(image_file, str):
            im = cv.imread(image_file)
            return cv.cvtColor(im, cv.COLOR_BGR2RGB)
        elif isinstance(image_file, list) and len(image_file) == 2:
            return decode_img(image_file[0], image_file[1])
        else:
            raise ValueError("type of image_file should be str or list")