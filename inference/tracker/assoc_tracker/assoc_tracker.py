import numpy as np
from utils.kalman_filter import KalmanFilterXYWH
from fast_reid.fast_reid_interfece import FastReIDInterface
import torch
import copy
import gurobipy as gp
from scipy.interpolate import interp1dx
from pulp import *
import time
from tracker.assoc_tracker.tracker_dists import TrackerDists
from .node import Node
from .detection import Detection
import json
import yaml
from .tracker import Tracker


# REID_MAPPING_FILE = "/Users/keshavgupta/desktop/CVIT/TrafficViolations/simil_off_diag.npy"

# TODO
# Not penalizing -1

# TODO : Can make this MUCH MORE memory efficent as Node contains detections and we are making many copies of a given node.
# We can avoid that by having a global memory for all the node.detections

class AssocTracker():
    def __init__(self, cfg, reid_model=None):
        if(type(cfg) == str):
            if(cfg.endswith("json")):
                cfg = self.read_json_cfg(cfg)
            elif(cfg.endswith("yaml")):
                cfg = self.read_yaml_cfg(cfg)

        self.cfg = cfg
        self.logger = self.null
        self.frame_num = 0
        self.prune_after = 1
        self.eps = 0

        self.no_helmet_obj_weight = self.cfg.get('no_helmet_obj_weight', 1)
        self.license_plate_obj_weight = self.cfg.get('license_plate_obj_weight', 1)
        self.assoc_weight = self.cfg.get('assoc_weight', 1)
        self.assoc_thresh = self.cfg.get('assoc_thresh', 0.4)
        self.last_n_frames = self.cfg.get("last_n_frames", 40)
        self.max_miss_count = self.cfg.get("max_miss_count", 20)
        self.max_tree_depth = self.cfg.get("max_tree_depth", 25)
        assert self.max_tree_depth > self.max_miss_count, "Max Tree Depth should be more than the max miss count"
        print("TrHERE")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if(reid_model is None):
            self.reid_encoder = FastReIDInterface(config_file=self.cfg.get("REID_CONFIG"), weights_path=self.cfg.get("REID_WEIGHTS"), device=self.device)
        else:
            self.reid_encoder = reid_model
        print("REID LOADED")
        self.no_helmet_tracker = Tracker("no_helmet", self.prune_after, self.logger, self.reid_encoder, self.max_tree_depth, cfg)
        self.license_plate_tracker = Tracker("license_plate", self.prune_after, self.logger, self.reid_encoder, self.max_tree_depth, cfg)
        
        self.masks_no_helmets = None
        self.cross_masks_no_helmets = None
        self.masks_license_plates = None
        self.cross_masks_license_plates = None
        self.boxes_no_helmets = None
        self.boxes_license_plates = None
        self.mask_shape = [50, 100]
        self.mask_shape = [93, 160]

        self.curr_no_helmet_tracks = {}
        self.curr_license_plate_tracks = {}
        self.curr_assocs = {} # license_plate to no_helmet
        self.curr_assocs_inv = {} # no_helmet to license_plate

        self.make_problem_time = []

    def read_json_cfg(self, cfg):
        with open(cfg, 'r') as f:
            contents = json.load(f)
        return contents

    def read_yaml_cfg(self, cfg):
        with open(cfg, 'r') as f:
            contents = yaml.load(f, Loader=yaml.loader.SafeLoader)
        return contents

    def append_to_array(self, arr, elem):
        num_dets = elem.shape[0]
        max_dets_till_now = arr.shape[1] if arr is not None else 0
        new_max_dets = max(num_dets, max_dets_till_now)
        print("*****************")
        print("MAX_DETS_TILL_NOW : ", new_max_dets)   
        print("*****************")
        if(len(elem.shape) == 1):
            # if elem is []
            elem = np.zeros((len(elem), *self.mask_shape)).astype(np.uint8)
        if(arr is not None):
            start = 0
            new_arr_size = arr.shape[0] + 1
            if(self.frame_num > self.last_n_frames):
                new_arr_size = self.last_n_frames
                start = 1
            new = np.zeros((new_arr_size, new_max_dets, *self.mask_shape)).astype(np.uint8)
            new[:-1, :max_dets_till_now] = arr[start:]
            new[-1, :elem.shape[0]] = elem
        else:
            new = elem[None]
        return new

    def append_box_to_array(self, arr, elem):
        # arr is (t, max_dets, 4)
        num_dets = elem.shape[0]
        max_dets_till_now = arr.shape[1] if arr is not None else 0
        new_max_dets = max(num_dets, max_dets_till_now)
        print("*****************")
        print("MAX_DETS_TILL_NOW : ", new_max_dets)   
        print("*****************")
        if(len(elem.shape) == 1):
            # if elem is []
            elem = np.zeros((len(elem), 4))
        if(arr is not None):
            start = 0
            new_arr_size = arr.shape[0] + 1
            if(self.frame_num > self.last_n_frames):
                new_arr_size = self.last_n_frames
                start = 1
            new = np.zeros((new_arr_size, new_max_dets, 4))
            new[:-1, :max_dets_till_now] = arr[start:]
            new[-1, :elem.shape[0]] = elem
        else:
            new = elem[None]
        return new
    
    def sigmoid(self, a):
        return 1 / (1 + np.exp(-a))

    def track(self, detections, masks, cross_masks, frame, no_helmet_reid_feats=None, license_plate_reid_feats=None):
        '''
        detections are the new detections in the frame. Assuming a new scan from the sensor. It is a numpy array of shape (N, 6)
        representing 
            the coordinates of the bounding box detection in (xyxy format),
            the confidence of the detection,
            the class of the detection (0 for no_helmet, 1 for license_plate)
        
        masks is the (N, H, W) array holding the masks of the correponding detections
        cross_masks is the (N, H, W) array holding the cross masks of the corresponding detections

        frame is a numpy array of shape (H, W, 3) representing the image of the environment

        The function updates the Tracker with the new detections
        '''
        # self.last_n_frames = self.frame_num
        self.frame_num += 1
        if(len(detections) == 0):
            no_helmet_dets = []
            license_plate_dets = []
        else:
            cls = detections[:, -1]
            no_helmet_dets = detections[cls == 0][:, :-1]
            no_helmet_boxes = detections[cls == 0][:, :-2]
            license_plate_dets = detections[cls == 1][:, :-1]
            license_plate_boxes = detections[cls == 1][:, :-2]
        
        if(self.frame_num % self.prune_after == 0):
            # make gp model
            model = gp.Model()
        else:
            model = None
        print("LEN", len(no_helmet_dets), len(license_plate_dets))
        
        no_helmet_feats, no_helmet_vars, no_helmet_all_paths, no_helmet_obj, no_helmet_costs = self.no_helmet_tracker.track(no_helmet_dets, frame, model, reid_features=no_helmet_reid_feats)
        license_plate_feats, license_plate_vars, license_plate_all_paths, license_plate_obj, license_plate_costs = self.license_plate_tracker.track(license_plate_dets, frame, model, reid_features=license_plate_reid_feats)

        # Need for defining the association costs while making the problem
        if(len(detections) != 0):
            self.masks_no_helmets = self.append_to_array(self.masks_no_helmets, masks[cls == 0].astype(np.uint8))
            self.masks_license_plates = self.append_to_array(self.masks_license_plates, masks[cls == 1].astype(np.uint8))
            self.cross_masks_no_helmets = self.append_to_array(self.cross_masks_no_helmets, cross_masks[cls == 0].astype(np.uint8))
            self.cross_masks_license_plates = self.append_to_array(self.cross_masks_license_plates, cross_masks[cls == 1].astype(np.uint8))
            self.boxes_no_helmets = self.append_box_to_array(self.boxes_no_helmets, no_helmet_boxes)
            self.boxes_license_plates = self.append_box_to_array(self.boxes_license_plates, license_plate_boxes)
        else:
            self.masks_no_helmets = self.append_to_array(self.masks_no_helmets, np.array([]))
            self.masks_license_plates = self.append_to_array(self.masks_license_plates, np.array([]))
            self.cross_masks_no_helmets = self.append_to_array(self.cross_masks_no_helmets, np.array([]))
            self.cross_masks_license_plates = self.append_to_array(self.cross_masks_license_plates, np.array([]))
            self.boxes_no_helmets = self.append_box_to_array(self.boxes_no_helmets, np.array([]))
            self.boxes_license_plates = self.append_box_to_array(self.boxes_license_plates, np.array([]))


        if(self.mask_shape is None):
            self.mask_shape = masks[0].shape

        if(self.frame_num % self.prune_after == 0):
            self.n_scan_pruning(model, no_helmet_vars, license_plate_vars, no_helmet_all_paths, license_plate_all_paths, no_helmet_obj, license_plate_obj, no_helmet_costs, license_plate_costs)

        return no_helmet_feats, license_plate_feats
    
    def calc_assoc_score(self, no_helmet, license_plate):
        
        no_helmet_dets = [None] * self.frame_num
        license_plate_dets = [None] * self.frame_num
        for det in no_helmet:
            # if(det.det_id != -1):
                no_helmet_dets[det.frame_num - 1] = det
        for det in license_plate:
            # if(det.det_id != -1):
                license_plate_dets[det.frame_num - 1] = det

        # print("No Helmet Dets : ", no_helmet_dets)
        # print("License Plate Dets", license_plate_dets)
        score = 0
        for i in range(self.frame_num):
            no_helmet_det = no_helmet_dets[i]
            license_plate_det = license_plate_dets[i]
            if(no_helmet_det is None or license_plate_det is None):
                continue
            
            no_helmet_det_id = no_helmet_det.det_id
            license_plate_det_id = license_plate_det.det_id

            if(license_plate_det_id == -1 and no_helmet_det_id == -1):
                continue
            mask_no_helmet = self.masks_no_helmets[i][no_helmet_det_id]
            mask_license_plate = self.masks_license_plates[i][license_plate_det_id]
            if(no_helmet_det_id == -1):
                cross_mask_no_helmet = np.zeros_like(mask_license_plate)
            else:
                cross_mask_no_helmet = self.cross_masks_no_helmets[i][no_helmet_det.det_id]
            if(license_plate_det_id == -1):
                mask_license_plate = np.zeros_like(mask_no_helmet)
            else:
                mask_license_plate = self.masks_license_plates[i][license_plate_det.det_id]
            # cross_mask_license_plate = self.cross_masks_license_plates[i][license_plate_det.det_id]

            # Calculate the iou between the mask_license_plate and the cross_mask_no_helmet
            if(license_plate_det_id == -1):
                # The predicted license_plate mask should be a null mask
                val = 0.5 - 1 / (1 + np.exp(-np.sum(cross_mask_no_helmet)))
                score += val
                # print(np.sum(cross_mask_no_helmet), val)
            else:
                intersection = np.sum(mask_license_plate * cross_mask_no_helmet)
                union = mask_license_plate + cross_mask_no_helmet
                union[union >= 1] = 1
                union = np.sum(union)
                iou = intersection / union
                score += (iou - 0.5)
                # print(i, iou)
        
        score = 1 / (1 + np.exp(-score))
        # print("Score : ", score)
        return score

    def make_problem_identity(self, model, rider_vars, rider_all_paths, rider_obj, rider_costs, motor_vars, motor_all_paths, motor_obj, motor_costs):
        # Add len(tracks_rider) * len(motor_vars) constraints to the problem
        assoc_variables = []
        for i in range(len(rider_all_paths)):
            rider_assocs = []
            rider_aux_var = []
            for j in range(len(motor_vars)):
                rider_assocs.append(model.addVar(vtype=gp.GRB.BINARY, name=f"assoc_{i}_{j}"))
                rider_aux_var.append(model.addVar(vtype=gp.GRB.BINARY, name=f"aux_{i}_{j}"))
            assoc_variables.append(rider_assocs)
        return assoc_variables

    def make_scores(self, assoc_scores_mcr, assoc_scores_rcm, frame_count, tracks_rider_dets, tracks_motor_dets, rider_all_paths, motor_all_paths, tracks_rider_tids, tracks_motor_tids):
        # The function sets the assoc_scores_mcr and the assoc_scores_rcm variables
        # self.masks_motors is of shape (T, M, H, W) and tracks_motor_dets is of shape (N, T)
        curr_motor_masks = []
        start = 0
        for i in range(len(motor_all_paths)):
            to_append = self.masks_license_plates[np.arange(start, start + frame_count, dtype=np.int32), tracks_motor_dets[i]]
            to_append[tracks_motor_dets[i] == -1] = np.zeros(self.mask_shape).astype(np.uint8)
            curr_motor_masks.append(to_append)
        curr_motor_masks = np.array(curr_motor_masks)

        curr_cross_rider_masks = []
        for i in range(len(rider_all_paths)):
            to_append = self.cross_masks_no_helmets[np.arange(start, start + frame_count, dtype=np.int32), tracks_rider_dets[i]]
            to_append[tracks_rider_dets[i] == -1] = np.zeros(self.mask_shape).astype(np.uint8)
            curr_cross_rider_masks.append(to_append)
        curr_cross_rider_masks = np.array(curr_cross_rider_masks)

        curr_rider_masks = []
        for i in range(len(rider_all_paths)):
            to_append = self.masks_no_helmets[np.arange(start, start + frame_count, dtype=np.int32), tracks_rider_dets[i]]
            to_append[tracks_rider_dets[i] == -1] = np.zeros(self.mask_shape).astype(np.uint8)
            curr_rider_masks.append(to_append)
        curr_rider_masks = np.array(curr_rider_masks)

        curr_cross_motor_masks = []
        for i in range(len(motor_all_paths)):
            to_append = self.cross_masks_license_plates[np.arange(start, start + frame_count, dtype=np.int32), tracks_motor_dets[i]]
            to_append[tracks_motor_dets[i] == -1] = np.zeros(self.mask_shape).astype(np.uint8)
            curr_cross_motor_masks.append(to_append)
        curr_cross_motor_masks = np.array(curr_cross_motor_masks)

        for i in range(len(curr_motor_masks)):
            scores = np.zeros_like(tracks_rider_dets).astype(np.float32) + self.eps
            intersection = (curr_cross_rider_masks & curr_motor_masks[i][None]).reshape(len(rider_all_paths), frame_count, -1).sum(-1)
            union = (curr_cross_rider_masks | curr_motor_masks[i][None]) * (curr_motor_masks[i][None] > 0)
            union = union.reshape(len(rider_all_paths), frame_count, -1).sum(-1)
            cond = (tracks_motor_dets[i][None] != -1) * (tracks_rider_dets != -1)
            scores[cond] = np.float32(intersection[cond]) / np.float32(union[cond]) - self.assoc_thresh
            scores[np.isnan(scores)] = 0
            scores[np.isinf(scores)] = 0
            assoc_scores_mcr[:, i] = np.sum(scores, -1)

            # scores[scores < 0] = -1000

        for i in range(len(curr_rider_masks)):
            scores = np.zeros_like(tracks_motor_dets).astype(np.float32) + self.eps
            intersection = (curr_cross_motor_masks & curr_rider_masks[i][None]).reshape(len(motor_all_paths), frame_count, -1).sum(-1)
            union = (curr_cross_motor_masks | curr_rider_masks[i][None]) * (curr_rider_masks[i][None] > 0)
            union = union.reshape(len(motor_all_paths), frame_count, -1).sum(-1)
            cond = (tracks_motor_dets != -1) * (tracks_rider_dets[i][None] != -1)
            scores[cond] = np.float32(intersection[cond]) / np.float32(union[cond]) - self.assoc_thresh
            scores[np.isnan(scores)] = 0
            scores[np.isinf(scores)] = 0
            scores[scores < 0] = -1000

            assoc_scores_rcm[i, :] = np.sum(scores, -1)

            curr_rider_id = tracks_rider_tids[i]

            if(curr_rider_id in self.curr_assocs_inv):
                assoc_motor_id = self.curr_assocs_inv[curr_rider_id]
                in_valid_paths = tracks_motor_tids != assoc_motor_id
                assoc_scores_rcm[i, in_valid_paths] = -10000
        
        return assoc_scores_rcm, assoc_scores_mcr
    
    def make_scores_boxes(self, assoc_scores_lpnh, assoc_scores_nhlp, frame_count, tracks_no_helmet_dets, tracks_license_plate_dets, no_helmet_all_paths, license_plate_all_paths, tracks_no_helmet_tids, tracks_license_plate_tids):
        # The function sets the assoc_scores_lpnh and the assoc_scores_nhlp variables
        # self.boxes_license_plates is of shape (T, M, 4) and tracks_license_plate_dets is of shape (N, T)
        curr_license_plate_boxes = []
        start = 0
        for i in range(len(license_plate_all_paths)):
            to_append = self.boxes_license_plates[np.arange(start, start + frame_count, dtype=np.int32), tracks_license_plate_dets[i]]
            to_append[tracks_license_plate_dets[i] == -1] = np.zeros(4).astype(np.uint8)
            curr_license_plate_boxes.append(to_append)
        curr_license_plate_boxes = np.array(curr_license_plate_boxes)

        curr_no_helmet_boxes = []
        for i in range(len(no_helmet_all_paths)):
            to_append = self.boxes_no_helmets[np.arange(start, start + frame_count, dtype=np.int32), tracks_no_helmet_dets[i]]
            to_append[tracks_no_helmet_dets[i] == -1] = np.zeros(4).astype(np.uint8)
            curr_no_helmet_boxes.append(to_append)
        curr_no_helmet_boxes = np.array(curr_no_helmet_boxes)

        print(curr_license_plate_boxes.shape) # these are of shape (N, T, 4)
        print(curr_no_helmet_boxes.shape) # these are of shape (M, T, 4)
        
        assoc_scores = np.zeros((len(no_helmet_all_paths), len(license_plate_all_paths)))
        
        # assuming t = 1 for now only, will change later
        for t in range(frame_count):
            for i in range(len(no_helmet_all_paths)):
                if(tracks_no_helmet_dets[i, t] == -1):
                    continue
                iou_array = np.zeros(len(license_plate_all_paths))
                no_helmet_box = curr_no_helmet_boxes[i, t]
                license_plate_boxes = curr_license_plate_boxes[:, t]
                license_plate_dets = tracks_license_plate_dets[:, t]
                # find the iou between no_helmet box and all license_plate_boxes vectorized
                for j in range(len(license_plate_all_paths)):
                    license_plate_box = license_plate_boxes[j]
                    if(license_plate_dets[j] == -1):
                        iou_array[j] = 0
                    else:
                        # print(no_helmet_box)
                        # print(license_plate_box)
                        intersection = np.maximum(0, np.minimum(no_helmet_box[2], license_plate_box[2]) - np.maximum(no_helmet_box[0], license_plate_box[0])) * np.maximum(0, np.minimum(no_helmet_box[3], license_plate_box[3]) - np.maximum(no_helmet_box[1], license_plate_box[1]))
                        union = (no_helmet_box[2] - no_helmet_box[0]) * (no_helmet_box[3] - no_helmet_box[1]) + (license_plate_box[2] - license_plate_box[0]) * (license_plate_box[3] - license_plate_box[1]) - intersection
                        iou_array[j] = intersection / union
                        print(iou_array)
                        # print(iou_array[j], intersection, union)
                        # iou_array[np.isnan(iou_array)] = 0
                        # iou_array[np.isinf(iou_array)] = 0
                        iou_array[iou_array < 0] = -1000
                
                if(np.max(iou_array) > 0):
                    idx = np.argmax(iou_array)
                    license_plate_det_id = license_plate_dets[idx]
                    iou_array[license_plate_dets == license_plate_det_id] = 1
                    iou_array[license_plate_dets != license_plate_det_id] = -1
                assoc_scores[i, :] = iou_array
                print(assoc_scores)
            
        assoc_scores[assoc_scores <= 0] = -1000
        
        for i in range(len(curr_no_helmet_boxes)):
            curr_no_helmet_id = tracks_no_helmet_tids[i]
            if(curr_no_helmet_id in self.curr_assocs_inv):
                assoc_license_plate_id = self.curr_assocs_inv[curr_no_helmet_id]
                in_valid_paths = tracks_license_plate_tids != assoc_license_plate_id
                assoc_scores[i, in_valid_paths] = -1000
        
        assoc_scores_lpnh = assoc_scores
        assoc_scores_nhlp = assoc_scores
        
        return assoc_scores_nhlp, assoc_scores_lpnh

    def make_problem(self, model, no_helmet_vars, no_helmet_all_paths, no_helmet_costs, license_plate_vars, license_plate_all_paths, license_plate_costs, obj):
        # Quickly return in case there is no cross_association information
        if(len(no_helmet_all_paths) == 0 or len(license_plate_all_paths) == 0):
            model.setObjective(obj, sense=gp.GRB.MAXIMIZE)
            return []
        
        assoc_variables = []
        aux_variables = []
        # Add len(tracks_no_helmet) * len(license_plate_vars) constraints to the problem
        for i in range(len(no_helmet_all_paths)):
            no_helmet_assocs = []
            no_helmet_aux_var = []
            for j in range(len(license_plate_vars)):
                no_helmet_assocs.append(model.addVar(vtype=gp.GRB.BINARY, name=f"assoc_{i}_{j}"))
                no_helmet_aux_var.append(model.addVar(vtype=gp.GRB.BINARY, name=f"aux_{i}_{j}")) 
            assoc_variables.append(no_helmet_assocs)
            aux_variables.append(no_helmet_aux_var)

        # Add the constraint that every no_helmet can be bound to only one license_plate
        for i in range(len(no_helmet_all_paths)):
            model.addConstr(gp.quicksum(assoc_variables[i]) <= 1)
        
        for i in range(len(no_helmet_all_paths)):
            for j in range(len(license_plate_all_paths)):
                model.addConstr(aux_variables[i][j] == no_helmet_vars[i] * license_plate_vars[j])
                model.addConstr(assoc_variables[i][j] * (1 - aux_variables[i][j]) == 0)
        
        # The problem
        # TODO: Vectorize Somehow
        # Make a cost array of shape (len(tracks_no_helmet), len(license_plate_vars)) representing the association scores (b/w [0,1])
        assoc_scores_lpnh = np.zeros((len(no_helmet_all_paths), len(license_plate_vars))) + self.eps
        assoc_scores_nhlp = np.zeros((len(no_helmet_all_paths), len(license_plate_vars))) + self.eps
        def get_property(obj, attr):
            return getattr(obj, attr)

        get_property_vectorized = np.vectorize(get_property)
        
        # Consider the last n frames of all the hypothesis
        frame_count = self.frame_num
        start = 0
        if(frame_count > self.last_n_frames):
            frame_count = self.last_n_frames 
            start += self.frame_num - self.last_n_frames

        tracks_no_helmet_dets = np.zeros((len(no_helmet_all_paths), frame_count), dtype=np.int32)
        tracks_license_plate_dets = np.zeros((len(license_plate_all_paths), frame_count), dtype=np.int32)
        
        tracks_no_helmet_tids = np.zeros((len(no_helmet_all_paths)), dtype=np.int32)
        tracks_license_plate_tids = np.zeros((len(license_plate_all_paths)), dtype=np.int32)

        for i in range(len(no_helmet_all_paths)):
            no_helmet = no_helmet_all_paths[i]
            no_helmet_det_ids = np.array(get_property_vectorized(no_helmet, "det_id"))
            no_helmet_frame_num = np.array(get_property_vectorized(no_helmet, "frame_num")) - 1
            no_helmet_dets_ = np.ones(frame_count, dtype=np.int32) * -1
            first_frame_no_helmet = no_helmet_frame_num[0]
            if(first_frame_no_helmet < self.frame_num - self.last_n_frames):
                first_frame_no_helmet = - self.last_n_frames
                no_helmet_dets_ = no_helmet_det_ids[first_frame_no_helmet:]
            else:
                no_helmet_dets_[-self.frame_num + first_frame_no_helmet:] = no_helmet_det_ids
            tracks_no_helmet_dets[i] = no_helmet_dets_
            tracks_no_helmet_tids[i] = no_helmet[-1].track_id
        
        for i in range(len(license_plate_all_paths)):
            license_plate = license_plate_all_paths[i]
            license_plate_det_ids = np.array(get_property_vectorized(license_plate, "det_id"))
            license_plate_frame_num = np.array(get_property_vectorized(license_plate, "frame_num")) - 1
            license_plate_dets_ = np.ones(frame_count, dtype=np.int32) * -1
            first_frame_license_plate = license_plate_frame_num[0]
            if(first_frame_license_plate < self.frame_num - self.last_n_frames):
                first_frame_license_plate = - self.last_n_frames
                license_plate_dets_ = license_plate_det_ids[first_frame_license_plate:]
            else:
                license_plate_dets_[-self.frame_num + first_frame_license_plate: ] = license_plate_det_ids
            tracks_license_plate_dets[i] = license_plate_dets_
            tracks_license_plate_tids[i] = license_plate[-1].track_id

        # Fill the assoc_scores_nhlp and assoc_scores_lpnh
        assoc_scores_nhlp, assoc_scores_lpnh = self.make_scores(assoc_scores_lpnh, assoc_scores_nhlp, frame_count, tracks_no_helmet_dets, tracks_license_plate_dets, no_helmet_all_paths, license_plate_all_paths, tracks_no_helmet_tids, tracks_license_plate_tids)
        # assoc_scores_nhlp, assoc_scores_lpnh = self.make_scores_boxes(assoc_scores_lpnh, assoc_scores_nhlp, frame_count, tracks_no_helmet_dets, tracks_license_plate_dets, no_helmet_all_paths, license_plate_all_paths, tracks_no_helmet_tids, tracks_license_plate_tids)
        print(assoc_scores_lpnh)
        print(assoc_scores_nhlp)

        time_up = time.time()

        for i in range(len(no_helmet_all_paths)):
            for j in range(len(license_plate_all_paths)):
                no_helmet_track = no_helmet_all_paths[i]
                license_plate_hypothesis = license_plate_all_paths[j]
                time_ = time.time()
                print(self.assoc_weight)
                obj += assoc_scores_lpnh[i, j] * assoc_variables[i][j] * self.assoc_weight
                obj += assoc_scores_nhlp[i, j] * assoc_variables[i][j] * self.assoc_weight

                print(no_helmet_all_paths[i])
                print(license_plate_all_paths[j])
                print(no_helmet_costs[i])
                print(license_plate_costs[j])
                print(assoc_scores_lpnh[i][j])
                print(assoc_scores_nhlp[i][j])
                print("Sum : ", self.assoc_weight * (assoc_scores_nhlp[i][j] + assoc_scores_lpnh[i][j]) + self.license_plate_obj_weight*(license_plate_costs[j]) + self.no_helmet_obj_weight*(no_helmet_costs[i]))
                print()

        print(time.time() - time_up)
        model.setObjective(obj, sense=gp.GRB.MAXIMIZE)
        return assoc_variables

    def n_scan_pruning(self, model, no_helmet_vars, license_plate_vars, no_helmet_all_paths, license_plate_all_paths, no_helmet_obj, license_plate_obj, no_helmet_costs, license_plate_costs):
        # Solve joint tracking problem and association
        # model = gp.Model()
        # no_helmet_vars, no_helmet_all_paths, no_helmet_obj, no_helmet_costs = self.no_helmet_tracker.solve_mwis_problem(model)
        # license_plate_vars, license_plate_all_paths, license_plate_obj, license_plate_costs = self.license_plate_tracker.solve_mwis_problem(model)
        tracks_no_helmet, tracks_license_plate = [], []
        # if(not len(no_helmet_all_paths) or not len(license_plate_all_paths)):
        #     return
        time0 = time.time()
        obj = self.no_helmet_obj_weight * no_helmet_obj + self.license_plate_obj_weight * license_plate_obj
        assoc_variables = self.make_problem(model, no_helmet_vars, no_helmet_all_paths, no_helmet_costs, license_plate_vars, license_plate_all_paths, license_plate_costs, obj)
        self.make_problem_time.append(time.time() - time0)
        model.optimize()

        if model.status == gp.GRB.OPTIMAL:
            for i in range(len(license_plate_vars)):
                if(license_plate_vars[i].x):
                    tracks_license_plate.append(license_plate_all_paths[i])

        if model.status == gp.GRB.OPTIMAL:
            for i in range(len(no_helmet_vars)):
                if(no_helmet_vars[i].x):
                    tracks_no_helmet.append(no_helmet_all_paths[i])

        print("Tracks No Helmet")
        for tr in tracks_no_helmet:
            print(tr)
        print("Tracks License Plate")
        for tm in tracks_license_plate:
            print(tm)
        print()
        associations_no_helmets = np.ones(len(tracks_no_helmet)).astype(np.int32) * -1
        for i in range(len(assoc_variables)):
            for j in range(len(assoc_variables[i])):
                if(assoc_variables[i][j].x == 1):
                    for k in range(len(tracks_no_helmet)):
                        if(tracks_no_helmet[k] == no_helmet_all_paths[i]):
                            no_helmet_idx = k
                    for k in range(len(tracks_license_plate)):
                        if(tracks_license_plate[k] == license_plate_all_paths[j]):
                            license_plate_idx = k
                    associations_no_helmets[no_helmet_idx] = license_plate_idx
                    # print("No Helmet : ", no_helmet_all_paths[i])
                    # print("License Plate : ", license_plate_all_paths[j])
                    # print()

        self.no_helmet_tracker.after_n_scan(tracks_no_helmet, tracked=True)
        self.license_plate_tracker.after_n_scan(tracks_license_plate, tracked=True)

        self.update_curr_tracks(tracks_no_helmet, tracks_license_plate, associations_no_helmets)
        self.print_tracks(self.curr_no_helmet_tracks)
        self.print_tracks(self.curr_license_plate_tracks)
        print()
        print(self.curr_assocs)
        # if(self.frame_num == 3):
            # exit(0)

    def update_curr_tracks(self, tracks_no_helmet, tracks_license_plate, associations_no_helmets):
        for i, nh_track in enumerate(tracks_no_helmet):
            tid = nh_track[-1].track_id
            try:
                self.curr_no_helmet_tracks[tid].extend([copy.deepcopy(nh_track[-1])])
            except:
                self.curr_no_helmet_tracks[tid] = nh_track
        
        for i, lp_track in enumerate(tracks_license_plate):
            tid = lp_track[-1].track_id
            try:
                self.curr_license_plate_tracks[tid].extend([copy.deepcopy(lp_track[-1])])
            except:
                self.curr_license_plate_tracks[tid] = lp_track

        for i in range(len(associations_no_helmets)):
            if(associations_no_helmets[i] == -1):
                continue
            no_helmet_tid = tracks_no_helmet[i][-1].track_id
            license_plate_tid = tracks_license_plate[associations_no_helmets[i]][-1].track_id
            try:
                self.curr_assocs[license_plate_tid][no_helmet_tid] = 1
            except:
                self.curr_assocs[license_plate_tid] = {no_helmet_tid : 1}

            try:
                assoc_license_plate_tid = self.curr_assocs_inv[no_helmet_tid]
                assert assoc_license_plate_tid == license_plate_tid, f"Association ID of no_helmet with {no_helmet_tid} changed from {assoc_license_plate_tid} to {license_plate_tid}"
            except:
                self.curr_assocs_inv[no_helmet_tid] = license_plate_tid

    def give_tracks(self):
        model = gp.Model()
        no_helmet_vars, no_helmet_all_paths, no_helmet_obj, no_helmet_costs = self.no_helmet_tracker.solve_mwis_problem(model)
        license_plate_vars, license_plate_all_paths, license_plate_obj, license_plate_costs = self.license_plate_tracker.solve_mwis_problem(model)
        tracks_no_helmet, tracks_license_plate, assocs = [], [], []
        
        print("Final License Plate paths : ")
        print(license_plate_all_paths)
        
        obj = self.no_helmet_obj_weight * no_helmet_obj + self.license_plate_obj_weight * license_plate_obj
        assoc_variables = self.make_problem(model, no_helmet_vars, no_helmet_all_paths, no_helmet_costs, license_plate_vars, license_plate_all_paths, license_plate_costs, obj)
        model.optimize()

        if model.status == gp.GRB.OPTIMAL:
            for i in range(len(license_plate_vars)):
                if(license_plate_vars[i].x):
                    tracks_license_plate.append(license_plate_all_paths[i])
        
        if model.status == gp.GRB.OPTIMAL:
            for i in range(len(no_helmet_vars)):
                if(no_helmet_vars[i].x):
                    tracks_no_helmet.append(no_helmet_all_paths[i])

        print("Tracks No Helmet")
        for tr in tracks_no_helmet:
            print(tr)
        print("Tracks License Plate")
        for tm in tracks_license_plate:
            print(tm)

        associations_no_helmets = np.ones(len(tracks_no_helmet)).astype(np.int32) * -1
        for i in range(len(assoc_variables)):
            for j in range(len(assoc_variables[i])):
                if(assoc_variables[i][j].x == 1):
                    for k in range(len(tracks_no_helmet)):
                        if(tracks_no_helmet[k] == no_helmet_all_paths[i]):
                            no_helmet_idx = k
                    for k in range(len(tracks_license_plate)):
                        if(tracks_license_plate[k] == license_plate_all_paths[j]):
                            license_plate_idx = k
                    associations_no_helmets[no_helmet_idx] = license_plate_idx
        
        # IMP
        # For n_scan=1, we dont need this
        # self.update_curr_tracks(tracks_no_helmet, tracks_license_plate, associations_no_helmets)

        assoc_no_helmets = np.ones(len(self.curr_no_helmet_tracks), dtype=np.int32) * -1
        assoc_license_plates = np.arange(len(self.curr_license_plate_tracks), dtype=np.int32)

        for lp_id, no_helmet_dict in self.curr_assocs.items():
            for nh_id, _ in no_helmet_dict.items():
                assoc_no_helmets[nh_id] = lp_id

        self.print_tracks(self.curr_no_helmet_tracks)
        self.print_tracks(self.curr_license_plate_tracks)
        print(assoc_no_helmets)
        print(assoc_license_plates)
        print(self.curr_assocs)
        out_no_helmet = self.no_helmet_tracker.give_tracks(self.curr_no_helmet_tracks, assoc_no_helmets)
        out_license_plate = self.license_plate_tracker.give_tracks(self.curr_license_plate_tracks, assoc_license_plates)

        license_plate_no_helmet_count = np.zeros(len(self.curr_license_plate_tracks), dtype=np.int32)
        for i in range(len(license_plate_no_helmet_count)):
            license_plate_no_helmet_count[i] = len(np.where(assoc_no_helmets == i)[0])

        print(license_plate_no_helmet_count)
        return out_no_helmet, out_license_plate, license_plate_no_helmet_count
    
    def print_tracks(self, track):
        for tr in track:
            print(tr)
    
    def null(self, *args, **kwargs):
        pass