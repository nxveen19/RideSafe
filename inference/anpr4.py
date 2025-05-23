import json
import cv2
from ultralytics import YOLO
import numpy as np
import os
import argparse
import re
from datetime import datetime
from paddleocr import PaddleOCR
import copy
from tracker.utils.kalman_filter import KalmanFilterXYWH

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Global variables
CLASS_NAME = ['bike', 'helmet', 'no-helmet', 'number-plate']

class Detection:
    def __init__(self, bounding_box, det_id):
        self.bounding_box_xyxyc = bounding_box.astype(np.int32)
        self.bounding_box_xywhc = self.xyxyc_to_xywhc()
        self.xcenter = bounding_box[0]
        self.ycenter = bounding_box[1]
        self.width = bounding_box[2]
        self.height = bounding_box[3]
        self.conf = bounding_box[4]
        self.det_id = det_id
        self.text = ""
    
    def xywhc_to_xyxyc(self):
        x, y, w, h, c = self.bounding_box_xywhc
        return np.array([x-w/2, y-h/2, x+w/2, y+h/2, c]).astype(np.int32)

    def xyxyc_to_xywhc(self):
        xmin, ymin, xmax, ymax, c = self.bounding_box_xyxyc
        return np.array([(xmin + xmax)/2, (ymin + ymax)/2, xmax - xmin, ymax - ymin, c]).astype(np.int32)

class Node:
    def __init__(self, frame_num, det, det_id):
        self.frame_num = frame_num
        self.det_id = det_id
        self.mot_score = 1e-3
        self.detection = det
        self.kf_mean, self.kf_cov = KalmanFilterXYWH().initiate(self.detection.bounding_box_xywhc[:4])
        self.children = None
        self.track_id = -1

    def find_depth(self):
        if(self.children == None):
            return 1

        max_depth = 0
        for child in self.children:
            depth = child.find_depth()
            max_depth = max(depth, max_depth)
        
        return max_depth + 1

    def add_children_to_all_leaves(self, children):
        '''
        add children (a list of Nodes) to all the leaves of the tree starting from self node
        '''
        if(self.children == None):
            # Creating a copy is necessary!
            self.children = copy.deepcopy(children)
            return
        
        for i in range(len(self.children)):
            self.children[i].add_children_to_all_leaves(children)
    
    def add_children_to_all_leaves_with_cost(self, children, motion_cost_fn, ancestors=[]):
        '''
        add children (a list of Nodes) to all the leaves of the tree starting from self node
        '''
        ancestors.append(self)
        if(self.children == None):
            # Creating a copy is necessary!
            self.children = copy.deepcopy(children)
            for child in self.children:
                ancestors.append(child)
                mean, cov = self.kf_mean, self.kf_cov
                child.cond_mot_score, child.mot_prune_score, mean, cov = motion_cost_fn(mean, cov, child.detection, self.det_id)
                child.mot_score = child.cond_mot_score + self.mot_score
                child.kf_mean, child.kf_cov = mean, cov
                child.track_id = self.track_id
                ancestors.pop()
            ancestors.pop()
            return
        
        for i in range(len(self.children)):
            self.children[i].add_children_to_all_leaves_with_cost(children, motion_cost_fn, ancestors=ancestors)
        ancestors.pop()

    def __repr__(self):
        return f"{self.frame_num}-{self.det_id}"
    
    def give_all_paths(self):
        '''
        returns a list of all the paths in the tree
        '''
        if(self.children == None):
            return [[self]]
        
        all_paths = []
        for child in self.children:
            child_paths = child.give_all_paths()
            for path in child_paths:
                all_paths.append([self] + path)
        
        return all_paths

    def prune_paths(self, path, prune_fn):
        '''
        path is the path of nodes till now
        prune_fn returns 1 if the path is to be pruned, else 0
        returns a list of all the paths in the tree
        '''
        if(self.children == None):
            to_prune = prune_fn(path + [self])
            return to_prune # Return to our parent

        i = 0
        while(i < len(self.children)):
            ret = self.children[i].prune_paths(path + [self], prune_fn)
            if(ret >= 1):
                del self.children[i] # Remove the node from the list of children, hence dissolving the path
                i -= 1
            i += 1

        # Don't return anything meaningful to our parent
        return -1

    def prune_all_children(self):
        '''
        recursively prunes all the children of the tree node
        '''
        if(self.children == None):
            return

        while(len(self.children)):
            self.children[0].prune_all_children()
            del self.children[0] # Remove the node from the list of children, hence dissolving the path

    def prune_except(self, track):
        '''
        path is the path of nodes
        removes all tracks except this one
        '''
        if(self.det_id == track[0].det_id and self.frame_num == track[0].frame_num):
            if(self.children == None):
                return True # Return to our parent
            i = 0
            while(i < len(self.children)):
                ret = self.children[i].prune_except(track[1:])
                if not ret:
                    del self.children[i]
                    i -= 1
                i += 1
            return True
        else:
            self.prune_all_children()
            return False # Indicate to our parent that we want to delete this child

class Tracker:
    def __init__(self, name, max_tree_depth, cfg):
        self.name = name
        self.cfg = cfg
        self.detections = []
        self.kalman_filter = KalmanFilterXYWH()
        self.max_det_per_frame = 50
        self.miss_prob = np.log(1 - 0.9)
        self.next_usable_track_id = 0
        self.frame_num = 0
        self.trees = []
        self.max_miss_count = self.cfg.get("max_miss_count", 20)
        self.max_tree_depth = max_tree_depth

        if(name == "no_helmet"):
            self.motion_gating_threshold = cfg.get("no_helmet_motion_gating_threshold", 250)
            self.motion_weight = cfg.get("no_helmet_motion_weight", 1)
        elif(name == "license_plate"):
            self.motion_gating_threshold = cfg.get("license_plate_motion_gating_threshold", 250)
            self.motion_weight = cfg.get("license_plate_motion_weight", 1)

    def track(self, detections, frame):
        '''
        detections are the new detections in the frame. Assuming a new scan from the sensor. It is a numpy array of shape (N, 5)
        representing the coordinates of the bounding box detection in (xyxy format) as well as the confidence of the detection.
        
        frame is a numpy array of shape (H, W, 3) representing the image of the environment

        The function updates the Tracker with the new detections
        '''
        self.frame_num += 1
        
        all_dets = []
        all_nodes = []
        for i, det in enumerate(detections):
            detection = Detection(det, i)
            node = Node(self.frame_num, detection, i)
            all_nodes.append(node)
            all_dets.append(detection)
        
        # If there are no detections, add a dummy detection to each tree
        if(len(all_dets) == 0):
            dummy_det = Detection(np.array([0, 0, 0, 0, 0]), -1)
            dummy_node = Node(self.frame_num, dummy_det, -1)
            for tree in self.trees:
                tree.add_children_to_all_leaves([dummy_node])
            return [], [], [], [], []
        
        # If there are no trees, create a new tree for each detection
        if(len(self.trees) == 0):
            for i, node in enumerate(all_nodes):
                node.track_id = self.next_usable_track_id
                self.next_usable_track_id += 1
                self.trees.append(node)
            return all_dets, all_nodes, [], [], []
        
        # Add the new detections to each tree
        for tree in self.trees:
            tree.add_children_to_all_leaves_with_cost(all_nodes, self.motion_conditional)
        
        # Prune the trees
        for tree in self.trees:
            tree.prune_paths([], self.prune_condition_function)
        
        # Get all paths from each tree
        all_paths = []
        for tree in self.trees:
            all_paths.extend(tree.give_all_paths())
        
        # Get all costs for each path
        all_costs = []
        for path in all_paths:
            all_costs.append(self.compute_dist_for_track(path))
        
        # Solve the optimization problem
        selected_paths = self.solve_mwis_problem(all_paths, all_costs)
        
        # Update the trees
        self.after_n_scan(selected_paths)
        
        # Return the selected detections and nodes
        selected_dets = []
        selected_nodes = []
        for path in selected_paths:
            if(path[-1].det_id != -1):
                selected_dets.append(path[-1].detection)
                selected_nodes.append(path[-1])
        
        return selected_dets, selected_nodes, [], [], []
    
    def motion_conditional(self, mean, covariance, detection, det_id):
        '''
        Compute the motion conditional probability for a detection
        '''
        if(det_id == -1):
            return self.miss_prob, 0, mean, covariance
        
        # Predict the state
        mean, covariance = self.kalman_filter.predict(mean, covariance)
        
        # Project the state to the measurement space
        projected_mean, projected_covariance = self.kalman_filter.project(mean, covariance)
        
        # Compute the gating distance
        gating_distance = self.kalman_filter.gating_distance(projected_mean, projected_covariance, detection.bounding_box_xywhc[:4])
        
        # If the gating distance is too large, return a low probability
        if(gating_distance > self.motion_gating_threshold):
            return self.miss_prob, 0, mean, covariance
        
        # Update the state
        mean, covariance = self.kalman_filter.update(mean, covariance, detection.bounding_box_xywhc[:4])
        
        # Compute the motion score
        motion_score = -gating_distance
        
        return motion_score, 0, mean, covariance
    
    def compute_dist_for_track(self, track):
        '''
        Compute the distance for a track
        '''
        motion_dist_probs = []
        for i in range(1, len(track)):
            motion_dist_probs.append(self.motion_conditional(track[i-1].kf_mean, track[i-1].kf_cov, track[i].detection, track[i].det_id)[0])
        
        motion_dist_probs = np.array(motion_dist_probs)
        motion_dist_probs = motion_dist_probs[motion_dist_probs != self.miss_prob]
        
        motion_dist = motion_dist_probs.sum()
        
        return motion_dist
    
    def prune_condition_function(self, path):
        '''
        Condition for pruning the tracks (returns 1 if track is to be pruned else 0)
        '''
        if(path[-1].det_id == -1):
            return 0
        
        if(len(path) <= 2):
            return 0
        
        gating_cond = (path[-1].mot_prune_score < 0)
        
        if(gating_cond):
            return 1
        else:
            return 0
    
    def solve_mwis_problem(self, paths, costs):
        '''
        Solve the maximum weight independent set problem
        '''
        # Create a graph where each node represents a path
        # Two paths are connected if they share a detection
        graph = np.zeros((len(paths), len(paths)))
        for i in range(len(paths)):
            for j in range(i+1, len(paths)):
                if(self.same_or_not(paths[i], paths[j])):
                    graph[i, j] = 1
                    graph[j, i] = 1
        
        # Solve the maximum weight independent set problem
        selected = []
        weights = np.array(costs)
        
        while(np.sum(weights) > 0):
            # Select the node with the highest weight
            node = np.argmax(weights)
            selected.append(node)
            
            # Remove the selected node and its neighbors
            weights[node] = 0
            for i in range(len(paths)):
                if(graph[node, i] == 1):
                    weights[i] = 0
        
        return [paths[i] for i in selected]
    
    def same_or_not(self, path1, path2):
        '''
        Check if two paths share a detection
        '''
        for i in range(len(path1)):
            for j in range(len(path2)):
                if(path1[i].det_id != -1 and path1[i].det_id == path2[j].det_id):
                    return True
        return False
    
    def after_n_scan(self, tracks):
        '''
        Update the trees after n-scan pruning
        '''
        # Remove all trees
        self.trees = []
        
        # Create a new tree for each track
        for track in tracks:
            if(track[-1].det_id != -1):
                track[-1].track_id = self.next_usable_track_id
                self.next_usable_track_id += 1
                self.trees.append(track[-1])

class AssocTracker:
    def __init__(self, cfg):
        if(type(cfg) == str):
            if(cfg.endswith("json")):
                cfg = self.read_json_cfg(cfg)

        self.cfg = cfg
        self.frame_num = 0
        self.max_miss_count = self.cfg.get("max_miss_count", 20)
        self.max_tree_depth = self.cfg.get("max_tree_depth", 25)
        assert self.max_tree_depth > self.max_miss_count, "Max Tree Depth should be more than the max miss count"
        
        self.no_helmet_tracker = Tracker("no_helmet", self.max_tree_depth, cfg)
        self.license_plate_tracker = Tracker("license_plate", self.max_tree_depth, cfg)
        
        self.curr_no_helmet_tracks = {}
        self.curr_license_plate_tracks = {}
        self.curr_assocs = {} # license_plate to no_helmet
        self.curr_assocs_inv = {} # no_helmet to license_plate

    def read_json_cfg(self, cfg):
        with open(cfg, 'r') as f:
            contents = json.load(f)
        return contents
    
    def track(self, detections, frame):
        '''
        detections are the new detections in the frame. Assuming a new scan from the sensor. It is a numpy array of shape (N, 6)
        representing 
            the coordinates of the bounding box detection in (xyxy format),
            the confidence of the detection,
            the class of the detection (0 for no_helmet, 1 for license_plate)
        
        frame is a numpy array of shape (H, W, 3) representing the image of the environment

        The function updates the Tracker with the new detections
        '''
        self.frame_num += 1
        if(len(detections) == 0):
            no_helmet_dets = []
            license_plate_dets = []
        else:
            cls = detections[:, -1]
            no_helmet_dets = detections[cls == 0][:, :-1]
            license_plate_dets = detections[cls == 1][:, :-1]
        
        print("LEN", len(no_helmet_dets), len(license_plate_dets))
        
        no_helmet_feats, no_helmet_vars, no_helmet_all_paths, no_helmet_obj, no_helmet_costs = self.no_helmet_tracker.track(no_helmet_dets, frame)
        license_plate_feats, license_plate_vars, license_plate_all_paths, license_plate_obj, license_plate_costs = self.license_plate_tracker.track(license_plate_dets, frame)

        # Solve joint tracking problem and association
        tracks_no_helmet, tracks_license_plate = [], []
        
        # Solve the optimization problem for no-helmet tracks
        no_helmet_tracks = self.no_helmet_tracker.solve_mwis_problem(no_helmet_all_paths, no_helmet_costs)
        
        # Solve the optimization problem for license plate tracks
        license_plate_tracks = self.license_plate_tracker.solve_mwis_problem(license_plate_all_paths, license_plate_costs)
        
        # Find associations between no-helmet and license plate tracks
        associations_no_helmets = np.ones(len(no_helmet_tracks)).astype(np.int32) * -1
        
        for i, nh_track in enumerate(no_helmet_tracks):
            # Find nearest license plate in the same frame
            nearest_plate = None
            min_distance = float('inf')
            
            for j, lp_track in enumerate(license_plate_tracks):
                # Calculate distance between centers
                nh_center = [(nh_track[-1].detection.bounding_box_xyxyc[0] + nh_track[-1].detection.bounding_box_xyxyc[2])/2, 
                            (nh_track[-1].detection.bounding_box_xyxyc[1] + nh_track[-1].detection.bounding_box_xyxyc[3])/2]
                lp_center = [(lp_track[-1].detection.bounding_box_xyxyc[0] + lp_track[-1].detection.bounding_box_xyxyc[2])/2, 
                            (lp_track[-1].detection.bounding_box_xyxyc[1] + lp_track[-1].detection.bounding_box_xyxyc[3])/2]
                distance = np.sqrt((nh_center[0] - lp_center[0])**2 + (nh_center[1] - lp_center[1])**2)
                
                if distance < min_distance:
                    min_distance = distance
                    nearest_plate = j
            
            if nearest_plate is not None and min_distance < 200:  # Only associate if within 200 pixels
                associations_no_helmets[i] = nearest_plate
        
        # Update current tracks
        self.update_curr_tracks(no_helmet_tracks, license_plate_tracks, associations_no_helmets)
        
        return no_helmet_feats, license_plate_feats
    
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
        tracks_no_helmet, tracks_license_plate, assocs = [], [], []
        
        # Get all tracks
        for tid, track in self.curr_no_helmet_tracks.items():
            tracks_no_helmet.append(track)
        
        for tid, track in self.curr_license_plate_tracks.items():
            tracks_license_plate.append(track)
        
        # Get associations
        assoc_no_helmets = np.ones(len(self.curr_no_helmet_tracks), dtype=np.int32) * -1
        assoc_license_plates = np.arange(len(self.curr_license_plate_tracks), dtype=np.int32)

        for lp_id, no_helmet_dict in self.curr_assocs.items():
            for nh_id, _ in no_helmet_dict.items():
                assoc_no_helmets[nh_id] = lp_id

        license_plate_no_helmet_count = np.zeros(len(self.curr_license_plate_tracks), dtype=np.int32)
        for i in range(len(license_plate_no_helmet_count)):
            license_plate_no_helmet_count[i] = len(np.where(assoc_no_helmets == i)[0])

        return tracks_no_helmet, tracks_license_plate, license_plate_no_helmet_count

class DetectionTracker:
    def __init__(self):
        self.detections = {
            'no_helmets': [],
            'license_plates': []
        }
        self.frame_count = 0
        self.associations = []
        # Map class IDs to their corresponding detection keys
        self.class_id_to_key = {
            2: 'no_helmets',  # no-helmet
            3: 'license_plates'  # license plate
        }
        self.class_frame_ids = {
            'no_helmets': 1,
            'license_plates': 1
        }
        
    def update(self, detections, timestamp):
        """Store detections with their information"""
        self.frame_count += 1
        
        for det in detections:
            class_key = self.class_id_to_key[det['class_id']]
            detection_info = {
                'frame_id': self.class_frame_ids[class_key],
                'timestamp': timestamp.isoformat(),
                'box': det['box'],
                'confidence': det['conf']
            }
            
            if det['class_id'] == 2:  # No helmet
                self.detections['no_helmets'].append(detection_info)
                self.class_frame_ids['no_helmets'] += 1
            elif det['class_id'] == 3:  # License plate
                if det.get('text'):  # Only store if OCR text is available
                    detection_info['text'] = det['text']
                self.detections['license_plates'].append(detection_info)
                self.class_frame_ids['license_plates'] += 1
    
    def get_formatted_data(self):
        """Get formatted detection data"""
        return {
            'frame_count': self.frame_count,
            'detections': self.detections,
            'associations': self.associations
        }

def parse_arguments():
    parser = argparse.ArgumentParser(description='ANPR System - Process video for no-helmet and license plate detection with tracking')
    parser.add_argument('--input_path', type=str, required=True,
                      help='Path to the input video file')
    parser.add_argument('--output_dir', type=str, default='output',
                      help='Directory to save output files (default: output)')
    parser.add_argument('--tracker_config', type=str, default='tracker/assoc_tracker/base_cfg.json',
                      help='Path to tracker configuration file (JSON format)')
    return parser.parse_args()

def preprocess_plate(plate_img):
    """Preprocess the plate image for better OCR"""
    # Resize to standard size
    plate_img = cv2.resize(plate_img, (300, 100))
    
    # Convert to grayscale
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter to reduce noise while preserving edges
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY, 11, 2)
    
    return thresh

def paddle_ocr(frame, x1, y1, x2, y2, ocr):
    """Enhanced OCR processing with better text extraction"""
    try:
        # Extract the plate region
        plate_img = frame[y1:y2, x1:x2]
        
        # Preprocess the plate image
        processed_plate = preprocess_plate(plate_img)
        
        # Perform OCR with multiple attempts
        results = []
        
        # Try with original image
        result1 = ocr.ocr(plate_img, det=False, rec=True, cls=False)
        if result1:
            results.extend(result1)
            
        # Try with processed image
        result2 = ocr.ocr(processed_plate, det=False, rec=True, cls=False)
        if result2:
            results.extend(result2)
        
        best_text = ""
        best_score = 0
        
        for r in results:
            scores = r[0][1]
            if np.isnan(scores):
                scores = 0
            else:
                scores = int(scores * 100)
            if scores > best_score and scores > 30:  # Lower threshold for better detection
                best_score = scores
                best_text = r[0][0]
        
        # Clean the text
        pattern = re.compile('[\W]')
        text = pattern.sub('', best_text)
        text = text.replace("???", "")
        text = text.replace("O", "0")
        text = text.replace("粤", "")
        
        # Basic validation
        if len(text) >= 5 and len(text) <= 10:  # Reasonable length for license plates
            return str(text)
        return ""
    except Exception as e:
        print(f"Error in OCR processing: {str(e)}")
        return ""

def save_json(detection_data, output_dir, input_file=None):
    """Save detection data to JSON file"""
    os.makedirs(output_dir, exist_ok=True)
    
    if input_file:
        # Use input filename for JSON file
        input_filename = os.path.splitext(os.path.basename(input_file))[0]
        output_file = os.path.join(output_dir, f"tracking_{input_filename}.json")
    else:
        # For video processing, use timestamp
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        output_file = os.path.join(output_dir, f"tracking_{timestamp}.json")
    
    # Save to file
    with open(output_file, 'w') as f:
        json.dump(detection_data, f, indent=2)
    
    print(f"Tracking data saved to: {output_file}")

def process_video(video_path, model, ocr, output_dir, tracker_config):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video at {video_path}")
        return
    
    # Initialize detection tracker
    tracker = DetectionTracker()
    
    # Initialize the association tracker
    assoc_tracker = AssocTracker(tracker_config)
    
    startTime = datetime.now()
    count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        currentTime = datetime.now()
        count += 1
        print(f"Frame Number: {count}")
        
        # Resize frame for better detection
        frame = cv2.resize(frame, (1280, 720))
        
        # Detect all classes
        results = model.predict(frame, conf=0.15, iou=0.45)
        
        # Process detections
        detections = []
        masks = []
        cross_masks = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Only process no-helmet and license plate detections
                if class_id not in [2, 3]:  # Skip if not no-helmet or license plate
                    continue
                
                # Draw bounding box with different colors for different classes
                if class_id == 3:  # License plate
                    color = (255, 0, 0)  # Blue
                else:  # No helmet
                    color = (0, 0, 255)  # Red
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Add class label
                label = CLASS_NAME[class_id]
                textSize = cv2.getTextSize(label, 0, fontScale=0.5, thickness=2)[0]
                c2 = x1 + textSize[0], y1 - textSize[1] - 3
                cv2.rectangle(frame, (x1, y1), c2, color, -1)
                cv2.putText(frame, label, (x1, y1 - 2), 0, 0.5, [255,255,255], thickness=1, lineType=cv2.LINE_AA)
                
                # Perform OCR for license plates
                text = ""
                if class_id == 3:  # License plate
                    text = paddle_ocr(frame, x1, y1, x2, y2, ocr)
                    if text:
                        # Draw license plate text
                        textSize = cv2.getTextSize(text, 0, fontScale=0.5, thickness=2)[0]
                        c2 = x1 + textSize[0], y2 + textSize[1] + 3
                        cv2.rectangle(frame, (x1, y2), c2, (255, 0, 0), -1)
                        cv2.putText(frame, text, (x1, y2 + textSize[1]), 0, 0.5, [255,255,255], thickness=1, lineType=cv2.LINE_AA)
                
                # Create detection array for AssocTracker
                # Format: [x1, y1, x2, y2, conf, class_id]
                # Note: AssocTracker expects class_id 0 for no-helmet and 1 for license plate
                det_class_id = class_id - 2  # Convert to 0-based index for AssocTracker
                detections.append([x1, y1, x2, y2, conf, det_class_id])
                
                # Create dummy masks for now (can be improved with actual segmentation)
                mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                mask[y1:y2, x1:x2] = 255
                masks.append(mask)
                
                # Create dummy cross masks
                cross_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                cross_mask[y1:y2, x1:x2] = 255
                cross_masks.append(cross_mask)
                
                # Store detection for our tracker
                tracker_det = {
                    'class_id': class_id,
                    'box': [x1, y1, x2, y2],
                    'conf': conf,
                    'text': text
                }
                tracker.update([tracker_det], currentTime)
        
        # Update association tracking
        if detections:
            # Convert to numpy arrays
            detections = np.array(detections)
            masks = np.array(masks)
            cross_masks = np.array(cross_masks)
            
            # Update the association tracker
            no_helmet_feats, license_plate_feats = assoc_tracker.track(detections, frame)
            
            # Get current tracks and associations
            no_helmet_tracks, license_plate_tracks, license_plate_no_helmet_count = assoc_tracker.give_tracks()
            
            # Store associations
            for i, nh_track in enumerate(no_helmet_tracks):
                if i < len(license_plate_no_helmet_count) and license_plate_no_helmet_count[i] > 0:
                    # Find the associated license plate
                    for j, lp_track in enumerate(license_plate_tracks):
                        if j < len(license_plate_no_helmet_count) and license_plate_no_helmet_count[j] > 0:
                            # Get the last detection from each track
                            nh_det = nh_track[-1]
                            lp_det = lp_track[-1]
                            
                            # Calculate distance between centers
                            nh_center = [(nh_det.detection.bounding_box_xyxyc[0] + nh_det.detection.bounding_box_xyxyc[2])/2, 
                                        (nh_det.detection.bounding_box_xyxyc[1] + nh_det.detection.bounding_box_xyxyc[3])/2]
                            lp_center = [(lp_det.detection.bounding_box_xyxyc[0] + lp_det.detection.bounding_box_xyxyc[2])/2, 
                                        (lp_det.detection.bounding_box_xyxyc[1] + lp_det.detection.bounding_box_xyxyc[3])/2]
                            distance = np.sqrt((nh_center[0] - lp_center[0])**2 + (nh_center[1] - lp_center[1])**2)
                            
                            # Store association
                            tracker.associations.append({
                                'frame_id': count,
                                'timestamp': currentTime.isoformat(),
                                'no_helmet_box': nh_det.detection.bounding_box_xyxyc[:4].tolist(),
                                'license_plate_box': lp_det.detection.bounding_box_xyxyc[:4].tolist(),
                                'license_plate_text': lp_det.detection.text if hasattr(lp_det.detection, 'text') else '',
                                'distance': distance
                            })
        
        # Save data every 20 seconds
        if (currentTime - startTime).seconds >= 20:
            detection_data = tracker.get_formatted_data()
            save_json(detection_data, output_dir)
            print(f"Total detections in self interval:")
            print(f"- No Helmets: {len(detection_data['detections']['no_helmets'])}")
            print(f"- License Plates: {len(detection_data['detections']['license_plates'])}")
            print(f"- Associations: {len(detection_data['associations'])}")
            startTime = currentTime
        
        # Save processed frame
        if count % 30 == 0:  # Save every 30th frame
            output_image = os.path.join(output_dir, f"processed_frame_{count}.jpg")
            cv2.imwrite(output_image, frame)
            print(f"Saved processed frame: {output_image}")
        
        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def main():
    args = parse_arguments()
    
    # Initialize the YOLOv10 Model
    model = YOLO("best.pt")
    # Initialize the Paddle OCR with optimized parameters
    ocr = PaddleOCR(use_angle_cls=True, use_gpu=False, lang='en', 
                    det_db_thresh=0.2, det_db_box_thresh=0.2)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process video
    process_video(args.input_path, model, ocr, args.output_dir, args.tracker_config)

if __name__ == "__main__":
    main()  