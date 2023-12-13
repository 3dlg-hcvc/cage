import sys, os, json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from iou import sampling_iou
from objects.dict_utils import get_bbox_vertices, get_base_part_idx
from tqdm import tqdm
from copy import deepcopy
import numpy as np
import quaternion
'''
AOR: Average Overlapping Ratio
- compute the vIoU between the sibling parts of the objects
'''
def transform_all_parts(bbox_vertices, obj_dict, joint_state, use_plucker=False, dry_run=True):
    """
    Transform all parts of the object according to the joint state\n

    - bbox_vertices: the bounding box vertices of the object in rest pose in the form: [[x0, y0, z0], [x1, y1, z1], ...]\n
    - obj_dict: the object dictionary\n
    - joint_state: the joint state in the range of [0, 1]\n
    - use_plucker (optional): whether to use plucker coordinate to transform the parts\n
    - dry_run (optional): if True, only return the transformation matrices without transforming the parts\n

    Return:\n
    - part_transformations: the transformation matrices used to transform the parts\n
    """

    # Get a visit order of the parts such that children parts are visited before parents
    part_visit_order = []
    base_idx = get_base_part_idx(obj_dict)
    indices_to_visit = [base_idx]
    while len(indices_to_visit) > 0: # Breadth-first traversal
        current_idx = indices_to_visit.pop(0)
        part_visit_order.append(current_idx)
        indices_to_visit += obj_dict["diffuse_tree"][current_idx]["children"]
    part_visit_order.reverse()

    part_transformations = [[] for _ in range(len(obj_dict["diffuse_tree"]))]

    # Transform the parts in the visit order - children first, then parents
    for i in part_visit_order:
        part = obj_dict["diffuse_tree"][i]
        joint = part["joint"]
        children_idxs = part["children"]
        
        # Store the transformation used to transform the part and its children
        applied_tramsformation_matrix = np.eye(4)
        applied_rotation_axis_origin = np.array([np.nan, np.nan, np.nan])
        applied_transformation_type = "none"
        
        if not use_plucker: # Direct translation and rotation
            if joint["type"] == "prismatic":
                # Translate the part and its children
                translation = np.array(joint["axis"]["direction"]) * joint["range"][1] * joint_state

                if not dry_run:
                    bbox_vertices[[i] + children_idxs] += translation
                
                # Store the transformation used
                applied_tramsformation_matrix[:3, 3] = translation
                applied_transformation_type = "translation"
                
            elif joint["type"] == "revolute" or joint["type"] == "continuous":
                if joint["type"] == "revolute":
                    rotation_radian = np.radians(joint["range"][1] * joint_state) 
                else:
                    rotation_radian = np.radians(360 * joint_state)
                
                # Prepare the rotation matrix via axis-angle representation and quaternion
                rotation_axis_origin = np.array(joint["axis"]["origin"])
                rotation_axis_direction = np.array(joint["axis"]["direction"]) / np.linalg.norm(joint["axis"]["direction"])
                rotation_matrix = quaternion.as_rotation_matrix(quaternion.from_rotation_vector(rotation_radian * rotation_axis_direction))
                
                if not dry_run:
                    # Rotate the part and its children
                    vertices_to_rotate = (bbox_vertices[[i] + children_idxs] - rotation_axis_origin)
                    bbox_vertices[[i] + children_idxs] = np.matmul(rotation_matrix, vertices_to_rotate.transpose([0, 2, 1])).transpose([0, 2, 1]) + rotation_axis_origin

                # Store the transformation used
                applied_tramsformation_matrix[:3, :3] = rotation_matrix
                applied_rotation_axis_origin = rotation_axis_origin
                applied_transformation_type = "rotation"
                

        else: # Translation and rotation together using the plucker coordinate as in NAP
            plucker_direction = np.array(joint["axis"]["plucker"])[:3]
            plucker_moment = np.array(joint["axis"]["plucker"])[3:]
            translation_distance = joint["raw_ranges"][0][1] * joint_state
            rotation_radian = np.radians(joint["raw_ranges"][1][1] * joint_state)

            # Prepare the transformation matrix via plucker coordinate using equation (1) in NAP
            transformation_matrix = np.eye(4)
            rotation_matrix = quaternion.as_rotation_matrix(quaternion.from_rotation_vector(rotation_radian * plucker_direction))
            translation = (np.eye(3) - rotation_matrix) @ np.cross(plucker_direction, plucker_moment) + plucker_direction * translation_distance
            transformation_matrix[:3, :3] = rotation_matrix
            transformation_matrix[:3, 3] = translation

            if not dry_run:
                # Transform the part and its children via homogeneous coordinates
                vertices_to_transform = np.concatenate([bbox_vertices[[i] + children_idxs], np.ones((len([i] + children_idxs), 8, 1))], axis=2)
                bbox_vertices[[i] + children_idxs] = np.matmul(transformation_matrix, vertices_to_transform.transpose([0, 2, 1])).transpose([0, 2, 1])[:, :, :3]

            # Store the transformation used
            applied_tramsformation_matrix = transformation_matrix
            applied_transformation_type = "plucker"

        # Record the transformation used
        if not applied_transformation_type == "none":
            record = {
                "type": applied_transformation_type,
                "matrix": applied_tramsformation_matrix,
                "rotation_axis_origin": applied_rotation_axis_origin
            }
            for idx in [i] + children_idxs:
                part_transformations[idx].append(record)
        
    return part_transformations
   
def AOR(tgt, num_states=20, transform_use_plucker=False):
    tree = tgt["diffuse_tree"]
    states = np.linspace(0, 1, num_states)
    original_bbox_vertices = np.array([get_bbox_vertices(tgt, i) for i in range(len(tgt["diffuse_tree"]))]) 
    
    ious = []
    for state_idx, state in enumerate(states):
        ious_per_state = []
        bbox_vertices = deepcopy(original_bbox_vertices)
        part_trans = transform_all_parts(bbox_vertices, tgt, state, transform_use_plucker)
        for node in tree:
            children = node['children']
            num_children = len(children)
            if num_children < 2:
                continue
            for i in range(num_children-1):
                for j in range(i+1, num_children):
                    child_id = children[i]
                    sibling_id = children[j]
                    bbox_v_0 = deepcopy(bbox_vertices[child_id])
                    bbox_v_1 = deepcopy(bbox_vertices[sibling_id])
                    iou = sampling_iou(bbox_v_0, bbox_v_1, part_trans[child_id], part_trans[sibling_id], num_samples=10000)
                    if np.isnan(iou):
                        continue
                    ious_per_state.append(iou)
        if len(ious_per_state) > 0:
            ious.append(np.mean(ious_per_state))
    if len(ious) == 0:
        return -1
    return np.mean(ious)

def compute_AOR(gen_cache):
    aors = []
    for gen_id in tqdm(gen_cache):
        gen = gen_cache[gen_id]
        aor = AOR(gen)
        if aor != -1:
            aors.append(aor)
    return np.mean(aors)
            
def eval_uncond(gen_root):
    gen_cache = {}
    for f in os.listdir(gen_root):
        if f.endswith('.json'):
            gen_id = f.split('.')[0]
            raw = json.load(open(os.path.join(gen_root, f), 'r'))
            gen_cache[gen_id] = raw

    aor = compute_AOR(gen_cache)
    print(f'AOR: {aor}')

