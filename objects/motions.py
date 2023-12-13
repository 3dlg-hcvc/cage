import os
import sys
import numpy as np
import quaternion
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from objects.dict_utils import get_base_part_idx

def transform_all_parts(part_vertices, obj_dict, joint_state, 
                        use_plucker=False, rotation_fix_range=True, dry_run=True):
    """
    Transform all parts of the object according to the joint state\n

    - part_vertices: vertices of the object in rest pose in the form:\n
        - [K_parts, N_vertices, 3]\n
    - obj_dict: the object dictionary\n
    - joint_state: the joint state in the range of [0, 1]\n
    - use_plucker (optional): whether to use plucker coordinate to transform the parts\n
    - rotation_fix_range (optional): whether to fix the rotation range to 90 degrees for revolute joints\n
    - dry_run (optional): if True, only return the transformation matrices without changing the vertices\n

    Return:\n
    - part_transformations: records of the transformations applied to the parts\n
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
                    part_vertices[[i] + children_idxs] += translation
                
                # Store the transformation used
                applied_tramsformation_matrix[:3, 3] = translation
                applied_transformation_type = "translation"
                
            elif joint["type"] == "revolute" or joint["type"] == "continuous":
                if joint["type"] == "revolute":
                    if not rotation_fix_range:
                        # Use the full range as specified in the object file
                        rotation_radian = np.radians(joint["range"][1] * joint_state) 
                    else: 
                        # Fix the rotation range to 90 degrees
                        rotation_range_sign = np.sign(joint["range"][1])
                        rotation_radian = np.radians(rotation_range_sign * 90 * joint_state)

                else:
                    rotation_radian = np.radians(360 * joint_state)
                
                # Prepare the rotation matrix via axis-angle representation and quaternion
                rotation_axis_origin = np.array(joint["axis"]["origin"])
                rotation_axis_direction = np.array(joint["axis"]["direction"]) / np.linalg.norm(joint["axis"]["direction"])
                rotation_matrix = quaternion.as_rotation_matrix(quaternion.from_rotation_vector(rotation_radian * rotation_axis_direction))
                
                if not dry_run:
                    # Rotate the part and its children
                    vertices_to_rotate = (part_vertices[[i] + children_idxs] - rotation_axis_origin)
                    part_vertices[[i] + children_idxs] = np.matmul(rotation_matrix, vertices_to_rotate.transpose([0, 2, 1])).transpose([0, 2, 1]) + rotation_axis_origin

                # Store the transformation used
                applied_tramsformation_matrix[:3, :3] = rotation_matrix
                applied_rotation_axis_origin = rotation_axis_origin
                applied_transformation_type = "rotation"
                

        else: # Translation and rotation together using the plucker coordinate as in NAP
            plucker_direction = np.array(joint["axis"]["plucker"])[:3]
            plucker_moment = np.array(joint["axis"]["plucker"])[3:]
            translation_distance = joint["raw_ranges"][0][1] * joint_state

            if not rotation_fix_range:
                # Use the full range as specified in the object file
                rotation_radian = np.radians(joint["raw_ranges"][1][1] * joint_state)
            else:
                # Fix the rotation range to 90 degrees
                rotation_range_sign = np.sign(joint["raw_ranges"][1][1])
                rotation_radian = np.radians(rotation_range_sign * 90 * joint_state)

            # Prepare the transformation matrix via plucker coordinate using equation (1) in NAP
            transformation_matrix = np.eye(4)
            rotation_matrix = quaternion.as_rotation_matrix(quaternion.from_rotation_vector(rotation_radian * plucker_direction))
            translation = (np.eye(3) - rotation_matrix) @ np.cross(plucker_direction, plucker_moment) + plucker_direction * translation_distance
            transformation_matrix[:3, :3] = rotation_matrix
            transformation_matrix[:3, 3] = translation

            if not dry_run:
                # Transform the part and its children via homogeneous coordinates
                vertices_to_transform = np.concatenate([part_vertices[[i] + children_idxs], np.ones((len([i] + children_idxs), 8, 1))], axis=2)
                part_vertices[[i] + children_idxs] = np.matmul(transformation_matrix, vertices_to_transform.transpose([0, 2, 1])).transpose([0, 2, 1])[:, :, :3]

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