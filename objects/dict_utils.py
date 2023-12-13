import numpy as np

def get_base_part_idx(obj_dict):
    """
    Get the index of the base part in the object dictionary\n

    - obj_dict: the object dictionary\n

    Return:\n
    - base_part_idx: the index of the base part
    """

    # Adjust for NAP's corner case
    base_part_ids = np.where([part["parent"] == -1 for part in obj_dict["diffuse_tree"]])[0]
    if len(base_part_ids) > 0:
        return base_part_ids[0].item()
    else:
        raise ValueError("No base part found")

def get_bbox_vertices(obj_dict, part_idx):
    """
    Get the 8 vertices of the bounding box\n
    The order of the vertices is the same as the order that pytorch3d.ops.box3d_overlap expects\n
    (This order is not necessary since we are not using pytorch3d.ops.box3d_overlap anymore)\n

    - bbox_center: the center of the bounding box in the form: [cx, cy, cz]\n
    - bbox_size: the size of the bounding box in the form: [lx, ly, lz]\n

    Return:\n
    - bbox_vertices: the 8 vertices of the bounding box in the form: [[x0, y0, z0], [x1, y1, z1], ...]
    """

    part = obj_dict["diffuse_tree"][part_idx]
    bbox_center = np.array(part["aabb"]["center"])
    bbox_size_half = np.array(part["aabb"]["size"]) / 2

    bbox_vertices = np.zeros((8, 3))

    # Get the 8 vertices of the bounding box in the order that pytorch3d.ops.box3d_overlap expects:
    # 0: (x0, y0, z0)    # 1: (x1, y0, z0)    # 2: (x1, y1, z0)    # 3: (x0, y1, z0)
    # 4: (x0, y0, z1)    # 5: (x1, y0, z1)    # 6: (x1, y1, z1)    # 7: (x0, y1, z1)
    bbox_vertices[0, :] = bbox_center - bbox_size_half
    bbox_vertices[1, :] = bbox_center + np.array([bbox_size_half[0], -bbox_size_half[1], -bbox_size_half[2]])
    bbox_vertices[2, :] = bbox_center + np.array([bbox_size_half[0], bbox_size_half[1], -bbox_size_half[2]])
    bbox_vertices[3, :] = bbox_center + np.array([-bbox_size_half[0], bbox_size_half[1], -bbox_size_half[2]])
    bbox_vertices[4, :] = bbox_center + np.array([-bbox_size_half[0], -bbox_size_half[1], bbox_size_half[2]])
    bbox_vertices[5, :] = bbox_center + np.array([bbox_size_half[0], -bbox_size_half[1], bbox_size_half[2]])
    bbox_vertices[6, :] = bbox_center + bbox_size_half
    bbox_vertices[7, :] = bbox_center + np.array([-bbox_size_half[0], bbox_size_half[1], bbox_size_half[2]])

    return bbox_vertices

def compute_overall_bbox_size(obj_dict):
    """
    Compute the overall bounding box size of the object\n

    - obj_dict: the object dictionary\n

    Return:\n
    - bbox_size: the overall bounding box size in the form: [lx, ly, lz]
    """

    bbox_min = np.zeros((len(obj_dict["diffuse_tree"]), 3))
    bbox_max = np.zeros((len(obj_dict["diffuse_tree"]), 3))

    # For each part, compute the bounding box and store the min and max vertices
    for part_idx, part in enumerate(obj_dict["diffuse_tree"]):
        bbox_center = np.array(part["aabb"]["center"])
        bbox_size_half = np.array(part["aabb"]["size"]) / 2
        bbox_min[part_idx] = bbox_center - bbox_size_half
        bbox_max[part_idx] = bbox_center + bbox_size_half
    
    # Compute the overall bounding box size
    bbox_min = np.min(bbox_min, axis=0)
    bbox_max = np.max(bbox_max, axis=0)
    bbox_size = bbox_max - bbox_min
    return bbox_size

def remove_handles(obj_dict):
    """
    Remove the handles from the object dictionary and adjust the id, parent, and children of the parts\n

    - obj_dict: the object dictionary\n

    Return:\n
    - obj_dict: the object dictionary without the handles
    """

    # Find the indices of the handles
    handle_idxs = np.array([i for i in range(len(obj_dict["diffuse_tree"])) 
                            if obj_dict["diffuse_tree"][i]["name"] == "handle"
                            and obj_dict["diffuse_tree"][i]["parent"] != -1])    # Added to avoid corner case of NAP where the handle is the base part

    # Remove the handles from the object dictionary and adjust the id, parent, and children of the parts
    for handle_idx in handle_idxs:
        handle = obj_dict["diffuse_tree"][handle_idx]
        parent_idx = handle["parent"]
        if handle_idx in obj_dict["diffuse_tree"][parent_idx]["children"]:
            obj_dict["diffuse_tree"][parent_idx]["children"].remove(handle_idx)
        obj_dict["diffuse_tree"].pop(handle_idx)

        # Adjust the id, parent, and children of the parts
        for part in obj_dict["diffuse_tree"]:
            if part["id"] > handle_idx:
                part["id"] -= 1
            if part["parent"] > handle_idx:
                part["parent"] -= 1
            for i in range(len(part["children"])):
                if part["children"][i] > handle_idx:
                    part["children"][i] -= 1
        
        handle_idxs -= 1

    return obj_dict

def rescale_object(object_dict, scale_factor):
    """
    Rescale the object as a whole\n

    - object_dict: the object dictionary\n
    - scale_factor: the scale factor to rescale the object
    """

    for part in object_dict["diffuse_tree"]:
        part["aabb"]["center"] = np.array(part["aabb"]["center"]) * scale_factor
        part["aabb"]["size"] = np.array(part["aabb"]["size"]) * scale_factor
        if part["joint"]["type"] != "fixed":
            part["joint"]["axis"]["origin"] = np.array(part["joint"]["axis"]["origin"]) * scale_factor

def find_part_mapping(obj1_dict, obj2_dict):
    """
    Find the correspondences from the first object to the second object based on closest bbox centers\n

    - obj1_dict: the first object dictionary\n
    - obj2_dict: the second object dictionary\n

    Return:\n
    - mapping: the mapping from the first object to the second object in the form: [[obj_part_idx, distance], ...]
    """

    # Initialize the distances to be +inf
    mapping = np.ones((len(obj1_dict["diffuse_tree"]), 2)) * np.inf

    # For each part in the first object, find the closest part in the second object based on the bounding box center
    for req_part_idx, req_part in enumerate(obj1_dict["diffuse_tree"]):
        for obj_part_idx, obj_part in enumerate(obj2_dict["diffuse_tree"]):
            distance = np.linalg.norm(np.array(req_part["aabb"]["center"]) - np.array(obj_part["aabb"]["center"]))
            if distance < mapping[req_part_idx, 1]:
                mapping[req_part_idx, :] = [obj_part_idx, distance]
    
    return mapping