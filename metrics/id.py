import os
import sys
import torch
import numpy as np
import open3d as o3d
from pytorch3d.structures import Meshes
from pytorch3d.io import load_ply
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import chamfer_distance
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from objects.motions import transform_all_parts

def _load_and_combine_plys(dir, ply_files, delete_temp_files=False, 
                           temp_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")):
    """
    Load and combine the ply files into one PyTorch3D mesh

    - dir: the directory of the object in which the ply files are from\n
    - ply_files: the list of ply files\n
    - delete_temp_files (optional): whether to delete the temp file after loading the mesh\n
    - temp_dir (optional): the directory to store the combined mesh file\n

    Return:\n
    - mesh: one PyTorch3D mesh of the combined ply files
    """
    
    # Create temp directory for storing the combined mesh file and delete it afterwards
    os.makedirs(temp_dir, exist_ok=True)

    # Combine the ply files into one
    full_part_mesh = o3d.geometry.TriangleMesh()
    for ply_file in ply_files:
        mesh = o3d.io.read_triangle_mesh(os.path.join(dir, ply_file))
        full_part_mesh += mesh
    
    # Save the combined mesh file
    temp_file_path = os.path.join(temp_dir, "temp.ply")
    o3d.io.write_triangle_mesh(temp_file_path, full_part_mesh)

    # Load the combined mesh file as PyTorch3D mesh
    vertices, faces = load_ply(temp_file_path)

    # Delete the temp file if specified
    if delete_temp_files:
        os.remove(temp_file_path)
    
    # Create the PyTorch3D mesh
    mesh = Meshes(verts=[vertices], faces=[faces])

    return mesh    

def _compute_chamfer_distance(obj1_part_points, obj2_part_points):
    """
    Compute the chamfer distance between the two set of points representing the two objects

    - obj1_part_points: the set of points representing the first object\n
    - obj2_part_points: the set of points representing the second object\n

    Return:\n
    - distance: the chamfer distance between the two objects
    """

    # Merge the points of all parts into one tensor
    obj1_part_points = obj1_part_points.reshape(-1, 3)
    obj2_part_points = obj2_part_points.reshape(-1, 3)

    # Compute the chamfer distance between the two objects
    with torch.no_grad():
        obj1_part_points = obj1_part_points.cuda()
        obj2_part_points = obj2_part_points.cuda()
        distance, _ = chamfer_distance(obj1_part_points[None, :], obj2_part_points[None, :], batch_reduction=None)
        distance = distance.item()
    
    return distance

def ID(gen_obj_dict, gen_obj_retrieved_parts, gt_obj_dict, gt_obj_path, num_states=5, num_samples=2048):
    """
    Compute the ID metric\n
    This metric is the average chamfer distance between the two objects over a number of articulation states\n

    - gen_obj_dict: the generated object dictionary\n
    - gen_obj_retrieved_parts: the list of retrieved parts of the generated object\n
    - gt_obj_dict: the ground truth object dictionary\n
    - gt_obj_path: the path to the ground truth object\n
    - num_states (optional): the number of articulation states to compute the metric\n
    - num_samples (optional): the number of samples to use\n

    Return:\n
    - score: the metric score, which is the overall average chamfer distance over the states\n
        - The score is in the range of [0, inf), lower is better
    """

    if "diffuse_tree" not in gt_obj_dict: # Rename for compatibility
        gt_obj_dict["diffuse_tree"] = gt_obj_dict.pop("arti_tree")

    # Get the number of parts of the two objects
    gen_obj_num_parts = len(gen_obj_dict["diffuse_tree"])
    gt_obj_num_parts = len(gt_obj_dict["diffuse_tree"])

    # Get the paths of the ply files of the two objects
    gen_obj_part_ply_paths = [{"dir": gen_obj_retrieved_parts[i]["dir"], "files": gen_obj_retrieved_parts[i]["files"]} 
                              for i in range(gen_obj_num_parts)]
    gt_obj_part_ply_paths = [{"dir": gt_obj_path, "files": gt_obj_dict["diffuse_tree"][i]["objs"]} 
                             for i in range(gt_obj_num_parts)]
    
    # Load the ply files of the two objects and sample points from them
    gen_obj_part_points = torch.zeros((gen_obj_num_parts, num_samples, 3))
    for i in range(gen_obj_num_parts):
        part_mesh = _load_and_combine_plys(gen_obj_part_ply_paths[i]["dir"], gen_obj_part_ply_paths[i]["files"])
        gen_obj_part_points[i] = sample_points_from_meshes(part_mesh, num_samples=num_samples).squeeze(0)

    gt_obj_part_points = torch.zeros((gt_obj_num_parts, num_samples, 3))
    for i in range(gt_obj_num_parts):
        part_mesh = _load_and_combine_plys(gt_obj_part_ply_paths[i]["dir"], gt_obj_part_ply_paths[i]["files"])
        gt_obj_part_points[i] = sample_points_from_meshes(part_mesh, num_samples=num_samples).squeeze(0)

    chamfer_distances = np.zeros(num_states)
    joint_states = np.linspace(0, 1, num_states)
    for state_idx, state in enumerate(joint_states):

        # Transform the part point clouds to the current state using the joints
        transform_all_parts(gen_obj_part_points.numpy(), gen_obj_dict, state, dry_run=False)
        transform_all_parts(gt_obj_part_points.numpy(), gt_obj_dict, state, dry_run=False)

        # Compute the chamfer distance between the two objects
        gen_to_gt_distance = _compute_chamfer_distance(gen_obj_part_points, gt_obj_part_points)
        gt_to_gen_distance = _compute_chamfer_distance(gt_obj_part_points, gen_obj_part_points)

        # Store the chamfer distance
        chamfer_distances[state_idx] = (gen_to_gt_distance + gt_to_gen_distance) * 0.5

    # Compute the ID
    score = np.mean(chamfer_distances)

    return score