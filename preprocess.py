import os
import json
import argparse
import numpy as np
import open3d as o3d
import networkx as nx
from copy import deepcopy
    
def _load_part_meshes(node_i, info):
    if 'children' not in node_i:  # leaf node
        if 'objs' in node_i:
            info[node_i['id']] = [f'textured_objs/{file}.obj' for file in node_i['objs']]
        return
    for child in node_i['children']:
        _load_part_meshes(child, info)

def _convert_joint(src_node, dst_node):
    if src_node['joint'] in ['free', 'heavy', 'static']:
        dst_node['joint']['type'] = 'fixed'
        dst_node['joint']['range'] = [0., 0.]
        dst_node['joint']['axis'] = {'origin': [0., 0., 0.], 'direction': [0., 0., 0.]}
    elif src_node['joint'] == 'hinge':
        if src_node['jointData']['limit']['noLimit']:
            dst_node['joint']['type'] = 'continuous'
            dst_node['joint']['range'] = [0., 0.]
        else:
            dst_node['joint']['type'] = 'revolute'
            dst_node['joint']['range'] = [src_node['jointData']['limit']['a'], src_node['jointData']['limit']['b']]
        dst_node['joint']['axis'] = src_node['jointData']['axis'] # [origin, direction]
    elif src_node['joint'] == 'slider':
        if "rotates" in src_node['jointData']['limit'] and src_node['jointData']['limit']['rotates']:
            dst_node['joint']['type'] = 'screw'
        else:
            dst_node['joint']['type'] = 'prismatic'
        dst_node['joint']['range'] = [src_node['jointData']['limit']['a'], src_node['jointData']['limit']['b']]
        dst_node['joint']['axis'] = src_node['jointData']['axis'] # [origin, direction]
    else:
        raise ValueError(f"Unsupported joint type: {src_node['joint']}")

def _get_meshes(src_node, dst_node, part_meshes):
    parts = src_node['parts']
    dst_node['objs'] = []
    dst_node['parts'] = []
    for part in parts:
        my_part = {}
        my_part['id'] = part['id']
        my_part['name'] = part['name']
        my_part['objs'] = part_meshes[part['id']]
        dst_node['parts'].append(my_part)
        dst_node['objs'].extend(part_meshes[part['id']])

def _R_axis_angle(axis, angle):
    '''
    Rodrigues' rotation formula
    args:
    * axis: direction unit vector of the axis to rotate about
    * angle: the (degree) angle to rotate with
    return:
    * 4x4 rotation matrix
    '''
    theta = np.deg2rad(angle)
    k = axis / np.linalg.norm(axis)
    kx, ky, kz = k[0], k[1], k[2]
    cos, sin = np.cos(theta), np.sin(theta)
    R = np.eye(4, dtype=np.float32)
    R[0, 0] = cos + (kx**2) * (1 - cos)
    R[0, 1] = kx * ky * (1 - cos) - kz * sin
    R[0, 2] = kx * kz * (1 - cos) + ky * sin
    R[1, 0] = kx * ky * (1 - cos) + kz * sin
    R[1, 1] = cos + (ky**2) * (1 - cos)
    R[1, 2] = ky * kz * (1 - cos) - kx * sin
    R[2, 0] = kx * kz * (1 - cos) - ky * sin
    R[2, 1] = ky * kz * (1 - cos) + kx * sin
    R[2, 2] = cos + (kz**2) * (1 - cos)
    return R

def _transform_to_matrix(joint):
    if joint['type'] == 'revolute':
        center = np.array(joint['axis']['origin'], dtype=np.float32)
        axis = np.array(joint['axis']['direction'], dtype=np.float32)
        R = _R_axis_angle(axis, angle=joint['range'][0])
        Tr = np.eye(4, dtype=np.float32)
        Tr[:3, 3] = -center
        Tl = np.eye(4, dtype=np.float32)
        Tl[:3, 3] = center
        H = np.matmul(Tl, np.matmul(R, Tr))
    elif joint['type'] == 'prismatic' or joint['type'] == 'screw':
        dist = joint['range'][0]
        axis_d = np.array(joint['axis']['direction'])
        axis_d = axis_d / np.linalg.norm(axis_d) # normalize
        H = np.eye(4, dtype=np.float32)
        H[:3, 3] = dist * axis_d
    else:
        raise NotImplementedError
    return H

def _apply_transform(transforms, src_dir, dst_dir):
    '''Apply the transformation to the meshes and save them to the output directory.'''
    out_dir = os.path.join(dst_dir, 'plys')
    os.makedirs(out_dir, exist_ok=True)
    for obj_file, transform in transforms.items():
        fname = obj_file.split('/')[1][:-4]
        mesh = o3d.io.read_triangle_mesh(os.path.join(src_dir, obj_file))
        if transform['flag']:
            mesh.transform(transform['transform'])
        mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(os.path.join(out_dir, f'{fname}.ply'), mesh)

def _collect_transform(data, node, transforms):
    '''
    Collect the transformation matrix for each node in the tree.
    args:
    * data: the data dictionary
    * node: the current node
    * transforms: the dictionary to store the transformation matrix for each node
    '''
    # leaf node
    if len(node['children']) == 0:
        if node['joint']['range'][0] == 0:
            for obj_file in node['objs']:
                transforms[obj_file] = {
                    'flag': False,
                    'transform': np.eye(4, dtype=np.float32),
                }
        else:
            H = _transform_to_matrix(node['joint'])
            for obj_file in node['objs']:
                transforms[obj_file] = {
                    'flag': True,
                    'transform': H,
                }
        return
    # internal nodes
    for child_id in node['children']:
        _collect_transform(data, data['arti_tree'][child_id], transforms)
        if node['joint']['range'][0] == 0:
            for obj_file in node['objs']:
                if obj_file not in transforms:
                    transforms[obj_file] = {
                        'flag': False,
                        'transform': np.eye(4, dtype=np.float32),
                    }
        else:
            H = _transform_to_matrix(node['joint'])
            for obj_file in node['objs']:
                if obj_file not in transforms:
                    transforms[obj_file] = {
                        'flag': True,
                        'transform': H,
                    }
                else:
                    transforms[obj_file]['flag'] = True
                    transforms[obj_file]['transform'] = np.matmul(transforms[obj_file]['transform'], H)


def _reset_state(data, root_node, src_dir, dst_dir):
    # collect the transformation matrix for each node
    reset_transforms = {}
    _collect_transform(data, root_node, reset_transforms)
    # apply the transformation to the meshes and save them to the output directory
    _apply_transform(reset_transforms, src_dir, dst_dir) 
    # update the joint range to [0, ..]
    for node in data['arti_tree']:
        l,r = node['joint']['range']
        node['joint']['range'] = [0, r-l]

def _clean_extend_tree(data):
    '''Clean the arti_tree and extend the actionable parts to the diffuse_tree.'''
    arti_idx = data['arti_tree'][-1]['id'] + 1
    for arti_node in data['diffuse_tree']:
        if arti_node['joint']['type'] != 'fixed':
            # clean: discard prismatic buttons from the articulation tree
            if arti_node['joint']['type'] == 'prismatic' and 'button' in arti_node['name']:
                parent_id = arti_node['parent']
                for node in data['diffuse_tree']:
                    if node['id'] == parent_id:
                        node['children'].remove(arti_node['id'])
                        if 'attach_objs' not in node.keys():
                            node['attach_objs'] = [obj.replace('textured_objs', 'plys')[:-3]+'ply' for obj in arti_node['objs']]
                        else:
                            node['attach_objs'] += [obj.replace('textured_objs', 'plys')[:-3]+'ply' for obj in arti_node['objs']]
                        break
                arti_node['id'] = -1
            # clean: merge parts for the wheel
            elif arti_node['name'] in ['caster_mounting_plate', 'caster_stem', 'brake']:
                parent_id = arti_node['parent']
                children_ids = arti_node['children']
                for node in data['diffuse_tree']:
                    if node['id'] == parent_id:
                        node['children'].remove(arti_node['id'])
                        node['children'] += children_ids
                        if 'attach_objs' not in node.keys():
                            node['attach_objs'] = [obj.replace('textured_objs', 'plys')[:-3]+'ply' for obj in arti_node['objs']]
                        else:
                            node['attach_objs'] += [obj.replace('textured_objs', 'plys')[:-3]+'ply' for obj in arti_node['objs']]
                    elif node['id'] in children_ids:
                        node['parent'] = parent_id
                arti_node['id'] = -1
            # extend: append "handle" parts if any
            if len(arti_node['parts']) > 1:
                for part in arti_node['parts']:
                    if 'handle' == part['name']:
                        new_node = {}
                        new_node['id'] = arti_idx
                        new_node['parent'] = arti_node['id']
                        if arti_idx not in arti_node['children']:
                            data['diffuse_tree'][arti_node['id']]['children'].append(arti_idx)
                        new_node['name'] = part['name']
                        new_node['children'] = []
                        new_node['joint'] = {'type': 'fixed', 'range': [0., 0.]} 
                        new_node['objs'] = part['objs']
                        # remove the part obj from the parent node
                        arti_node['objs'] = list(set(arti_node['objs']) - set(part['objs']))
                        # append the new node to the diffuse_tree
                        data['diffuse_tree'].append(new_node)
                        arti_idx += 1

        arti_node['objs'] = [obj.replace('textured_objs', 'plys')[:-3]+'ply' for obj in arti_node['objs']]

        if 'parts' in arti_node.keys():
            arti_node.pop('parts')
    
    # remove the nodes that are not in the articulation tree
    data['diffuse_tree'] = [i for i in data['diffuse_tree'] if i['id'] != -1]

def _compute_AABB(data, dst_dir):
    for diff_node in data['diffuse_tree']:
        # load the meshes
        meshes = o3d.geometry.TriangleMesh()
        for obj_file in diff_node['objs']:
            fname = obj_file.split('/')[1][:-4] + '.ply'
            mesh = o3d.io.read_triangle_mesh(os.path.join(dst_dir, 'plys', fname))
            mesh.compute_vertex_normals()
            meshes += mesh    
        # compute the AABB
        aabb = meshes.get_axis_aligned_bounding_box()
        size = aabb.get_max_bound() - aabb.get_min_bound()
        center = aabb.get_center()
        diff_node['aabb'] = {
            'center': center.tolist(),
            'size': size.tolist(),
        }

def _clean_semantic_label(data):
    tree = data['diffuse_tree']
    for node in tree:
        if node['parent'] == -1: # base part
            node['name'] = 'base'
        elif node['joint']['type'] != 'fixed': # artiuclated parts
            # merge the node name
            if node['joint']['type'] == 'revolute':
                if any(keyword in node['name'] for keyword in ['door', 'mirror', 'glass', 'board', 'countertop', 'control_panel', 'display_panel', 'other_leaf']):
                    node['name'] = 'door'
                elif 'shelf' in node['name']:
                    node['name'] = 'shelf'
                    
            elif node['joint']['type'] == 'prismatic':
                if any(keyword in node['name'] for keyword in ['drawer', 'display_panel']):
                    node['name'] = 'drawer'
                elif any(keyword in node['name'] for keyword in ['door', 'mirror', 'glass']):
                    node['name'] = 'door'
                elif 'tray' in node['name']:
                    node['name'] = 'tray'
                elif 'shelf' in node['name']:
                    node['name'] = 'shelf'

            elif node['joint']['type'] == 'continuous':
                if 'button' in node['name']:
                    node['name'] = 'knob' 
                elif 'tray' in node['name']:
                    node['name'] = 'tray'
                elif 'wheel' in node['name']:
                    node['name'] = 'wheel'

def _renumber(data):
    '''Renumber the nodes so that the node id is same as the index in the list.'''
    tree = data['diffuse_tree']
    mapping = {-1: -1}
    i = 0
    for node in tree:
        old_id = node['id']
        mapping[old_id] = i
        i += 1
    for node in tree:
        node['id'] = mapping[node['id']]
        node['parent'] = mapping[node['parent']]
        node['children'] = [mapping[c] for c in node['children']]
    # remove the arti_tree
    data.pop('arti_tree')

def _get_tree_hash(data):
    tree = data['diffuse_tree']
    G = nx.DiGraph()
    for node in tree:
        G.add_node(node['id'])
        if node['parent'] != -1:
            G.add_edge(node['id'], node['parent'])
    hashcode = nx.weisfeiler_lehman_graph_hash(G)
    return hashcode


def process_model(model_id, src_dir, dst_dir):
    src_dir = os.path.join(src_dir, model_id)
    assert os.path.exists(src_dir), f"Model directory does not exist: {src_dir}."
    # result.json
    with open(os.path.join(src_dir, 'result.json'), 'r') as f:
        result = json.load(f)
    # meta.json
    with open(os.path.join(src_dir, 'meta.json'), 'r') as f:
        meta = json.load(f)
        category = meta['model_cat']
    # mobility_v2.json
    with open(os.path.join(src_dir, 'mobility_v2.json'), 'r') as f:
        mobility = json.load(f)
    # save directory
    dst_dir = os.path.join(dst_dir, category, model_id)
    os.makedirs(dst_dir, exist_ok=True)
    # step 0: find meshes for parts
    part_meshes = {}
    _load_part_meshes(result[0], part_meshes)
    # step 1: parse articulation
    new_data = {'meta': {'obj_cat': category}, 'arti_tree':[]}
    for node in mobility:
        my_node = {}
        # basic info
        my_node['id'] = node['id']
        my_node['parent'] = node['parent']
        my_node['name'] = node['name']
        my_node['children'] = []
        # convert to joint types: [fixed, revolute, continuous, prismatic, screw]
        my_node['joint'] = {}
        _convert_joint(node, my_node)
        # merge geometry info to my_node
        _get_meshes(node, my_node, part_meshes)
        # append to new_data
        new_data['arti_tree'].append(my_node)
    # step 2: update children relations
    root_node = None
    for node in new_data['arti_tree']:
        if node['parent'] != -1:
            children = new_data['arti_tree'][node['parent']]['children']
            if node['id'] not in children:
                children.append(node['id'])
        else:
            root_node = node
    # step 3: "reset" the parts to the resting state
    _reset_state(new_data, root_node, src_dir, dst_dir)
    # step 4: clean and extend the tree with actionable parts
    new_data['diffuse_tree'] = deepcopy(new_data['arti_tree'])
    _clean_extend_tree(new_data)
    # step 5: compute AABB for each node
    _compute_AABB(new_data, dst_dir)
    # step 6: clean semantic labels
    _clean_semantic_label(new_data)
    # step 7: renumber the nodes
    _renumber(new_data)
    # save the json file
    with open(os.path.join(dst_dir, 'train.json'), 'w') as f:
        json.dump(new_data, f, indent=4)
if __name__ == '__main__':
    '''
    This script is used to preprocess the PartNet-Mobility data into the data format used in CAGE.
    Args:
    * model_id: The model id of the PartNet-Mobility data.
    * src_dir: we will look for the model data in <src_dir>/<model_id>.
    * dst_dir: we will save the processed data in <dst_dir>/<object_category>/<model_id> with the following files:
        * train.json: the json file recording the part hierarchy and articulation information.
        * plys: the directory containing the processed mesh files in .ply format.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, required=True, help='The model id of the PartNet-Mobility data.')
    parser.add_argument('--src_dir', type=str, required=True, help='The source directory of the PartNet-Mobility data.')
    parser.add_argument('--dst_dir', type=str, required=True, help='The destination directory to save the processed data.')
    args = parser.parse_args()

    model_id = args.model_id
    src_dir = args.src_dir
    dst_dir = args.dst_dir

    assert os.path.exists(src_dir), f"The source directory does not exist: {src_dir}"
    os.makedirs(dst_dir, exist_ok=True)

    process_model(model_id, src_dir, dst_dir)
