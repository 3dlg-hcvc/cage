import torch
import models
import trimesh
import numpy as np
import lightning.pytorch as pl
from diffusers import DDPMScheduler
from utils.savermixins import SaverMixin
from utils.refs import sem_ref, joint_ref
from utils.plot import viz_graph, make_grid, add_text
from retrieval.obj_retrieval import find_obj_candidates, pick_and_rescale_parts
from utils.render import rescale_axis, draw_boxes_axiss_anim, get_bbox_mesh_pair, get_axis_mesh

class BaseSystem(pl.LightningModule, SaverMixin):
    def __init__(self, hparams):
        super().__init__()
        self.hparams.update(hparams) 
        self.model = models.make(hparams.model.name, hparams.model)
        self.scheduler = DDPMScheduler(**self.hparams.scheduler.config)
        self.save_hyperparameters()

    def setup(self, stage: str):
        self.set_save_dir(stage) # config the logger dir for images
    
    def configure_optimizers(self):
        raise NotImplementedError
    
    def training_step(self, batch, batch_idx):
        raise NotImplementedError
    
    def validation_step(self, batch, batch_idx):
        raise NotImplementedError
    
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        raise NotImplementedError
    
    # ------------------------------- data converters ------------------------------- #
    def convert_data_range(self, x):
        x = x.reshape(-1, 30) # (K, 30)
        aabb_max = self.convert_format(x[:, 0:3])
        aabb_min = self.convert_format(x[:, 3:6])
        center = (aabb_max + aabb_min) / 2.
        size = (aabb_max - aabb_min).clip(min=1e-3)
        
        j_type = torch.mean(x[:, 6:12], dim=1)
        j_type = self.convert_format((j_type+0.5) * 5).clip(min=1., max=5.).round()

        axis_d = self.convert_format(x[:, 12:15])
        axis_d = axis_d / (np.linalg.norm(axis_d, axis=1, keepdims=True) + np.finfo(float).eps)
        axis_o = self.convert_format(x[:, 15:18])

        j_range = (x[:, 18:20] + x[:, 20:22] + x[:, 22:24]) / 3
        j_range = self.convert_format(j_range).clip(min=-1., max=1.)
        j_range[:, 0] = j_range[:, 0] * 360
        j_range[:, 1] = j_range[:, 1]

        label = torch.mean(x[:, 24:30], dim=1)
        label = self.convert_format((label+0.8) * 5).clip(min=0., max=7.).round()
        return {
            'center': center,
            'size': size,
            'type': j_type,
            'axis_d': axis_d,
            'axis_o': axis_o,
            'range': j_range,
            'label': label
        }
    
    def convert_json_graph_only(self, c, idx):
        out = {'diffuse_tree': []}
        n_nodes = c['n_nodes'][idx].item()
        par = c['parents'][idx].cpu().numpy().tolist()
        adj = c['adj'][idx].cpu().numpy()
        np.fill_diagonal(adj, 0)

        for i in range(n_nodes):
            node = {'id': i}
            node['parent'] = int(par[i])
            node['children'] = [int(child) for child in np.where(adj[i] == 1)[0] if child != par[i]]
            out['diffuse_tree'].append(node)
        return out

    def convert_json(self, x, c, idx):
        n_nodes = c['n_nodes'][idx].item()
        par = c['parents'][idx].cpu().numpy().tolist()
        adj = c['adj'][idx].cpu().numpy()
        np.fill_diagonal(adj, 0)

        # convert the data to original range
        data = self.convert_data_range(x)
        # convert to json format
        out = {'diffuse_tree': []}
        out['meta'] = {
            'obj_cat': c['obj_cat'][idx], 
            'tree_hash': c['tree_hash'][idx]  
        }
        for i in range(n_nodes):
            node = {'id': i}
            node['name'] = sem_ref['bwd'][int(data['label'][i].item())]
            node['parent'] = int(par[i])
            node['children'] = [int(child) for child in np.where(adj[i] == 1)[0] if child != par[i]]
            node['aabb'] = {}
            node['aabb']['center'] = data['center'][i].tolist()
            node['aabb']['size'] = data['size'][i].tolist()
            node['joint'] = {}
            node['joint']['type'] = joint_ref['bwd'][int(data['type'][i].item())]
            if node['joint']['type'] == 'fixed':
                node['joint']['range'] = [0., 0.]
            elif node['joint']['type'] == 'revolute':
                node['joint']['range'] = [0., float(data['range'][i][0])]
            elif node['joint']['type'] == 'continuous':
                node['joint']['range'] = [0., 360.]
            elif node['joint']['type'] == 'prismatic' or node['joint']['type'] == 'screw':
                node['joint']['range'] = [0., float(data['range'][i][1])]
            node['joint']['axis'] = {}
            # relocate the axis to visualize well
            axis_o, axis_d = rescale_axis(int(data['type'][i].item()), data['axis_d'][i], data['axis_o'][i], data['center'][i])
            node['joint']['axis']['direction'] = axis_d
            node['joint']['axis']['origin'] = axis_o

            out['diffuse_tree'].append(node)
        return out

    # ------------------------------- visualizations ------------------------------- #
    def prepare_meshes(self, info_dict):
        '''
        Function to prepare the bbox and axis meshes for visualization

        Args:
        - info_dict (dict): output json containing the graph information
        '''
        tree = info_dict['diffuse_tree']
        bbox_0, bbox_1, axiss, labels, jtypes = [], [], [], [], []
        root_id = 0
        # get root id
        for node in tree:
            if node['parent'] == -1:
                root_id = node['id']        
        for node in tree:
            # retrieve info
            box_cen = np.array(node['aabb']['center'])
            box_size = np.array(node['aabb']['size'])
            jrange = node['joint']['range']
            jtype = node['joint']['type']
            axis_d = np.array(node['joint']['axis']['direction'])
            axis_o = np.array(node['joint']['axis']['origin'])
            label = sem_ref['fwd'][node['name']]
            jtype_id = joint_ref['fwd'][node['joint']['type']]
            # construct meshes for bbox
            if node['id'] == root_id or node['parent'] == root_id: # no transform
                bb_0, bb_1 = get_bbox_mesh_pair(box_cen, box_size, jtype=jtype_id, jrange=jrange, axis_d=axis_d, axis_o=axis_o)
            else:
                parent_id = node['parent']
                jrange_p = tree[parent_id]['joint']['range']
                jtype_p = tree[parent_id]['joint']['type']
                jtype_p_id = joint_ref['fwd'][jtype_p]
                axis_d_p = np.array(tree[parent_id]['joint']['axis']['direction'])
                axis_o_p = np.array(tree[parent_id]['joint']['axis']['origin'])
                bb_0, bb_1 = get_bbox_mesh_pair(box_cen, box_size, jtype=jtype_p_id, jrange=jrange_p, axis_d=axis_d_p, axis_o=axis_o_p)
            # construct mesh for axis (the axis is not supporting transform for now)
            axis = get_axis_mesh(axis_d, axis_o, box_cen, jtype)
            # append
            bbox_0.append(bb_0)
            bbox_1.append(bb_1)
            axiss.append(axis)
            labels.append(label)
            jtypes.append(jtype_id)

        return {
            'bbox_0': bbox_0,
            'bbox_1': bbox_1,
            'axiss': axiss,
            'labels': labels,
            'jtypes': jtypes
        }

    def save_val_img(self, pred, gt, cond):
        B = pred.shape[0]
        pred_imgs, gt_imgs, gt_graphs = [], [], []
        for b in range(B):
            # convert to humnan readable format json
            pred_json = self.convert_json(pred[b], cond, b)
            gt_json = self.convert_json(gt[b], cond, b)
            # visualize bbox and axis
            pred_meshes = self.prepare_meshes(pred_json)
            bbox_0, bbox_1, axiss = pred_meshes['bbox_0'], pred_meshes['bbox_1'], pred_meshes['axiss']
            pred_img = draw_boxes_axiss_anim(bbox_0, bbox_1, axiss, mode='graph', resolution=128)
            gt_meshes = self.prepare_meshes(gt_json)
            bbox_0, bbox_1, axiss = gt_meshes['bbox_0'], gt_meshes['bbox_1'], gt_meshes['axiss']
            gt_img = draw_boxes_axiss_anim(bbox_0, bbox_1, axiss, mode='graph', resolution=128)
            # visualize graph
            gt_graph = viz_graph(gt_json)
            gt_graph = add_text(cond['name'][b], gt_graph)

            pred_imgs.append(pred_img)
            gt_imgs.append(gt_img)
            gt_graphs.append(gt_graph)
        
        # save images for generated results
        epoch = str(self.current_epoch).zfill(5)
        pred_thumbnails = np.concatenate(pred_imgs, axis=1) # concat batch in width
        self.save_rgb_image(f'{epoch}_pred.png', pred_thumbnails)
        # save images for ground truth
        gt_thumbnails = np.concatenate(gt_imgs, axis=1) # concat batch in width
        gt_graph_imgs = np.concatenate(gt_graphs, axis=1)
        gt_grid = np.concatenate([gt_graph_imgs, gt_thumbnails], axis=0)
        self.save_rgb_image(f'{epoch}_gt.png', gt_grid)

    def save_pred_uncond(self, pred, gt, batch_idx):
        epoch = self.trainer.current_epoch
        _, cond = gt
        B = pred.shape[0]
        offset = batch_idx * B

        for b in range(B):
            # convert to human readable format json
            pred_json = self.convert_json(pred[b], cond, b)
            img_graph = viz_graph(pred_json)
            # visualize bbox and axis
            meshes = self.prepare_meshes(pred_json)
            bbox_0, bbox_1, axiss = meshes['bbox_0'], meshes['bbox_1'], meshes['axiss']
            thumbnail = draw_boxes_axiss_anim(bbox_0, bbox_1, axiss, mode='graph', resolution=128)
            # save
            self.save_json(f'uncond/{epoch}/#{b+offset}.json', pred_json)
            self.save_rgb_image(f'uncond/{epoch}/#{b+offset}_graph.png', img_graph)
            self.save_rgb_image(f'uncond/{epoch}/#{b+offset}_thumbnail.png', thumbnail)
    
    def save_pred_cond_graph(self, pred, gt):
        save_mesh = self.hparams.datamodule.pred_save_mesh
        epoch = self.trainer.current_epoch
        mode = self.hparams.datamodule.pred_mode
        x, c = gt
        cat = c['obj_cat'][0]
        hashcode = c['tree_hash'][0]
        thumbnails = []
        B = pred.shape[0]

        if mode == 'ood':
            gt_json = self.convert_json_graph_only(c, 0)
        else:
            gt_json = self.convert_json(x[0], c, 0)
        gt_graph = viz_graph(gt_json)
        self.save_rgb_image(f'{mode}/{epoch}/{cat}/{hashcode}.png', gt_graph)
        self.save_json(f'{mode}/{epoch}/{cat}/{hashcode}.json', gt_json)

        for b in range(B):
            # convert to human readable format json
            pred_json = self.convert_json(pred[b], c, 0)
            # visualize bbox and axis
            meshes = self.prepare_meshes(pred_json)
            bbox_0, bbox_1, axiss = meshes['bbox_0'], meshes['bbox_1'], meshes['axiss']
            thumbnail = draw_boxes_axiss_anim(bbox_0, bbox_1, axiss, mode='graph', resolution=128)

            self.save_json(f'{mode}/{epoch}/{cat}/{hashcode}/#{b}.json', pred_json)
            self.save_rgb_image(f'{mode}/{epoch}/{cat}/{hashcode}/#{b}_thumbnail.png', thumbnail)
            
            thumbnails.append(thumbnail)
            if save_mesh:
                self.retrieve_meshes(pred_json, f'{mode}/{epoch}/{cat}/{hashcode}/#{b}.ply')

        thumbnails_grid = make_grid(thumbnails, cols=5)
        self.save_rgb_image(f'{mode}/{epoch}/{cat}/{hashcode}_thumbnails.png', thumbnails_grid)


    def save_pred_cond_other(self, raw_pred, masked_pred, gt):
        epoch = self.trainer.current_epoch
        mode = self.hparams.datamodule.pred_mode
        x, c = gt
        cat = c['obj_cat'][0]
        model_name = c['name'][0].split('/')[-1]
        B = raw_pred.shape[0]

        imgs_graph, imgs_semantic, imgs_jtype = [], [], []

        # save ground truth
        gt_json = self.convert_json(x[0], c, 0)
        graph_viz = viz_graph(gt_json)
        self.save_rgb_image(f'{mode}/{epoch}/{cat}/{model_name}/gt_graph.png', graph_viz)
        self.save_json(f'{mode}/{epoch}/{cat}/{model_name}/gt.json', gt_json)

        meshes = self.prepare_meshes(gt_json)
        bbox_0, bbox_1, axiss = meshes['bbox_0'], meshes['bbox_1'], meshes['axiss']

        img_gt = draw_boxes_axiss_anim(bbox_0, bbox_1, axiss, mode='graph', resolution=128)
        self.save_rgb_image(f'{mode}/{epoch}/{cat}/{model_name}/gt.png', img_gt)

        # save predictions
        for b in range(B):
            # raw data
            raw_json = self.convert_json(raw_pred[b], c, 0)
            meshes = self.prepare_meshes(raw_json)
            bbox_0, bbox_1, axiss, labels, jtypes = meshes['bbox_0'], meshes['bbox_1'], meshes['axiss'], meshes['labels'], meshes['jtypes']
            # save
            img_raw = draw_boxes_axiss_anim(bbox_0, bbox_1, axiss, mode='graph', resolution=128)
            self.save_json(f'{mode}/{epoch}/{cat}/{model_name}/#{b}_raw.json', raw_json)
            
            # masked data
            masked_json = self.convert_json(masked_pred[b], c, 0)
            meshes = self.prepare_meshes(masked_json)
            bbox_0, bbox_1, axiss, labels, jtypes = meshes['bbox_0'], meshes['bbox_1'], meshes['axiss'], meshes['labels'], meshes['jtypes']
            # save
            img_graph = draw_boxes_axiss_anim(bbox_0, bbox_1, axiss, mode='graph', resolution=128) # color corresponding to graph nodes
            img_semantic = draw_boxes_axiss_anim(bbox_0, bbox_1, axiss, mode='semantic', resolution=128, types=labels) # color corresponding to semantic labels
            img_jtype = draw_boxes_axiss_anim(bbox_0, bbox_1, axiss, mode='jtype', resolution=128, types=jtypes) # color corresponding to joint types
            # concat the thumbnails
            thumb = np.concatenate([img_raw, img_graph, img_semantic, img_jtype], axis=1)
            self.save_json(f'{mode}/{epoch}/{cat}/{model_name}/#{b}.json', masked_json)
            # self.save_rgb_image(f'{mode}/{epoch}/{cat}/{model_name}/#{b}_thumbnail.png', thumb)

            imgs_graph.append(img_graph)
            imgs_semantic.append(img_semantic)
            imgs_jtype.append(img_jtype)

        img_grid_g = make_grid(imgs_graph, cols=5)
        img_grid_s = make_grid(imgs_semantic, cols=5)
        img_grid_j = make_grid(imgs_jtype, cols=5)
        self.save_rgb_image(f'{mode}/{epoch}/{cat}/{model_name}/thumbnails_graph.png', img_grid_g)
        self.save_rgb_image(f'{mode}/{epoch}/{cat}/{model_name}/thumbnails_semantic.png', img_grid_s)
        self.save_rgb_image(f'{mode}/{epoch}/{cat}/{model_name}/thumbnails_jtype.png', img_grid_j)
    
    def retrieve_meshes(self, obj_spec, save_path):
        DATASET_PATH = self.hparams.datamodule.root
        HASHBOOK_PATH = "./retrieval/retrieval_hash_no_handles.json"
        # Retrieve meshes for the object
        obj_candidates = find_obj_candidates(obj_spec, DATASET_PATH, HASHBOOK_PATH, gt_file_name="train.json")
        retrieved_mesh_specs = pick_and_rescale_parts(obj_spec, obj_candidates, DATASET_PATH, gt_file_name="train.json")

        # ============================ Load the meshes and save them as a GLB file

        scene = trimesh.Scene()
        for i, mesh_spec in enumerate(retrieved_mesh_specs):
            part_spec = obj_spec["diffuse_tree"][i]

            # A part may be composed of multiple meshes
            part_meshes = []
            for file in mesh_spec["files"]:
                mesh = trimesh.load(os.path.join(mesh_spec["dir"], file), force="mesh")
                part_meshes.append(mesh)
            part_mesh = trimesh.util.concatenate(part_meshes)
            
            # Center the mesh at the origin
            part_mesh.vertices -= part_mesh.bounding_box.centroid

            # Rotate the mesh 90 degrees about the z-axis if specified
            if mesh_spec["z_rotate_90"]:
                part_mesh.apply_transform(trimesh.transformations.rotation_matrix(np.radians(90), [0, 0, 1]))

            # Scale and position the mesh
            transform_matrix = trimesh.transformations.scale_and_translate(mesh_spec["scale_factor"], part_spec["aabb"]["center"])
            part_mesh.apply_transform(transform_matrix)

            # Add the mesh to the scene
            scene.add_geometry(part_mesh)

        # Export the scene as a PLY file
        scene.export(save_path)