#@title loading utils
import torch
from omegaconf import OmegaConf
import PIL
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import random
import json
import os
import h5py
import tempfile
from einops import rearrange
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
import argparse
import pickle

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def get_model():
    config = OmegaConf.load("./config_vg.yaml")
    # model = load_model_from_config(config, "./pretrained_model.ckpt")
    model = load_model_from_config(config, "./pretrained/last.ckpt")
    return model

def build_loaders():
    dset_kwargs = {
        'vocab_path': './datasets/vg/vocab.json',
        'h5_path': './datasets/vg/test.h5',
        'image_dir': './datasets/vg/images',
        'image_size': (256, 256),
        'max_samples': None,
        'max_objects': 30,
        'use_orphaned_objects': True,
        'include_relationships': True,
    }
    dset = VgSceneGraphDataset(**dset_kwargs)
    collate_fn = vg_collate_fn

    loader = DataLoader(dset, batch_size=1, num_workers=4, shuffle=False, collate_fn=collate_fn)
    return loader

def build_loaders_customized(data_vocab_path, data_path, image_dir):
    dset_kwargs = {
        'use_vocab_path': './datasets/vg/vocab.json',
        'data_vocab_path': data_vocab_path,
        'data_path': data_path,
        'image_dir': image_dir,
        'image_size': (256, 256),
        'max_samples': None,
        'max_objects': 30,
        'use_orphaned_objects': True,
        'include_relationships': True,
    }
    dset = VgSceneGraphDatasetCustomized(**dset_kwargs)
    collate_fn = vg_collate_fn_customized

    loader = DataLoader(dset, batch_size=1, num_workers=4, shuffle=False, collate_fn=collate_fn)
    return loader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_set", type=str, choices=["vg4998", "triplet_swap_val", ""], default="", help="The evaluation set")
    args = parser.parse_args()

    model = get_model()
    sampler = DDIMSampler(model)

    ddim_steps = 200
    ddim_eta = 1.0

    vocab_file = './datasets/vg/vocab.json'
    with open(vocab_file, 'r') as f:
        vocab = json.load(f)

    if args.eval_set == "":
        loader = build_loaders()
    else:
        if args.eval_set == "vg4998":
            data_vocab_path = "/scratch/bichengx/VG-SGG/V2-Last/idx_to_word.pkl"
            data_path = "/scratch/bichengx/VG-SGG/V2-Last/validation_common_data_bbox_dbox32_np.pkl"
            image_dir = "/scratch/bichengx/VG-SGG/VG_100K_3"
        elif args.eval_set == "triplet_swap_val":
            data_vocab_path = "/scratch/bichengx/VG-SGG/V2-Last/idx_to_word.pkl"
            data_path = "/scratch/bichengx/VG-SGG/VG-relation-test/triplet_swaped_dict_validation_1_input.pkl"
            image_dir = "/scratch/bichengx/VG-SGG/VG-relation-test/triplet_swaped_dict_validation_1_images"
        else:
            raise ValueError(f"Unknown eval set: {args.eval_set}")
        loader = build_loaders_customized(data_vocab_path, data_path, image_dir)

    root_dir = './test_results_whole_test' if args.eval_set == "" else f'./test_results_{args.eval_set}'
    scene_graph_dir = os.path.join(root_dir, 'scene_graph')
    generate_img_dir = os.path.join(root_dir, 'img')
    gt_img_dir = os.path.join(root_dir, 'gt_img')

    if not os.path.exists(root_dir):
        os.mkdir(root_dir)
    if not os.path.exists(scene_graph_dir):
        os.mkdir(scene_graph_dir)
    if not os.path.exists(generate_img_dir):
        os.mkdir(generate_img_dir)
    if not os.path.exists(gt_img_dir):
        os.mkdir(gt_img_dir)

    num_sample_times = 5
    n_samples_per_scene_graph = 1

    with torch.no_grad():
        with model.ema_scope():
            img_idx = -1
            for batch_data in tqdm(loader):
                img_idx += 1
                # if img_idx < 2500:
                #     continue
                if args.eval_set == "":
                    imgs, objs, boxes, triples, obj_to_img, triple_to_img = [x.cuda() for x in batch_data]
                    file_name_id = img_idx
                else:
                    imgs, objs, boxes, triples, file_name_id, obj_to_img, triple_to_img = [x.cuda() for x in batch_data]
                    file_name_id = file_name_id.item()

                scene_graph_path = os.path.join(scene_graph_dir, str(file_name_id)+'_graph.png')

                draw_scene_graph(objs=objs, triples=triples, vocab=vocab, output_filename=scene_graph_path)                
                graph_info = [imgs, objs, None, triples, obj_to_img, triple_to_img]
                cond = model.get_learned_conditioning(graph_info)

                for num_sample in range(num_sample_times):
                    samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                    conditioning=cond,
                                                    batch_size=n_samples_per_scene_graph,
                                                    shape=[4, 32, 32],
                                                    verbose=False,
                                                    eta=ddim_eta)

                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                    x_samples_ddim = x_samples_ddim.squeeze(0)
                    x_samples_ddim = 255. * rearrange(x_samples_ddim, 'c h w -> h w c').cpu().numpy()
                    results = Image.fromarray(x_samples_ddim.astype(np.uint8))
                    results.save(os.path.join(generate_img_dir, str(file_name_id)+'_'+str(num_sample)+'_img.png'))
                gt_img = to_pil_image(imgs[0], mode="RGB")
                gt_img.save(os.path.join(gt_img_dir, str(file_name_id)+'_gt.png'))
                # if img_idx > 3000:
                #     break
    return None

def draw_scene_graph(objs, triples, vocab=None, **kwargs):
    output_filename = kwargs.pop('output_filename', 'graph.png')
    orientation = kwargs.pop('orientation', 'V')
    edge_width = kwargs.pop('edge_width', 6)
    arrow_size = kwargs.pop('arrow_size', 1.5)
    binary_edge_weight = kwargs.pop('binary_edge_weight', 1.2)
    ignore_dummies = kwargs.pop('ignore_dummies', True)

    if orientation not in ['V', 'H']:
        raise ValueError('Invalid orientation "%s"' % orientation)
    rankdir = {'H': 'LR', 'V': 'TD'}[orientation]

    if vocab is not None:
        assert torch.is_tensor(objs)
        assert torch.is_tensor(triples)
        objs_list, triples_list = [], []
        for i in range(objs.size(0)):
            objs_list.append(vocab['object_idx_to_name'][objs[i].item()])
        for i in range(triples.size(0)):
            s = triples[i, 0].item()
            # p = vocab['pred_name_to_idx'][triples[i, 1].item()]
            p = triples[i, 1].item()
            o = triples[i, 2].item()
            triples_list.append([s, p, o])
        objs, triples = objs_list, triples_list

    lines = [
        'digraph{',
        'graph [size="5,3",ratio="compress",dpi="300",bgcolor="transparent"]',
        'rankdir=%s' % rankdir,
        'nodesep="0.5"',
        'ranksep="0.5"',
        'node [shape="box",style="rounded,filled",fontsize="48",color="none"]',
        'node [fillcolor="lightpink1"]',
    ]

    for i, obj in enumerate(objs):
        if ignore_dummies and obj == '__image__':
            continue
        lines.append('%d [label="%s"]' % (i, obj))

    next_node_id = len(objs)
    lines.append('node [fillcolor="lightblue1"]')
    for s, p, o in triples:
        p = vocab['pred_idx_to_name'][p]
        if ignore_dummies and p == '__in_image__':
            continue
        lines += [
            '%d [label="%s"]' % (next_node_id, p),
            '%d->%d [penwidth=%f,arrowsize=%f,weight=%f]' % (
                s, next_node_id, edge_width, arrow_size, binary_edge_weight),
            '%d->%d [penwidth=%f,arrowsize=%f,weight=%f]' % (
                next_node_id, o, edge_width, arrow_size, binary_edge_weight)
        ]
        next_node_id += 1
    lines.append('}')

    ff, dot_filename = tempfile.mkstemp()
    with open(dot_filename, 'w') as f:
        for line in lines:
            f.write('%s\n' % line)
    os.close(ff)

    output_format = os.path.splitext(output_filename)[1][1:]
    os.system('dot -T%s %s > %s' % (output_format, dot_filename, output_filename))
    return None

def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=True)
    model.cuda()
    model.eval()
    return model

class VgSceneGraphDataset(Dataset):
    def __init__(self, vocab_path, h5_path, image_dir, image_size=(256, 256), max_objects=10,
                 max_samples=None,
                 include_relationships=True, use_orphaned_objects=True):
        super(VgSceneGraphDataset, self).__init__()
        with open(vocab_path, 'r') as f:
            vocab = json.load(f)
        self.image_dir = image_dir
        self.image_size = image_size
        self.vocab = vocab
        self.num_objects = len(vocab['object_idx_to_name'])
        self.use_orphaned_objects = use_orphaned_objects
        self.max_objects = max_objects
        self.max_samples = max_samples
        self.include_relationships = include_relationships

        transform = [Resize(image_size), transforms.ToTensor()]  # augmentation
        self.transform = transforms.Compose(transform)

        self.data = {}

        with h5py.File(h5_path, 'r') as f:
            for k, v in f.items():
                if k == 'image_paths':
                    self.image_paths = list(v)
                else:
                    self.data[k] = torch.IntTensor(np.asarray(v))

    def __len__(self):
        num = self.data['object_names'].size(0)
        if self.max_samples is not None:
            return min(self.max_samples, num)
        return num

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, str(self.image_paths[index], encoding="utf-8"))

        with open(img_path, 'rb') as f:
            with PIL.Image.open(f) as image:
                WW, HH = image.size
                image = self.transform(image.convert('RGB'))

        H, W = self.image_size

        obj_idxs_with_rels = set()
        obj_idxs_without_rels = set(range(self.data['objects_per_image'][index].item()))
        for r_idx in range(self.data['relationships_per_image'][index]):
            s = self.data['relationship_subjects'][index, r_idx].item()
            o = self.data['relationship_objects'][index, r_idx].item()
            obj_idxs_with_rels.add(s)
            obj_idxs_with_rels.add(o)
            obj_idxs_without_rels.discard(s)
            obj_idxs_without_rels.discard(o)

        obj_idxs = list(obj_idxs_with_rels)
        obj_idxs_without_rels = list(obj_idxs_without_rels)
        if len(obj_idxs) > self.max_objects - 1:
            obj_idxs = random.sample(obj_idxs, self.max_objects)
        if len(obj_idxs) < self.max_objects - 1 and self.use_orphaned_objects:
            num_to_add = self.max_objects - 1 - len(obj_idxs)
            num_to_add = min(num_to_add, len(obj_idxs_without_rels))
            obj_idxs += random.sample(obj_idxs_without_rels, num_to_add)
        O = len(obj_idxs) + 1

        objs = torch.LongTensor(O).fill_(-1)

        boxes = torch.FloatTensor([[0, 0, 1, 1]]).repeat(O, 1)
        obj_idx_mapping = {}
        for i, obj_idx in enumerate(obj_idxs):
            objs[i] = self.data['object_names'][index, obj_idx].item()
            x, y, w, h = self.data['object_boxes'][index, obj_idx].tolist()
            x0 = float(x) / WW
            y0 = float(y) / HH
            x1 = float(x + w) / WW
            y1 = float(y + h) / HH
            boxes[i] = torch.FloatTensor([x0, y0, x1, y1])
            obj_idx_mapping[obj_idx] = i

        objs[O - 1] = self.vocab['object_name_to_idx']['__image__']

        triples = []
        for r_idx in range(self.data['relationships_per_image'][index].item()):
            if not self.include_relationships:
                break

            s = self.data['relationship_subjects'][index, r_idx].item()
            p = self.data['relationship_predicates'][index, r_idx].item()
            o = self.data['relationship_objects'][index, r_idx].item()
            s = obj_idx_mapping.get(s, None)
            o = obj_idx_mapping.get(o, None)
            if s is not None and o is not None:
                triples.append([s, p, o])

        # Add dummy __in_image__ relationships for all objects
        in_image = self.vocab['pred_name_to_idx']['__in_image__']
        for i in range(O - 1):
            triples.append([i, in_image, O - 1])

        triples = torch.LongTensor(triples)
        return image, objs, boxes, triples

class Resize(object):
    def __init__(self, size, interp=PIL.Image.BILINEAR):
        if isinstance(size, tuple):
            H, W = size
            self.size = (W, H)
        else:
            self.size = (size, size)
        self.interp = interp

    def __call__(self, img):
        return img.resize(self.size, self.interp)

def vg_collate_fn(batch):
    all_imgs, all_objs, all_boxes, all_triples = [], [], [], []
    all_obj_to_img, all_triple_to_img = [], []
    obj_offset = 0
    for i, (img, objs, boxes, triples) in enumerate(batch):
        all_imgs.append(img[None])
        O, T = objs.size(0), triples.size(0)
        all_objs.append(objs)
        all_boxes.append(boxes)
        triples = triples.clone()
        triples[:, 0] += obj_offset
        triples[:, 2] += obj_offset
        all_triples.append(triples)

        all_obj_to_img.append(torch.LongTensor(O).fill_(i))
        all_triple_to_img.append(torch.LongTensor(T).fill_(i))
        obj_offset += O

    all_imgs = torch.cat(all_imgs)
    all_objs = torch.cat(all_objs)
    all_boxes = torch.cat(all_boxes)
    all_triples = torch.cat(all_triples)
    all_obj_to_img = torch.cat(all_obj_to_img)
    all_triple_to_img = torch.cat(all_triple_to_img)

    out = (all_imgs, all_objs, all_boxes, all_triples, all_obj_to_img, all_triple_to_img)
    return out

class VgSceneGraphDatasetCustomized(Dataset):
    def __init__(self, use_vocab_path, data_vocab_path, data_path, image_dir, image_size=(256, 256),
                 max_objects=10, max_samples=None, include_relationships=True, use_orphaned_objects=True):
        super(VgSceneGraphDatasetCustomized, self).__init__()
        with open(use_vocab_path, 'r') as f:
            self.vocab = json.load(f)
        with open(data_vocab_path, 'rb') as f:
            data_vocab = pickle.load(f)
        self.data_ind_to_classes = data_vocab['ind_to_classes']
        self.data_ind_to_predicates = data_vocab['ind_to_predicates']
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        self.image_dir = image_dir
        self.image_size = image_size
        transform = [Resize(image_size), transforms.ToTensor()]  # augmentation
        self.transform = transforms.Compose(transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        file_name = item['file_name']
        file_name_id = int(file_name.split('.')[0])
        img_path = os.path.join(self.image_dir, file_name)

        # image
        with open(img_path, 'rb') as f:
            with PIL.Image.open(f) as image:
                image = self.transform(image.convert('RGB'))

        # object labels
        num_nodes = item['node_labels'].shape[0]
        O = num_nodes + 1
        objs = torch.LongTensor(O).fill_(-1)
        for node_idx in range(num_nodes):
            node_name = self.data_ind_to_classes[item['node_labels'][node_idx]]
            assert (node_name in self.vocab['object_name_to_idx'])
            objs[node_idx] = self.vocab['object_name_to_idx'][node_name]
        objs[O - 1] = self.vocab['object_name_to_idx']['__image__']

        # object boxes
        node_bboxes_xcyc = item['node_bboxes_xcyc']
        node_bboxes_xyxy = np.zeros_like(node_bboxes_xcyc)
        node_bboxes_xyxy[:, 0] = (node_bboxes_xcyc[:, 0] - node_bboxes_xcyc[:, 2]/2).clip(0, 1)
        node_bboxes_xyxy[:, 1] = (node_bboxes_xcyc[:, 1] - node_bboxes_xcyc[:, 3]/2).clip(0, 1)
        node_bboxes_xyxy[:, 2] = (node_bboxes_xcyc[:, 0] + node_bboxes_xcyc[:, 2]/2).clip(0, 1)
        node_bboxes_xyxy[:, 3] = (node_bboxes_xcyc[:, 1] + node_bboxes_xcyc[:, 3]/2).clip(0, 1)
        assert (np.all(node_bboxes_xyxy[:, 2] >= node_bboxes_xyxy[:, 0]))
        assert (np.all(node_bboxes_xyxy[:, 3] >= node_bboxes_xyxy[:, 1]))
        boxes = torch.FloatTensor([[0, 0, 1, 1]]).repeat(O, 1)
        boxes[:num_nodes] = torch.FloatTensor(node_bboxes_xyxy)

        # triplets
        edge_map = item['edge_map']
        subj_node_idx, obj_node_idx = np.where(edge_map)
        subj_node_idx_list = list(subj_node_idx)
        obj_node_idx_list = list(obj_node_idx)
        triples = []
        for subj_idx, obj_idx in zip(subj_node_idx_list, obj_node_idx_list):
            edge_name = self.data_ind_to_predicates[edge_map[subj_idx, obj_idx]]
            if edge_name in self.vocab['pred_name_to_idx']:
                triples.append([subj_idx, self.vocab['pred_name_to_idx'][edge_name], obj_idx])

        # Add dummy __in_image__ relationships for all objects
        in_image = self.vocab['pred_name_to_idx']['__in_image__']
        for i in range(O - 1):
            triples.append([i, in_image, O - 1])

        triples = torch.LongTensor(triples)
        return image, objs, boxes, triples, file_name_id

def vg_collate_fn_customized(batch):
    all_imgs, all_objs, all_boxes, all_triples = [], [], [], []
    all_file_name_id = []
    all_obj_to_img, all_triple_to_img = [], []
    obj_offset = 0
    for i, (img, objs, boxes, triples, file_name_id) in enumerate(batch):
        all_imgs.append(img[None])
        O, T = objs.size(0), triples.size(0)
        all_objs.append(objs)
        all_boxes.append(boxes)
        triples = triples.clone()
        triples[:, 0] += obj_offset
        triples[:, 2] += obj_offset
        all_triples.append(triples)
        all_file_name_id.append(torch.tensor([file_name_id]))

        all_obj_to_img.append(torch.LongTensor(O).fill_(i))
        all_triple_to_img.append(torch.LongTensor(T).fill_(i))
        obj_offset += O

    all_imgs = torch.cat(all_imgs)
    all_objs = torch.cat(all_objs)
    all_boxes = torch.cat(all_boxes)
    all_triples = torch.cat(all_triples)
    all_file_name_id = torch.cat(all_file_name_id)
    all_obj_to_img = torch.cat(all_obj_to_img)
    all_triple_to_img = torch.cat(all_triple_to_img)

    out = (all_imgs, all_objs, all_boxes, all_triples, all_file_name_id, all_obj_to_img, all_triple_to_img)
    return out

if __name__ == '__main__':
    main()