import os
from PIL import Image
import torch
import torch.nn.functional as F
import glob
import functools
from tqdm import trange
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from archieve import CACHE_ROOT, EVAL_ROOT, DATA_ROOT
from utils.assets import QUERY_DICT, DATASET
from utils.models import load_owl, load_dinov2, load_dreamsim, load_dino

CACHE_ROOT = '/data/sangwon/multi_subject/utils/cache'
EVAL_ROOT = ''
DATA_ROOT = ''

def load_query_image(query_dict):
    query_path = query_dict['query_path']
    query_img = []
    for dir_path in query_path:
        imgs = []
        img_paths = glob.glob(os.path.join(dir_path, "0*.jpg"))
        img_paths.sort()
        for img_path in img_paths:
            img = Image.open(img_path)
            img = img.convert('RGB')
            imgs.append(img)
        query_img.append(imgs)
    query_dict['query_img'] = query_img

    return query_dict

def detect_and_crop(img, query_dict, owl_model, owl_processor, threshold=0.25):
    # img: PIL image
    # if 'owl_query' exist, use it as a text query
    device = owl_model.device
    text_queries = query_dict['owl_query'] if 'owl_query' in query_dict else query_dict['query_name'] # ['chow chow', 'pembroke welsh corgi']
    categories = query_dict['query_name']
    inputs = owl_processor(text=text_queries, images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = owl_model(**inputs)

    target_sizes = torch.Tensor([img.size[::-1]])
    # Convert outputs (bounding boxes and class logits) to Pascal VOC Format (xmin, ymin, xmax, ymax)
    results = owl_processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=threshold)
    boxes, scores, labels = results[0]["boxes"].tolist(), results[0]["scores"].tolist(), results[0]["labels"].tolist()
    
    results = {}
    metadata = []
    bboxes = []
    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        box = [round(i, 2) for i in box]
        bboxes.append(box)
        metadata.append({'box': box, 'score': score, 'label': categories[label]})
    results['metadata'] = metadata
    results['bbox'] = bboxes

    return results # dict['metadata'] = list of dict, dict['bbox'] = list of bbox
          
def crop_and_pad_image(image, bbox):
    """
    Crop an image with a bounding box and apply zero padding if necessary.
    
    Parameters:
    - image: PIL Image object.
    - bbox: Bounding box coordinates as a list or tuple (left, upper, right, lower).
    
    Returns:
    - Cropped and padded PIL Image object.
    """
    # Extract image dimensions and bounding box coordinates
    img_width, img_height = image.size
    left, upper, right, lower = bbox

    cropped_image = image.crop((left, upper, right, lower))
    cropped_width, cropped_height = cropped_image.size

    # Calculate padding
    longer_side = max(cropped_width, cropped_height)    
    final_size = (longer_side, longer_side)

    # Create a new image and paste the cropped one
    padded_image = Image.new("RGB", final_size, (255, 255, 255))
    x_offset = (longer_side - cropped_width) // 2
    y_offset = (longer_side - cropped_height) // 2
    padded_image.paste(cropped_image, (x_offset, y_offset))
    
    return padded_image


class eval_with_dreamsim:
    def __init__(self, cache_dir=CACHE_ROOT, device='cuda'):
        self.owl_processor, self.owl_model = load_owl(device)
        self.ds_model, self.ds_preprocess = load_dreamsim(pretrained=True, cache_dir=cache_dir)
        self.device = device

    def cache_query_embedding(self, query_dict):
        query_img = query_dict['query_img'] # already sorted
        query_emb = []
        for imgs in query_img:
            embs = []
            for img in imgs:
                emb = self.ds_model.embed(self.ds_preprocess(img).to(self.device)) # 1, 1792
                embs.append(emb.to('cpu'))
            query_emb.append(torch.cat(embs, dim=0)) # 5, 1792
        query_dict['query_emb'] = query_emb
        return query_dict
    
    def query_dict_update(self, query_dict):
        query_dict = load_query_image(query_dict)
        return self.cache_query_embedding(query_dict)
    
    def compute_distance(self, candidate, query_dict):
        # candidate: cropped single image
        # return:
        results = {}
        candidate_emb = self.ds_model.embed(self.ds_preprocess(candidate).to(self.device)) # 1, 1792
        for i, query_emb in enumerate(query_dict['query_emb']):
            # query_emb: 5, 1792
            distance = 1 - F.cosine_similarity(candidate_emb, query_emb.to(self.device), dim=-1) # 5
            distance = distance.tolist() # list of 5 distances
            results[query_dict['query_name'][i]] = distance
        return results # dict[query_name] = list of 5 distances
    
    def owl_dreamsim_distance(self, img, query_dict, threshold=0.25):
        if img.mode != 'RGB':
            img = img.convert('RGB')
        categories = query_dict['query_name'] # ['chow chow', 'pembroke welsh corgi']
        owl_out = detect_and_crop(img, query_dict, self.owl_model, self.owl_processor, threshold)
        bboxes = owl_out['bbox']
        results = []
        dreamsim_out = {}
        for bbox in bboxes:
            result = {}
            x1, y1, x2, y2 = bbox
            # candidate = img.crop((x1, y1, x2, y2))
            candidate = crop_and_pad_image(img, (x1, y1, x2, y2))
            result['image'] = candidate
            result['bbox'] = bbox
            dreamsim_out = self.compute_distance(candidate, query_dict)
            distances = []
            for i, category in enumerate(categories):
                distance = dreamsim_out[category] # list
                distances.append(distance)
            result['distance'] = distances
            results.append(result)
        return results, owl_out, dreamsim_out
    
    def score(self, image, query_dict, threshold=0.4, return_round=False):
        if 'query_emb' not in query_dict:
            query_dict = self.query_dict_update(query_dict)
        scores = []
        scores_ = []
        results, owl_out, dreamsim_out = self.owl_dreamsim_distance(image, query_dict, threshold)
        for result in results:
            distance = result['distance']
            distance_score = [[1 - x for x in y] for y in distance]
            distance_score_ = [[round(1 - x, 2) for x in y] for y in distance]
            scores.append(distance_score)
            scores_.append(distance_score_)
        return scores_ if return_round else scores


class eval_with_dinov2:
    def __init__(self, device):
        super().__init__()
        self.owl_processor, self.owl_model = load_owl(device)
        self.dino_preprocess, self.dino_model = load_dinov2(device)
        self.device = device

    # add query_emb to query_dict (PIL image to embedding)
    def cache_query_embedding(self, query_dict):
        query_img = query_dict['query_img'] # already sorted
        query_emb = []
        for imgs in query_img:
            embs = []
            for img in imgs:
                dino_input = self.dino_preprocess(images=img, return_tensors='pt').to(self.device)
                emb = self.dino_model(**dino_input).last_hidden_state.mean(dim=1) # 1, 768
                embs.append(emb.to('cpu'))
            query_emb.append(torch.cat(embs, dim=0)) # 5, 768
        query_dict['query_emb'] = query_emb
        return query_dict

    def query_dict_update(self, query_dict):
        query_dict = load_query_image(query_dict)
        return self.cache_query_embedding(query_dict)

    def compute_distance(self, candidate, query_dict):
        # candidate: cropped single image
        # return: 
        results = {}
        dino_input = self.dino_preprocess(images=candidate, return_tensors='pt').to(self.device)
        candidate_emb = self.dino_model(**dino_input).last_hidden_state.mean(dim=1) # 1, 768
        for i, query_emb in enumerate(query_dict['query_emb']):
            # query_emb: 1, 768
            distance = 1 - F.cosine_similarity(candidate_emb, query_emb.to(self.device), dim=-1) # 5
            distance = distance.tolist() # list of 5 distances
            results[query_dict['query_name'][i]] = distance
        return results # dict[query_name] = list of 5 distances
    
    def owl_dinov2_distance(self, img, query_dict, threshold=0.25):
        if img.mode != 'RGB':
            img = img.convert('RGB')
        categories = query_dict['query_name'] # ['chow chow', 'pembroke welsh corgi']
        owl_out = detect_and_crop(img, query_dict, self.owl_model, self.owl_processor, threshold)
        bboxes = owl_out['bbox']
        results = []
        dinov2_out = {}
        for bbox in bboxes:
            result = {}
            x1, y1, x2, y2 = bbox
            candidate = crop_and_pad_image(img, (x1, y1, x2, y2))
            result['image'] = candidate
            result['bbox'] = bbox
            dinov2_out = self.compute_distance(candidate, query_dict)
            distances = []
            for i, category in enumerate(categories):
                distance = dinov2_out[category] # list
                distances.append(distance)
            result['distance'] = distances
            results.append(result)
        return results, owl_out, dinov2_out
    
    def score(self, image, query_dict, threshold=0.4, return_round=False):
        if 'query_emb' not in query_dict:
            query_dict = self.query_dict_update(query_dict)
        scores = []
        scores_ = []
        results, owl_out, dinov2_out = self.owl_dinov2_distance(image, query_dict, threshold)
        for result in results:
            distance = result['distance']
            distance_score = [[1 - x for x in y] for y in distance]
            distance_score_ = [[round(1 - x, 2) for x in y] for y in distance]
            scores.append(distance_score)
            scores_.append(distance_score_)
        return scores_ if return_round else scores


class eval_with_dino:
    def __init__(self, device):
        super().__init__()
        self.owl_processor, self.owl_model = load_owl(device)
        self.dino_preprocess, self.dino_model = load_dino(device)
        self.device = device

    # add query_emb to query_dict (PIL image to embedding)
    def cache_query_embedding(self, query_dict):
        query_img = query_dict['query_img'] # already sorted
        query_emb = []
        for imgs in query_img:
            embs = []
            for img in imgs:
                dino_input = self.dino_preprocess(images=img, return_tensors='pt').to(self.device)
                emb = self.dino_model(**dino_input).last_hidden_state.mean(dim=1) # 1, 768
                embs.append(emb.to('cpu'))
            query_emb.append(torch.cat(embs, dim=0)) # 5, 768
        query_dict['query_emb'] = query_emb
        return query_dict

    def query_dict_update(self, query_dict):
        query_dict = load_query_image(query_dict)
        return self.cache_query_embedding(query_dict)

    def compute_distance(self, candidate, query_dict):
        # candidate: cropped single image
        # return: 
        results = {}
        dino_input = self.dino_preprocess(images=candidate, return_tensors='pt').to(self.device)
        candidate_emb = self.dino_model(**dino_input).last_hidden_state.mean(dim=1) # 1, 768
        for i, query_emb in enumerate(query_dict['query_emb']):
            # query_emb: 1, 768
            distance = 1 - F.cosine_similarity(candidate_emb, query_emb.to(self.device), dim=-1) # 5
            distance = distance.tolist() # list of 5 distances
            results[query_dict['query_name'][i]] = distance
        return results # dict[query_name] = list of 5 distances
    
    def owl_dino_distance(self, img, query_dict, threshold=0.25):
        if img.mode != 'RGB':
            img = img.convert('RGB')
        categories = query_dict['query_name'] # ['chow chow', 'pembroke welsh corgi']
        owl_out = detect_and_crop(img, query_dict, self.owl_model, self.owl_processor, threshold)
        bboxes = owl_out['bbox']
        results = []
        dino_out = {}
        for bbox in bboxes:
            result = {}
            x1, y1, x2, y2 = bbox
            candidate = crop_and_pad_image(img, (x1, y1, x2, y2))
            result['image'] = candidate
            result['bbox'] = bbox
            dino_out = self.compute_distance(candidate, query_dict)
            distances = []
            for i, category in enumerate(categories):
                distance = dino_out[category] # list
                distances.append(distance)
            result['distance'] = distances
            results.append(result)
        return results, owl_out, dino_out
    
    def score(self, image, query_dict, threshold=0.4, return_round=False):
        if 'query_emb' not in query_dict:
            query_dict = self.query_dict_update(query_dict)
        scores = []
        scores_ = []
        results, owl_out, dino_out = self.owl_dino_distance(image, query_dict, threshold)
        for result in results:
            distance = result['distance']
            distance_score = [[1 - x for x in y] for y in distance]
            distance_score_ = [[round(1 - x, 2) for x in y] for y in distance]
            scores.append(distance_score)
            scores_.append(distance_score_)
        return scores_ if return_round else scores


class IdentityMetric:
    def __init__(self, method, conf_type="max-max", device="cuda:0"):
        super().__init__()
        eval_ = {
            'dino': eval_with_dino,
            'dinov2': eval_with_dinov2,
            'dreamsim': functools.partial(eval_with_dreamsim, cache_dir=CACHE_ROOT),
        }
        self.eval_with = eval_.get(method, 'Wrong method name')(device=device)
        self.conf_type = conf_type
        self.device = device
    
    def get_sim(self, scores, agg_type):
        if agg_type=='max':
            agg_op = lambda x: max(x)
        elif agg_type=='mean':
            agg_op = lambda x: sum(x)/len(x)
        else:
            raise NotImplementedError(f'{agg_type} not implemented.')
        if len(scores) != 2:
            return [[0.,0.], [0.,0]]
        sim = [
            [agg_op(scores[0][0]), agg_op(scores[0][1])], 
            [agg_op(scores[1][0]), agg_op(scores[1][1])]
        ]
        return sim

    def linear_score(self, scores, agg_type):
        sim = self.get_sim(scores, agg_type)
        conf_1 = sim[0][0] - sim[0][1]
        conf_2 = sim[1][0] - sim[1][1]

        if conf_1 * conf_2 > 0:
            return abs(conf_1 - conf_2)
        else:
            return abs(conf_1) + abs(conf_2)
        
    def max_score(self, scores, agg_type):
        sim = self.get_sim(scores, agg_type)
        return np.array(sim).max(-1).sum().tolist()
    
    def get_score(self, img_path, concepts=["dog6", "dog2"], thr=0.4):
        query_dict = {
            "query_name": [QUERY_DICT[concepts[0]], QUERY_DICT[concepts[1]]],
            "query_path": [os.path.join(DATA_ROOT, concepts[0]), os.path.join(DATA_ROOT, concepts[1])]
        }
        image = Image.open(img_path)
        scores = self.eval_with.score(image, query_dict, return_round=False, threshold=thr)
        return scores, self.linear_score(scores, 'max'), self.max_score(scores, 'max'), self.linear_score(scores, 'mean'), self.max_score(scores, 'mean')