import os
import re
import time
import json
from PIL import Image
import torchvision.transforms as transforms
from argparse import Namespace
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed
import gc

from src_files.models import create_model
from src_files.models.caformer.metaformer_baselines import use_xformers
from src_files.helper_functions.helper_functions import crop_fix
from src_files.models.tresnet.tresnet import InplacABN_to_ABN

from tqdm.auto import tqdm

MODELS_NAME = [
    'caformer_m36',
    'tresnet_d',
]

MODELS_NAME_WITH_FILE_PATH = {
    MODELS_NAME[0]: "ckpt/ml_caformer_m36_fp16_dec-5-97527.ckpt"
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_hf_file_path(file_path, repo_id='7eu7d7/ML-Danbooru'):
    from huggingface_hub import hf_hub_download
    if os.path.exists(file_path) and os.path.isfile(file_path):  # if file_path is a file, return it.
        return file_path
    else:
        path = hf_hub_download(repo_id=repo_id, filename=file_path)
        return path

class Infer:
    MODELS_PATH = [
        'ml_caformer_m36_fp16_dec-5-97527.ckpt',
        'TResnet-D-FLq_ema_6-30000.ckpt',
    ]
    DEFAULT_MODEL_PATH = MODELS_PATH[0]
    MODELS_NAME = [
        'caformer_m36',
        'tresnet_d',
    ]
    num_classes = 12547
    RE_SPECIAL = re.compile(r'([\\()])')
    IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp", ".bmp"]

    def __init__(self):
        self.ca_former_args = Namespace()
        self.ca_former_args.decoder_embedding = 384
        self.ca_former_args.num_layers_decoder = 4
        self.ca_former_args.num_head_decoder = 8
        self.ca_former_args.num_queries = 80
        self.ca_former_args.scale_skip = 1

        self.tresnet_args = Namespace()
        self.tresnet_args.decoder_embedding = 1024
        self.tresnet_args.num_of_groups = 32

        self.args_list = [self.ca_former_args, self.tresnet_args]

        self.last_model_name = None

        self.load_class_map()

        self.load_model()

    def load_model(self, model_name=MODELS_NAME[0]):
        # ckpt_file = hf_hub_download(repo_id='7eu7d7/ML-Danbooru', filename=path)

        path = MODELS_NAME_WITH_FILE_PATH[model_name]
        print("load model ", model_name)
        ckpt_file = get_hf_file_path(path)
        model_idx = 0
        for i in range(len(self.MODELS_PATH)):
            key = self.MODELS_PATH[i]
            if path.find(key) >= 0:
                model_idx = i
                break
        # model_idx = self.MODELS_PATH.index(path)
        self.model = create_model(self.MODELS_NAME[model_idx], self.num_classes, self.args_list[model_idx]).to(device)
        state = torch.load(ckpt_file, map_location='cpu')
        # use_xformers  ml_decoder.use_xformers
        if model_idx == 1 and not use_xformers and 'head.decoder.layers.0.multihead_attn.in_proj_container.q_proj.weight' in state:
            in_proj_weight = torch.cat([state['head.decoder.layers.0.multihead_attn.in_proj_container.q_proj.weight'],
                                        state['head.decoder.layers.0.multihead_attn.in_proj_container.k_proj.weight'],
                                        state[
                                            'head.decoder.layers.0.multihead_attn.in_proj_container.v_proj.weight'], ],
                                       dim=0)
            in_proj_bias = torch.cat([state['head.decoder.layers.0.multihead_attn.in_proj_container.q_proj.bias'],
                                      state['head.decoder.layers.0.multihead_attn.in_proj_container.k_proj.bias'],
                                      state['head.decoder.layers.0.multihead_attn.in_proj_container.v_proj.bias'], ],
                                     dim=0)
            state['head.decoder.layers.0.multihead_attn.out_proj.weight'] = state[
                'head.decoder.layers.0.multihead_attn.proj.weight']
            state['head.decoder.layers.0.multihead_attn.out_proj.bias'] = state[
                'head.decoder.layers.0.multihead_attn.proj.bias']
            state['head.decoder.layers.0.multihead_attn.in_proj_weight'] = in_proj_weight
            state['head.decoder.layers.0.multihead_attn.in_proj_bias'] = in_proj_bias

            del state['head.decoder.layers.0.multihead_attn.in_proj_container.q_proj.weight']
            del state['head.decoder.layers.0.multihead_attn.in_proj_container.k_proj.weight']
            del state['head.decoder.layers.0.multihead_attn.in_proj_container.v_proj.weight']
            del state['head.decoder.layers.0.multihead_attn.in_proj_container.q_proj.bias']
            del state['head.decoder.layers.0.multihead_attn.in_proj_container.k_proj.bias']
            del state['head.decoder.layers.0.multihead_attn.in_proj_container.v_proj.bias']
            del state['head.decoder.layers.0.multihead_attn.proj.weight']
            del state['head.decoder.layers.0.multihead_attn.proj.bias']

        self.model.load_state_dict(state, strict=True)

    def load_class_map(self):
        # classes_file = hf_hub_download(repo_id='7eu7d7/ML-Danbooru', filename='class.json')
        classes_file = get_hf_file_path(file_path='class.json')
        with open(classes_file, 'r') as f:
            self.class_map = json.load(f)

    def build_transform(self, image_size, keep_ratio=False):
        if keep_ratio:
            trans = transforms.Compose([
                transforms.Resize(image_size),
                crop_fix,
                transforms.ToTensor(),
            ])
        else:
            trans = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ])
        return trans

    def infer_(self, img: Image.Image, thr: float):
        img = self.trans(img.convert('RGB')).to(device)
        with torch.cuda.amp.autocast():
            img = img.unsqueeze(0)
            output = torch.sigmoid(self.model(img)).cpu().view(-1)
        pred = torch.where(output > thr)[0].numpy()

        pre_result_class_list = [(self.class_map[str(i)], output[i]) for i in pred]
        return pre_result_class_list

    @torch.no_grad()
    def infer_one(self, img: Image.Image, threshold: float, image_size: int, keep_ratio: bool, model_name: str,
                  space: bool, escape: bool, conf: bool):
        if self.last_model_name != model_name:
            self.load_model(model_name)
            self.last_model_name = model_name

        self.trans = self.build_transform(image_size, keep_ratio)
        pre_result_class_list = self.infer_(img, threshold)
        pre_result_class_list.sort(reverse=True, key=lambda x: x[1])
        if space:
            pre_result_class_list = [(cls.replace('_', ' '), score) for cls, score in pre_result_class_list]
        if escape:
            pre_result_class_list = [(re.sub(self.RE_SPECIAL, r'\\\1', cls), score) for cls, score in
                                     pre_result_class_list]
        pre_score_result = {
            cls: float(score) for cls, score in pre_result_class_list}
        text_result = ', '.join([f'{cls}:{score:.2f}' if conf else cls for cls, score in pre_result_class_list])
        return text_result, json.dumps(pre_score_result)

    @torch.no_grad()
    def infer_folder(self, id_task, path: str, threshold: float, image_size: int, keep_ratio: bool, model_name: str,
                     space: bool, escape: bool,
                     out_type: str):
        if self.last_model_name != model_name:
            self.load_model(model_name)
            self.last_model_name = model_name

        self.trans = self.build_transform(image_size, keep_ratio)

        tag_dict = {}
        img_list = [os.path.join(path, x) for x in os.listdir(path) if
                    x[x.rfind('.'):].lower() in self.IMAGE_EXTENSIONS]
        job_count = len(img_list)
        for i, item in enumerate(img_list):
            job_no = i
            img = Image.open(item)
            cls_list = self.infer_(img, threshold)
            cls_list.sort(reverse=True, key=lambda x: x[1])
            if space:
                cls_list = [(cls.replace('_', ' '), score) for cls, score in cls_list]
            if escape:
                cls_list = [(re.sub(self.RE_SPECIAL, r'\\\1', cls), score) for cls, score in cls_list]

            if out_type == 'txt':
                with open(item[:item.rfind('.')] + '.txt', 'w', encoding='utf8') as f:
                    f.write(', '.join([name for name, prob in cls_list]))
            elif out_type == 'json':
                tag_dict[os.path.basename(item)] = ', '.join([name for name, prob in cls_list])

        if out_type == 'json':
            with open(os.path.join(path, 'image_captions.json'), 'w', encoding='utf8') as f:
                f.write(json.dumps(tag_dict, indent=2, ensure_ascii=False))

        return 'finish', ""

    def unload(self):
        if hasattr(self, 'model') and self.model is not None:
            self.last_model_name = None
            del self.model
            gc.collect()

            return 'model unload'
        return 'no model found'

