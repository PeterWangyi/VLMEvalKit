import sys
import torch
import warnings
from .base import BaseModel
from ..smp import *
from PIL import Image
from ..dataset import DATASET_TYPE
from vlmeval.smp.misc import get_cache_path
from vlmeval.smp.file import HFCacheRoot
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights
import json
from typing import (
    Any,
    Dict
)
BASE_PARAMS: Dict[str, Dict[str, Any]] = {
    "generate": dict(
        cfg_text_scale=4.0,
        cfg_img_scale=1.0,
        cfg_interval=[0.4, 1.0],
        timestep_shift=3.0,
        num_timesteps=50,
        cfg_renorm_min=1.0,
        cfg_renorm_type="global",
    ),
    "think_generate": dict(
        max_think_token_n=1000,
        do_sample=False,
        cfg_text_scale=4.0,
        cfg_img_scale=1.0,
        cfg_interval=[0.4, 1.0],
        timestep_shift=3.0,
        num_timesteps=50,
        cfg_renorm_min=1.0,
        cfg_renorm_type="global",
        think=True,
    ),
    "edit": dict(
        cfg_text_scale=4.0,
        cfg_img_scale=2.0,
        cfg_interval=[0.0, 1.0],
        timestep_shift=3.0,
        num_timesteps=50,
        cfg_renorm_min=0.0,
        cfg_renorm_type="text_channel",
    ),
    "think_edit": dict(
        max_think_token_n=1000,
        do_sample=False,
        cfg_text_scale=4.0,
        cfg_img_scale=2.0,
        cfg_interval=[0.0, 1.0],
        timestep_shift=3.0,
        num_timesteps=50,
        cfg_renorm_min=0.0,
        cfg_renorm_type="text_channel",
        think=True,
    ),
    "understanding": dict(
        max_think_token_n=4096,
        do_sample=False,
        understanding_output=True,
    ),
    "think_understanding": dict(
        max_think_token_n=4096,
        do_sample=False,
        understanding_output=True,
        think=True,
    ),
}


def get_model_path(model_path):
    if os.path.exists(model_path):
        return model_path
    else:
        cache_path = get_cache_path(model_path, repo_type='models')
        if cache_path is not None and os.path.exists(cache_path):
            return cache_path
        else:
            cache_root = HFCacheRoot()
            model_name = model_path.split('/')[-1]
            cache_path = os.path.join(cache_root, model_name)
            if os.path.exists(cache_path):
                return cache_path
            else:
                raise FileNotFoundError(f'Model {model_path} not found in {cache_root}')


def add_special_tokens(tokenizer):
    all_special_tokens = []
    for k, v in tokenizer.special_tokens_map.items():
        if isinstance(v, str):
            all_special_tokens.append(v)
        elif isinstance(v, list):
            all_special_tokens += v

    new_tokens = []

    if '<|im_start|>' not in all_special_tokens:
        new_tokens.append('<|im_start|>')

    if '<|im_end|>' not in all_special_tokens:
        new_tokens.append('<|im_end|>')

    if '<|vision_start|>' not in all_special_tokens:
        new_tokens.append('<|vision_start|>')

    if '<|vision_end|>' not in all_special_tokens:
        new_tokens.append('<|vision_end|>')

    num_new_tokens = tokenizer.add_tokens(new_tokens)
    bos_token_id = tokenizer.convert_tokens_to_ids('<|im_start|>')
    eos_token_id = tokenizer.convert_tokens_to_ids('<|im_end|>')
    start_of_image = tokenizer.convert_tokens_to_ids('<|vision_start|>')
    end_of_image = tokenizer.convert_tokens_to_ids('<|vision_end|>')

    new_token_ids = dict(
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        start_of_image=start_of_image,
        end_of_image=end_of_image,
    )

    return tokenizer, new_token_ids, num_new_tokens


class Bagel(BaseModel):

    INSTALL_REQ = True
    INTERLEAVE = True

    def __init__(
        self,
        model_path='ByteDance-Seed/BAGEL-7B-MoT',
        config_path='./zoe_eval/configs/bagel_config.json',
        precision='bf16',
        variant='origin',
        **kwargs
    ):
        self.variant = variant
        self.precision = precision
        self.model_path = model_path
        self.config_path = os.environ.get('Bagel_cfg_path', config_path)
        self.max_new_token = int(os.environ.get('Bagel_max_new_token', 2048))

        # 1. Load config and set bagel project root
        config, root = type(self).load_config(self.config_path)

        print(f"in origin bagel")

        self.root = root
        if self.root not in sys.path:
            sys.path.append(self.root)

        # 2. Parse config
        self.model_path = get_model_path(config.get('model_path', self.model_path))
        self.mode = config['mode']
        self.out_img_dir = config['out_img_dir']
        self.checkpoint_path = config.get("checkpoint_path") or os.path.join(self.model_path, "ema.safetensors")

        ckpt_override = os.environ.get('Bagel_ckpt_path', None)
        if ckpt_override is not None:
            # assert self.variant == 'vggt', "Only VGGT models support ckpt override now!"
            self.checkpoint_path = ckpt_override

        print(f"Using bagel root dir: {self.root}, Using checkpoint: {self.checkpoint_path}")

        # 3. Build model
        model, vae_model, tokenizer, new_token_ids, vit_transform, vae_transform = self._build_model_variant()

        # 4. Load Checkpoint
        model = self._load_model_weights(model)

        # 5. Build inferencer
        self.tokenizer = tokenizer
        self.new_token_ids = new_token_ids
        self.vit_transform = vit_transform

        from inferencer import InterleaveInferencer
        self.inferencer = InterleaveInferencer(
            model=model,
            vae_model=vae_model,
            tokenizer=tokenizer,
            vae_transform=vae_transform,
            vit_transform=vit_transform,
            new_token_ids=new_token_ids,
        )

        torch.cuda.empty_cache()

    @classmethod
    def load_config(cls, config_path):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"{config_path} not found")

        config = json.load(open(config_path))

        root = config.get('Bagel_ROOT') or os.environ.get('Bagel_ROOT')
        if not root:
            raise FileNotFoundError("Bagel_ROOT not set")

        return config, root

    def _build_model_variant(self):
        from data.transforms import ImageTransform
        from modeling.bagel import (
            BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
        )
        from modeling.qwen2 import Qwen2Tokenizer
        from modeling.autoencoder import load_ae

        # build llm config
        llm_config = Qwen2Config.from_json_file(os.path.join(self.model_path, "llm_config.json"))
        llm_config.qk_norm = True
        llm_config.tie_word_embeddings = False
        llm_config.layer_module = "Qwen2MoTDecoderLayer"

        # build vit config
        vit_config = SiglipVisionConfig.from_json_file(os.path.join(self.model_path, "vit_config.json"))
        vit_config.rope = False
        vit_config.num_hidden_layers -= 1

        vit_transform = ImageTransform(980, 224, 14)
        vae_transform = ImageTransform(1024, 512, 16)

        # build vae config
        vae_model, vae_config = load_ae(local_path=os.path.join(self.model_path, "ae.safetensors"))

        # build tokenizer
        tokenizer = Qwen2Tokenizer.from_pretrained(self.model_path)
        tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

        # build model
        if self.variant == 'origin':
            model_config = BagelConfig(
                visual_gen=True,
                visual_und=True,
                llm_config=llm_config,
                vit_config=vit_config,
                vae_config=vae_config,
                latent_patch_size=2,
                max_latent_size=64,
                vit_max_num_patch_per_side=70,
                connector_act='gelu_pytorch_tanh'
            )

            # Tips:
            # Currently using direct model initialization (without init_empty_weights)
            # to ensure compatibility with models trained by zoetrope,
            # which do not rely on meta-init or parameter dispatching.
            # This simplifies code and avoids force_hooks logic during checkpoint loading.

            # with init_empty_weights():
            language_model = Qwen2ForCausalLM(llm_config)
            vit_model = SiglipVisionModel(vit_config)

            model = Bagel(language_model, vit_model, model_config)
            model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)

        elif self.variant == 'vggt':
            from modeling.zoevl3d import ZoeVL3DConfig, ZoeVL3D
            from modeling.vggt.models.vggt import VGGT
            model_config = ZoeVL3DConfig(
                visual_gen=False,
                visual_und=True,
                visual_und_3d=True,
                llm_config=llm_config,
                vit_config=vit_config,
                vggt_embed_dim=1024,
                vggt_max_num_patch_per_side=70,
                latent_patch_size=2,
                max_latent_size=64,
                vit_max_num_patch_per_side=70
            )
            # with init_empty_weights():
            language_model = Qwen2ForCausalLM(llm_config)
            vit_model = SiglipVisionModel(vit_config)
            vggt_model = VGGT.from_pretrained("facebook/VGGT-1B")

            model = ZoeVL3D(
                language_model,
                vit_model,
                vggt_model,
                model_config
            )
            model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)

        else:
            raise ValueError(f"Unsupported variant: {self.variant}")

        return model, vae_model, tokenizer, new_token_ids, vit_transform, vae_transform

    def _load_model_weights(self, model):
        device_map = infer_auto_device_map(
            model,
            max_memory={i: "80GiB" for i in range(torch.cuda.device_count())},
            no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"]
        )

        same_device_modules = [
            'language_model.model.embed_tokens',
            'time_embedder',
            'latent_pos_embed',
            'vae2llm',
            'llm2vae',
            'connector',
            'vit_pos_embed'
        ]

        if torch.cuda.device_count() == 1:
            first_device = device_map.get(same_device_modules[0], "cuda:0")
            for k in same_device_modules:
                if k in device_map:
                    device_map[k] = first_device
                else:
                    device_map[k] = "cuda:0"
        else:
            first_device = device_map.get(same_device_modules[0])
            for k in same_device_modules:
                if k in device_map:
                    device_map[k] = first_device

        if self.precision == 'bf16':
            # This disables force_hooks, since the model is initialized with real weights.
            # Do NOT disable force_hooks if you use init_empty_weights(), otherwise it will fail at runtime.

            model = load_checkpoint_and_dispatch(
                model,
                checkpoint=self.checkpoint_path,
                device_map=device_map,
                offload_buffers=True,
                offload_folder="offload",
                dtype=torch.bfloat16,
                force_hooks=False,
            ).eval()

        elif self.precision == 'nf4':
            from accelerate.utils import BnbQuantizationConfig, load_and_quantize_model

            model = load_and_quantize_model(
                model, weights_location=self.checkpoint_path,
                bnb_quantization_config=BnbQuantizationConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=False, bnb_4bit_quant_type="nf4"
                ),
                device_map=device_map, offload_folder="offload"
            ).eval()

        elif self.precision == 'int8':
            from accelerate.utils import BnbQuantizationConfig, load_and_quantize_model

            model = load_and_quantize_model(
                model, weights_location=self.checkpoint_path,
                bnb_quantization_config=BnbQuantizationConfig(
                    load_in_8bit=True, torch_dtype=torch.float32
                ),
                device_map=device_map, offload_folder="offload"
            ).eval()

        else:
            raise NotImplementedError(f"Unsupported precision: {self.precision}")

        return model

    def generate_inner(self, message, dataset=None):
        '''
        nums_beams = 3
        '''
        if "EDIT" in DATASET_TYPE(dataset):
            if self.mode in ['edit','think_edit']:
                mode = self.mode
            else:
                print(f'{dataset} only support edit or think_edit, default mode is think_edit')
                mode = 'think_edit'
        elif "GEN" in DATASET_TYPE(dataset):
            if self.mode in ['think_generate','generate']:
                mode = self.mode
            else:
                print(f'{dataset} only support generate or think_generate, default mode is think_generate')
                mode = 'think_generate'
        else:
            mode = self.mode

        input_lists = []
        for x in message:
            if x['type'] == 'text':
                conversation = x['value']
                if mode == "generate" or mode == "think_generate":
                    # TODO: split index && prompt
                    instruction = conversation.split(':')[1]
                    input_lists.append(instruction)
                else:
                    input_lists.append(conversation)

            elif x['type'] == 'image':
                img_path = x['value']
                image = Image.open(img_path)
                input_lists.append(image)

        # print(f"------------------------")
        # print(f" Bagel input img pic: {print_items} ")
        # print(f"\n")
        # print(f" Bagel input list   : {input_lists}")
        # print(f"------------------------")

        params = dict(BASE_PARAMS[mode])
        understanding_output_flag = params.pop("understanding_output", False)
        think_flag = params.pop("think", False)
        params['max_think_token_n'] = self.max_new_token

        if self.variant == "vggt":
            params["vggt"] = True

        if mode == "understanding" or mode == "think_understanding":
            res = self.inferencer.interleave_inference(
                input_lists=input_lists,
                understanding_output=understanding_output_flag,
                think=think_flag,
                **params)
        elif mode == "edit" or mode == "think_edit":
            res = self.inferencer.interleave_inference(
                input_lists=input_lists,
                think=think_flag,
                understanding_output=understanding_output_flag,
                **params)
        elif mode == "generate" or mode == "think_generate":
            res = self.inferencer.interleave_inference(
                input_lists=input_lists,
                think=think_flag,
                understanding_output=understanding_output_flag,
                **params)

        ret = {'image': [], 'text': []}
        for i in res:
            if isinstance(i, Image.Image):
                ret['image'].append(i)
            elif isinstance(i, str):
                ret['text'].append(i)

        img_cnt, txt_cnt = len(ret['image']), len(ret['text'])
        if img_cnt + txt_cnt != 1:
            raise ValueError(
                f"[OutputError] Current expect exactly 1 image OR 1 text. Got images={img_cnt}, texts={txt_cnt}"
            )

        ret['image'] = ret['image'][0] if img_cnt else None
        ret['text'] = ret['text'][0] if txt_cnt else None

        if mode in ['edit','think_edit']:
            if not os.path.exists(self.out_img_dir):
                os.makedirs(self.out_img_dir)
            if not os.path.exists(os.path.join(self.out_img_dir,f'{dataset}_images/')):
                os.makedirs(os.path.join(self.out_img_dir, f'{dataset}_images/'))
            subdir = img_path.split('/')[-2].replace('_images','')
            img_out_path = os.path.join(self.out_img_dir,f'{dataset}_images/',subdir)
            if not os.path.exists(img_out_path):
                os.makedirs(img_out_path)
            ret['image'].save(os.path.join(img_out_path, subdir + '_' + img_path.split('/')[-1]))
            ret['image'] = os.path.join(img_out_path, subdir + '_' + img_path.split('/')[-1])

        elif mode in ['generate','think_generate']:
            if not os.path.exists(self.out_img_dir):
                os.makedirs(self.out_img_dir)
            img_out_path = os.path.join(self.out_img_dir,f'{dataset}_images/')
            if not os.path.exists(img_out_path):
                os.makedirs(img_out_path)
            index = conversation.split(':')[0]
            ret['image'].save(os.path.join(img_out_path, index + '.jpg'))
            ret['image'] = os.path.join(img_out_path, index + '.jpg')

        else:
            res = ret['text']

        return res


class Bagel_SingleCard(Bagel):
    PARALLEL_SAFE = True

    def __init__(self,
                 model_path='ByteDance-Seed/BAGEL-7B-MoT',
                 config_path='./zoe_eval/configs/bagel_config.json',
                 precision='bf16',
                 variant='origin',
                 device_id=None,
                 force_single_device=True,
                 replica_id=0,
                 **kwargs):
        self._device_id = device_id
        self._force_single_device = force_single_device
        self._replica_id = int(replica_id)

        if torch.cuda.is_available() and device_id is not None:
            torch.cuda.set_device(device_id)
        self.device = torch.device(
            f"cuda:{device_id}" if (torch.cuda.is_available() and device_id is not None) else "cpu"
        )

        print(f"[Bagel_Single] target device = {self.device}, input device id : {self._device_id}")
        self.variant = variant
        self.precision = precision
        self.model_path = model_path
        self.config_path = os.environ.get('Bagel_cfg_path', config_path)
        self.max_new_token = int(os.environ.get('Bagel_max_new_token', 1024))

        # 1. Load config and set bagel project root
        config, root = type(self).load_config(self.config_path)

        print(f"in origin bagel")

        self.root = root
        if self.root not in sys.path:
            sys.path.append(self.root)

        # 2. Parse config
        self.model_path = get_model_path(config.get('model_path', self.model_path))
        self.mode = config['mode']
        self.out_img_dir = config['out_img_dir']
        self.checkpoint_path = config.get("checkpoint_path") or os.path.join(self.model_path, "ema.safetensors")

        ckpt_override = os.environ.get('Bagel_ckpt_path', None)
        if ckpt_override is not None:
            # assert self.variant == 'vggt', "Only VGGT models support ckpt override now!"
            self.checkpoint_path = ckpt_override

        print(f"Using bagel root dir: {self.root}, Using checkpoint: {self.checkpoint_path}")

        # 3. Build model
        model, vae_model, tokenizer, new_token_ids, vit_transform, vae_transform = self._build_model_variant()

        # 4. Load Checkpoint
        model = self._load_model_weights(model)

        # 5. Build inferencer
        self.tokenizer = tokenizer
        self.new_token_ids = new_token_ids
        self.vit_transform = vit_transform

        from inferencer import InterleaveInferencer
        self.inferencer = InterleaveInferencer(
            model=model,
            vae_model=vae_model,
            tokenizer=tokenizer,
            vae_transform=vae_transform,
            vit_transform=vit_transform,
            new_token_ids=new_token_ids,
        )

        torch.cuda.empty_cache()

    # def _load_model_weights(self, model, offload_folder="offload"):
    #     if not (self._force_single_device and self._device_id is not None):
    #         return super()._load_model_weights(model)

    #     raw_map = infer_auto_device_map(
    #         model,
    #         max_memory={i: "80GiB" for i in range(torch.cuda.device_count())},
    #         no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"]
    #     )

    #     # Force on one card
    #     target = f"cuda:{self._device_id}" if torch.cuda.is_available() else "cpu"
    #     print(f"torch.cuda.device_count() = {torch.cuda.device_count()}",f"target device = {self._device_id}")
    #     device_map = {k: target for k in raw_map.keys()} if isinstance(raw_map, dict) else {"": target}

    #     offload_folder = f"{offload_folder}_gpu{self._device_id}_r{self._replica_id}"
    #     # os.makedirs(offload_folder, exist_ok=True)

    #     if self.precision == 'bf16':
    #         model = load_checkpoint_and_dispatch(
    #             model,
    #             checkpoint=self.checkpoint_path,
    #             device_map=device_map,
    #             offload_buffers=True,
    #             offload_folder=offload_folder,
    #             dtype=torch.bfloat16,
    #             force_hooks=False,      ######## Attention!!!
    #         ).eval()
    #     elif self.precision == 'nf4':
    #         from accelerate.utils import BnbQuantizationConfig, load_and_quantize_model
    #         model = load_and_quantize_model(
    #             model, weights_location=self.checkpoint_path,
    #             bnb_quantization_config=BnbQuantizationConfig(
    #                 load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
    #                 bnb_4bit_use_double_quant=False, bnb_4bit_quant_type="nf4"
    #             ),
    #             device_map=device_map, offload_folder=offload_folder
    #         ).eval()
    #     elif self.precision == 'int8':
    #         from accelerate.utils import BnbQuantizationConfig, load_and_quantize_model
    #         model = load_and_quantize_model(
    #             model, weights_location=self.checkpoint_path,
    #             bnb_quantization_config=BnbQuantizationConfig(
    #                 load_in_8bit=True, torch_dtype=torch.float32
    #             ),
    #             device_map=device_map, offload_folder=offload_folder
    #         ).eval()
    #     else:
    #         raise NotImplementedError(self.precision)

    #     return model


    def _load_model_weights(self, model, offload_folder="offload"):
        # 与父类保持行为一致：只有在明确要求“单卡强制”时，才走单卡分支
        if not (self._force_single_device and self._device_id is not None):
            return super()._load_model_weights(model)

        # 目标设备
        target = f"cuda:{self._device_id}" if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            torch.cuda.set_device(self._device_id)

        # === 关键差异：只给目标 GPU 设置显存上限，并允许 CPU 承接溢出 ===
        # 这让 infer_auto_device_map 生成“单卡 + CPU”的 device_map，而不是“全塞单卡”
        max_memory = {
            # 注意：这里的 key 用“GPU 序号 int”即可（与父类一致），只给这一张卡
            int(self._device_id): os.environ.get("BAGEL_GPU_MEM", "80GiB"),
            # "cpu": os.environ.get("BAGEL_CPU_MEM", "128GiB"),
        }

        device_map = infer_auto_device_map(
            model,
            max_memory=max_memory,
            no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
            # 允许回退到 CPU（否则会强塞进 GPU 导致 OOM）
            # fallback_to_cpu=True,
            # 与父类对齐精度提示（可选，不同 accelerate 版本可能忽略）
            dtype=(torch.bfloat16 if self.precision == "bf16" else torch.float16),
        )

        # 与父类对齐：以下模块必须在同一张设备上（这里统一放到 target）
        same_device_modules = [
            'language_model.model.embed_tokens',
            'time_embedder',
            'latent_pos_embed',
            'vae2llm',
            'llm2vae',
            'connector',
            'vit_pos_embed'
        ]

        # 1) 把 same_device_modules 统一映射到 target
        for k in same_device_modules:
            device_map[k] = target

        # 2) 如果 map 中出现了其它 cuda:x（非目标卡），统一改成 target
        for k, v in list(device_map.items()):
            if isinstance(v, str) and v.startswith("cuda:") and v != target:
                device_map[k] = target

        # offload 目录按副本隔离，避免多副本/多进程踩目录
        offload_folder = f"{offload_folder}_gpu{self._device_id}_r{self._replica_id}"
        os.makedirs(offload_folder, exist_ok=True)

        if self.precision == 'bf16':
            # **与父类保持一致**：不开 force_hooks（父类是 False）
            model = load_checkpoint_and_dispatch(
                model,
                checkpoint=self.checkpoint_path,
                device_map=device_map,
                offload_buffers=True,       # 只 offload buffer，不是权重；权重落 CPU 由 device_map['cpu'] 决定
                offload_folder=offload_folder,
                dtype=torch.bfloat16,
                force_hooks=False,          # ← 关键修正：不要开 True
            ).eval()

        elif self.precision == 'nf4':
            from accelerate.utils import BnbQuantizationConfig, load_and_quantize_model
            model = load_and_quantize_model(
                model, weights_location=self.checkpoint_path,
                bnb_quantization_config=BnbQuantizationConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=False, bnb_4bit_quant_type="nf4"
                ),
                device_map=device_map, offload_folder=offload_folder
            ).eval()

        elif self.precision == 'int8':
            from accelerate.utils import BnbQuantizationConfig, load_and_quantize_model
            model = load_and_quantize_model(
                model, weights_location=self.checkpoint_path,
                bnb_quantization_config=BnbQuantizationConfig(
                    load_in_8bit=True, torch_dtype=torch.float32
                ),
                device_map=device_map, offload_folder=offload_folder
            ).eval()

        else:
            raise NotImplementedError(f"Unsupported precision: {self.precision}")

        return model
