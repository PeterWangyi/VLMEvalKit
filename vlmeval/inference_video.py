import torch
import torch.distributed as dist
from vlmeval.config import supported_VLM
from vlmeval.utils import track_progress_rich
from vlmeval.smp import *

from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
import time
from queue import Empty, Full

FAIL_MSG = 'Failed to obtain answer via API.'
FAIL_MSGS = [
    'Failed to obtain answer via API.',
    '[ERROR]',
    'Hit max new token.',
    'Failed: Model exited.',
    'Failed',
    '<',
]

logger = get_logger(name='test')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, nargs='+', required=True)
    parser.add_argument('--model', type=str, nargs='+', required=True)
    parser.add_argument('--nproc', type=int, default=4, required=True)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    return args


# Only API model is accepted
def infer_data_api(model, work_dir, model_name, dataset, samples_dict={}, api_nproc=4):
    rank, world_size = get_rank_and_world_size()
    assert rank == 0 and world_size == 1
    dataset_name = dataset.dataset_name
    model = supported_VLM[model_name]() if isinstance(model, str) else model
    assert getattr(model, 'is_api', False)

    indices = list(samples_dict.keys())
    if getattr(model,'backend', None) == 'genai':
        if dataset.nframe > 0:
            print(
                'Gemini model (with genai backend) does not support nframe, '
                'will set its VIDEO_LLM to False to enable multi-image input for video.'
            )
            setattr(model, 'VIDEO_LLM', False)
        else:
            print('Gemini model (with genai backend) is a video-llm, '
                  'will reset fps setting in model to match the dataset.')
            setattr(model, 'fps', dataset.fps)
            print(f'The fps is set to {dataset.fps} for the model {model_name}.')
    elif getattr(model,'backend', None) == 'vertex':
        print('Gemini model (with vertex backend) does not support video input, '
              'will set its VIDEO_LLM to False to enable multi-image input for video.')
        setattr(model, 'VIDEO_LLM', False)

    packstr = 'pack' if getattr(dataset, 'pack', False) else 'nopack'
    build_prompt_input = [(samples_dict[idx], getattr(model, 'VIDEO_LLM', False)) for idx in indices]
    if dataset.nframe > 0:
        struct_tmp_file = f'{work_dir}/{model_name}_{dataset_name}_{dataset.nframe}frame_{packstr}_structs.pkl'
    else:
        struct_tmp_file = f'{work_dir}/{model_name}_{dataset_name}_{dataset.fps}fps_{packstr}_structs.pkl'
    structs = track_progress_rich(
        dataset.build_prompt,
        tasks=build_prompt_input,
        nproc=api_nproc,
        save=struct_tmp_file,
        keys=indices,
    )

    if dataset.nframe > 0:
        out_file = f'{work_dir}/{model_name}_{dataset_name}_{dataset.nframe}frame_{packstr}_supp.pkl'
    else:
        out_file = f'{work_dir}/{model_name}_{dataset_name}_{dataset.fps}fps_{packstr}_supp.pkl'
    res = load(out_file) if osp.exists(out_file) else {}

    structs = [s for i, s in zip(indices, structs) if i not in res or res[i] == FAIL_MSG]
    structs = [struct for struct in structs if struct is not None]
    indices = [i for i in indices if i not in res or res[i] == FAIL_MSG]

    gen_func = model.generate
    structs = [dict(message=struct, dataset=dataset_name) for struct in structs]

    if len(structs):
        track_progress_rich(gen_func, structs, nproc=api_nproc, chunksize=api_nproc, save=out_file, keys=indices)

    res = load(out_file)
    return res


def infer_data(model, model_name, work_dir, dataset, out_file, verbose=False, api_nproc=4, use_vllm=False):
    res = load(out_file) if osp.exists(out_file) else {}
    rank, world_size = get_rank_and_world_size()
    dataset_name = dataset.dataset_name

    sample_indices = list(dataset.videos) if getattr(dataset, 'pack', False) else list(dataset.data['index'])
    samples = list(dataset.videos) if getattr(dataset, 'pack', False) else list(range(len(dataset.data)))
    sample_map = {i: s for i, s in zip(sample_indices, samples)}

    sample_indices_sub = sample_indices[rank::world_size]
    if np.all([idx in res for idx in sample_indices_sub]):
        return model
    sample_indices_subrem = [x for x in sample_indices_sub if x not in res]

    kwargs = {}
    if model_name is not None and (
        'Llama-4' in model_name
        or 'Qwen2-VL' in model_name
        or 'Qwen2.5-VL' in model_name
        or 'Qwen2.5-Omni' in model_name
    ):
        kwargs = {'use_vllm': use_vllm}

    # (25.06.05) In newer version of transformers (after 4.50), with device_map='auto' and torchrun launcher,
    # Transformers automatically adopt TP parallelism, which leads to compatibility problems with VLMEvalKit
    # (In VLMEvalKit, we use torchrun to launch multiple model instances on a single node).
    # To bypass this problem, we unset `WORLD_SIZE` before building the model to not use TP parallel.
    ws_bak = os.environ.pop('WORLD_SIZE', None)
    model = supported_VLM[model_name](**kwargs) if isinstance(model, str) else model
    if ws_bak:
        os.environ['WORLD_SIZE'] = ws_bak

    is_api = getattr(model, 'is_api', False)
    if is_api:
        assert world_size == 1
        supp = infer_data_api(
            model=model,
            work_dir=work_dir,
            model_name=model_name,
            dataset=dataset,
            samples_dict={k: sample_map[k] for k in sample_indices_subrem},
            api_nproc=api_nproc)
        for k in sample_indices_subrem:
            assert k in supp
        res.update(supp)
        dump(res, out_file)
        return model

    assert not getattr(dataset, 'pack', False), 'Current model not supported pack mode!'
    if 'megabench' in dataset_name.lower() and 'llava_onevision' in model_name:
        print(
            'LLaVA-OneVision does not support Megabench dataset as video dataset, '
            'will set its VIDEO_LLM to False to enable multi-image input for video.'
        )
        setattr(model, 'VIDEO_LLM', False)

    for i, idx in tqdm(enumerate(sample_indices_subrem)):
        if idx in res:
            continue
        if getattr(model, 'nframe', None) is not None and getattr(model, 'nframe', 0) > 0:
            if dataset.nframe > 0:
                if getattr(model, 'nframe', 0) != dataset.nframe:
                    print(f'{model_name} is a video-llm model, nframe is set to {dataset.nframe}, not using default')
                    setattr(model, 'nframe', dataset.nframe)
            elif getattr(model, 'fps', 0) == 0:
                raise ValueError(f'fps is not suitable for {model_name}')
            else:
                setattr(model, 'nframe', None)
        if getattr(model, 'fps', None) is not None and getattr(model, 'fps', 0) > 0:
            if dataset.fps > 0:
                if getattr(model, 'fps', 0) != dataset.fps:
                    print(f'{model_name} is a video-llm model, fps is set to {dataset.fps}, not using default')
                    setattr(model, 'fps', dataset.fps)
            elif getattr(model, 'nframe', 0) == 0:
                raise ValueError(f'nframe is not suitable for {model_name}')
            else:
                setattr(model, 'fps', None)
        if (
            'Qwen2-VL' in model_name
            or 'Qwen2.5-VL' in model_name
            or 'Qwen2.5-Omni' in model_name
        ):
            if getattr(model, 'nframe', None) is None and dataset.nframe > 0:
                print(f'using {model_name} default setting for video, dataset.nframe is ommitted')
            if getattr(model, 'fps', None) is None and dataset.fps > 0:
                print(f'using {model_name} default setting for video, dataset.fps is ommitted')
        if 'SUB_DATASET' in dataset.data.iloc[sample_map[idx]]:
            dataset_name = dataset.data.iloc[sample_map[idx]]['SUB_DATASET']
        if hasattr(model, 'use_custom_prompt') and model.use_custom_prompt(dataset_name):
            if dataset.nframe == 0:
                raise ValueError(f'nframe must be set for custom prompt, fps is not suitable for {model_name}')
            struct = model.build_prompt(
                dataset.data.iloc[sample_map[idx]], dataset=dataset, video_llm=getattr(model, 'VIDEO_LLM', False)
            )
        else:
            struct = dataset.build_prompt(
                sample_map[idx], video_llm=getattr(model, 'VIDEO_LLM', False)
            )
        if struct is None:
            continue

        # If `SKIP_ERR` flag is set, the model will skip the generation if error is encountered
        if os.environ.get('SKIP_ERR', False) == '1':
            FAIL_MSG = 'Failed to obtain answer'
            try:
                response = model.generate(message=struct, dataset=dataset_name)
            except RuntimeError as err:
                torch.cuda.synchronize()
                warnings.error(f'{type(err)} {str(err)}')
                response = f'{FAIL_MSG}: {type(err)} {str(err)}'
        else:
            response = model.generate(message=struct, dataset=dataset_name)
        torch.cuda.empty_cache()

        if verbose:
            print(response, flush=True)

        res[idx] = response
        if (i + 1) % 20 == 0:
            dump(res, out_file)

    res = {k: res[k] for k in sample_indices_sub}
    dump(res, out_file)
    return model


# A wrapper for infer_data, do the pre & post processing
def infer_data_job_video(
        model,
        work_dir,
        model_name,
        dataset,
        result_file_name,
        verbose=False,
        api_nproc=4,
        use_vllm=False):

    dataset_name = dataset.dataset_name
    rank, world_size = get_rank_and_world_size()
    result_file = osp.join(work_dir, result_file_name)
    # Dump Predictions to Prev File if result file exists
    # if osp.exists(result_file):
    #     return model

    tmpl = osp.join(work_dir, '{}' + f'{world_size}_{osp.splitext(result_file_name)[0]}.pkl')
    out_file = tmpl.format(rank)

    print(f"outfile: {out_file}")

    if osp.exists(result_file):

        print(f"Result file exist: {result_file}, dump to pkl.")

        res = load(result_file)

        results = {k: v for k, v in zip(res['index'], res['prediction'])}
        results = {k: v for k, v in results.items()
                if not any(msg in str(v) for msg in FAIL_MSGS)}

        dump(results, out_file)


    if 'ddp' in model_name.lower() or 'single' in model_name.lower():
        devices = list(range(torch.cuda.device_count()))
        replicas_per_device = int(os.environ.get('replicas_per_device', 1))                # 或 2

        print(f"in ddp --------------------------")

        model = infer_data_unified(
            model=model,                  # 可以是注册名字符串，或已构造好的实例
            model_name=model_name,
            work_dir=work_dir,
            dataset=dataset,
            out_file=out_file,
            verbose=verbose,
            api_nproc=api_nproc,
            use_vllm=use_vllm,
            devices=devices,              # 例如 [0,1,2,3]；不传则默认用所有可见 GPU
            replicas_per_device=replicas_per_device  # 例如 2（每卡起2个进程/模型）
        )
    else:
        model = infer_data(
            model=model,
            model_name=model_name,
            work_dir=work_dir,
            dataset=dataset,
            out_file=out_file,
            verbose=verbose,
            api_nproc=api_nproc,
            use_vllm=use_vllm)

    if world_size > 1:
        dist.barrier()

    if rank == 0:
        data_all = {}
        for i in range(world_size):
            data_all.update(load(tmpl.format(i)))

        meta = dataset.data
        if dataset_name == 'MMBench-Video' and getattr(dataset, 'pack', False):
            meta, vstats = dataset.load_pack_answers(data_all)
            print(f'Statitics of Pack Video Inference: {vstats}')
        else:
            for x in meta['index']:
                assert x in data_all
            meta['prediction'] = [str(data_all[x]) for x in meta['index']]
            if 'image' in meta:
                meta.pop('image')

        dump(meta, result_file)
        for i in range(world_size):
            os.remove(tmpl.format(i))
    return model


def make_registry_builder(model_name: str, extra_kwargs: dict):
    ctor = supported_VLM[model_name]  # 可能是类或 partial，反正是可调用
    def _builder(*, device_id=None, force_single_device=True, **kw):
        # 传参优先级：调用侧 kw > extra_kwargs
        merged = {**extra_kwargs, **kw}
        return ctor(device_id=device_id, force_single_device=force_single_device, **merged)
    return _builder


# ========= 子进程：一进程绑定一张物理卡；进程内只见到“局部 cuda:0” =========
def _video_worker_loop(physical_gpu_id, in_q, out_q, model_name: str, extra_kwargs: dict):
    import os, sys, warnings
    # 1) 先隔离可见卡（务必在 import torch/transformers 之前）
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu_id)
    for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):
        os.environ.pop(k, None)

    # 2) 让子进程能 import 到你的工程包（按你的路径改）
    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if pkg_root not in sys.path:
        sys.path.insert(0, pkg_root)

    # 3) 再 import（此时只会看到 1 张卡：local cuda:0）
    import torch

    # 4) 构造模型 —— 进程内只见到一张卡，绑定“局部 cuda:0”
    kw = dict(extra_kwargs or {})
    kw["force_single_device"] = True
    kw["device_id"] = physical_gpu_id  # 进程内局部 0
    builder = make_registry_builder(model_name, {})
    model = builder(**kw)

    model.fps = kw['fps']
    model.nframe = kw['nframe']

    # 5) 处理任务
    while True:
        item = in_q.get()
        if item is None:
            break
        idx, struct, dataset_name = item
        try:
            resp = model.generate(message=struct, dataset=dataset_name)
        except RuntimeError as err:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            warnings.warn(f'{type(err)} {str(err)}')
            resp = f'Failed to obtain answer: {type(err)} {str(err)}'
        except Exception as err:
            resp = f'Failed to obtain answer: {type(err)} {str(err)}'
        out_q.put((idx, resp))

        logger.info(f"queue put idx : {idx} {resp}")

def infer_data_unified(model,
               model_name,
               work_dir,
               dataset,
               out_file,
               verbose=True,
               api_nproc=4,
               use_vllm=False,
               devices=None,
               replicas_per_device=1):
    """
    多卡多模型（多进程）视频推理版本：
    - 当 `model` 是字符串（注册名）时：父进程起子进程池（每卡 replicas_per_device 个），
      父进程构建输入并轮询分发到子进程；子进程在本地 cuda:0 上推理并回传结果。
    - 当 `model` 已是实例时：保持你的原始串行路径。
    """
    import os, warnings, numpy as np, torch, multiprocessing as mp
    from tqdm import tqdm

    res = load(out_file) if osp.exists(out_file) else {}
    rank, world_size = get_rank_and_world_size()
    dataset_name = dataset.dataset_name

    sample_indices = list(dataset.videos) if getattr(dataset, 'pack', False) else list(dataset.data['index'])
    samples = list(dataset.videos) if getattr(dataset, 'pack', False) else list(range(len(dataset.data)))
    sample_map = {i: s for i, s in zip(sample_indices, samples)}

    sample_indices_sub = sample_indices[rank::world_size]
    if np.all([idx in res for idx in sample_indices_sub]):
        return model
    sample_indices_subrem = [x for x in sample_indices_sub if x not in res]
    if len(sample_indices_subrem) == 0:
        return model

    # 额外 kwargs
    kwargs = {}
    if model_name is not None and (
        'Llama-4' in model_name
        or 'Qwen2-VL' in model_name
        or 'Qwen2.5-VL' in model_name
        or 'Qwen2.5-Omni' in model_name
    ):
        kwargs = {'use_vllm': use_vllm}

    # 统一：只在“模型已实例”时本地构造；字符串时由子进程构造
    if not isinstance(model, str):
        # ====== 原串行路径（保持你现有逻辑不变） ======
        # Qwen 默认 VIDEO_LLM 提示
        if 'megabench' in dataset_name.lower() and 'llava_onevision' in model_name:
            print(
                'LLaVA-OneVision does not support Megabench dataset as video dataset, '
                'will set its VIDEO_LLM to False to enable multi-image input for video.'
            )
            setattr(model, 'VIDEO_LLM', False)

        for i, idx in tqdm(enumerate(sample_indices_subrem), total=len(sample_indices_subrem),
                           desc=f'Infer {model_name}/{dataset_name}, Rank {rank}/{world_size}'):
            if idx in res:
                continue

            # —— 与你原本完全一致的 “视频参数对齐 + 构造输入” —— #
            if getattr(model, 'nframe', None) is not None and getattr(model, 'nframe', 0) > 0:
                if dataset.nframe > 0:
                    if getattr(model, 'nframe', 0) != dataset.nframe:
                        print(f'{model_name} is a video-llm model, nframe is set to {dataset.nframe}, not using default')
                        setattr(model, 'nframe', dataset.nframe)
                elif getattr(model, 'fps', 0) == 0:
                    raise ValueError(f'fps is not suitable for {model_name}')
                else:
                    setattr(model, 'nframe', None)

            if getattr(model, 'fps', None) is not None and getattr(model, 'fps', 0) > 0:
                if dataset.fps > 0:
                    if getattr(model, 'fps', 0) != dataset.fps:
                        print(f'{model_name} is a video-llm model, fps is set to {dataset.fps}, not using default')
                        setattr(model, 'fps', dataset.fps)
                elif getattr(model, 'nframe', 0) == 0:
                    raise ValueError(f'nframe is not suitable for {model_name}')
                else:
                    setattr(model, 'fps', None)

            if (
                'Qwen2-VL' in model_name
                or 'Qwen2.5-VL' in model_name
                or 'Qwen2.5-Omni' in model_name
            ):
                if getattr(model, 'nframe', None) is None and dataset.nframe > 0:
                    print(f'using {model_name} default setting for video, dataset.nframe is ommitted')
                if getattr(model, 'fps', None) is None and dataset.fps > 0:
                    print(f'using {model_name} default setting for video, dataset.fps is ommitted')

            dname = dataset_name
            if 'SUB_DATASET' in dataset.data.iloc[sample_map[idx]]:
                dname = dataset.data.iloc[sample_map[idx]]['SUB_DATASET']

            if hasattr(model, 'use_custom_prompt') and model.use_custom_prompt(dname):
                if dataset.nframe == 0:
                    raise ValueError(f'nframe must be set for custom prompt, fps is not suitable for {model_name}')
                struct = model.build_prompt(
                    dataset.data.iloc[sample_map[idx]], dataset=dataset, video_llm=getattr(model, 'VIDEO_LLM', False)
                )
            else:
                struct = dataset.build_prompt(
                    sample_map[idx], video_llm=getattr(model, 'VIDEO_LLM', False)
                )

            # SKIP_ERR 保持不变
            if os.environ.get('SKIP_ERR', False) == '1':
                FAIL_MSG = 'Failed to obtain answer'
                try:
                    response = model.generate(message=struct, dataset=dname)
                except RuntimeError as err:
                    torch.cuda.synchronize()
                    warnings.warn(f'{type(err)} {str(err)}')
                    response = f'{FAIL_MSG}: {type(err)} {str(err)}'
            else:
                response = model.generate(message=struct, dataset=dname)
            torch.cuda.empty_cache()

            if verbose:
                print(response, flush=True)
            res[idx] = response
            if ((i + 1) % 20) == 0:
                dump(res, out_file)

        res = {k: res[k] for k in sample_indices_sub}
        dump(res, out_file)
        return model

    # ====== 多进程多卡路径（model 是字符串） ======
    # 规避 TP：unset WORLD_SIZE，只在子进程构造模型（你图片版也这样做的）
    ws_bak = os.environ.pop('WORLD_SIZE', None)

    # 设备列表（父进程视角的“物理 id”）
    if devices is None:
        devices = list(range(torch.cuda.device_count()))
    assert len(devices) >= 1, "No CUDA devices found."

    # 支持每卡多副本：展开物理 id 列表
    physical_device_list = []
    for d in devices:
        physical_device_list.extend([d] * max(1, int(replicas_per_device)))

    # 进程池与队列
    ctx = mp.get_context("spawn")
    inqs, procs = [], []
    outq = ctx.Queue(maxsize=6000)
    # outq = ctx.SimpleQueue()

    fps = None
    nframe = None

    if dataset.fps > 0:
        fps = dataset.fps
    elif dataset.nframe > 0:
        nframe = dataset.nframe

    kwargs['fps'] = fps
    kwargs['nframe'] = nframe

    # 启动子进程
    for phys_id in physical_device_list:
        iq = ctx.Queue(maxsize=6000)
        # iq = ctx.SimpleQueue()
        p = ctx.Process(
            target=_video_worker_loop,
            args=(phys_id, iq, outq, model_name, kwargs),
            daemon=True
        )
        p.start()
        inqs.append(iq)
        procs.append(p)

    # 分发任务（父进程构造输入，严格保留 idx）
    rr = 0
    want = 0
    dispatched = []

    qwen_video_llm_model = [
        # "Qwen2.5-VL-3B-Instruct_DDP",
        # "Qwen2.5-VL-7B-Instruct_DDP",

        # "Qwen3-VL-2B-Instruct_DDP",
        # "Qwen3-VL-4B-Instruct_DDP",
        # "Qwen3-VL-8B-Instruct_DDP",

        "SpaceR-SFT-7B_qwen25_DDP",

        "VST-3B-SFT_DDP",
        "VST-7B-SFT_DDP",

        # "Qwen2.5-VL-3B-Instruct",
        # "Qwen2.5-VL-7B-Instruct",

        # "Qwen3-VL-2B-Instruct",
        # "Qwen3-VL-4B-Instruct",
        # "Qwen3-VL-8B-Instruct",

    ]


    for i, idx in tqdm(enumerate(sample_indices_subrem), total=len(sample_indices_subrem),
                       desc=f'Infer {model_name}/{dataset_name}, Rank {rank}/{world_size}'):
        if idx in res:
            continue

        # —— 与串行路径相同的“视频参数对齐” —— #
        # 注意：这些参数作用在“父进程的 model”上，但我们是多进程构造模型；
        # 对于多数模型，nframe/fps 是“构造输入”层面的要求，仍以构造 struct 为准。
        dname = dataset_name
        if 'SUB_DATASET' in dataset.data.iloc[sample_map[idx]]:
            dname = dataset.data.iloc[sample_map[idx]]['SUB_DATASET']

        if hasattr(model_name, 'use_custom_prompt') and False:
            # model_name 是字符串，不能调实例方法；保持 dataset 侧统一构造
            pass

        # 这里直接用 dataset 侧构造（和你原逻辑一致）
        video_llm = False if ('megabench' in dataset_name.lower() and 'llava_onevision' in model_name) else getattr(model, 'VIDEO_LLM', False)

        video_llm = True if model_name in qwen_video_llm_model else video_llm
        # video_llm = False

        # print(f"video llm : {video_llm}, model_name.lower(): {dataset_name.lower()}")

        struct = dataset.build_prompt(
            sample_map[idx], video_llm=video_llm
        )

        # inqs[rr % len(inqs)].put((idx, struct, dname))

        q = inqs[rr % len(inqs)]
        while True:
            try:
                q.put((idx, struct, dname), timeout=1.0)
                break
            except Full:
                # 心跳：若所有进程都不在了，就报错退出
                if sum(p.is_alive() for p in procs) == 0:
                    raise RuntimeError("All video workers exited while dispatching tasks")

        rr += 1
        want += 1
        dispatched.append(idx)

    # 关闭输入
    for iq in inqs:
        iq.put(None)

    # 收集 + 每 20 条落盘
    completed = 0

    import queue as _q
    import time

    time.sleep(60)

    while completed < want:
        try:
            idx, response = outq.get(timeout=30.0)
        except Empty:
            alive = sum(p.is_alive() for p in procs)
            logger.info(f"[collector] waiting... completed={completed}/{want}, alive={alive}")
            if alive == 0:
                logger.warning(f"All video workers exited early: completed={completed}, want={want}, remain save as Failed.")

                pending = [i for i in dispatched if i not in res]

                if pending:
                    for i in pending:
                        # 若你的 response 需要携带更多上下文，可在此构造
                        res[i] = FAIL_MSGS[3]
                    dump(res, out_file)
                    completed += len(pending)
                    break

            continue
        res[idx] = response
        # if verbose:
        logger.info(f"queue get idx: {idx} {response}")

        completed += 1
        if (completed % 20) == 0:
            dump(res, out_file)
            logger.info(f"📀 checkpoint saved at {completed}/{want}")

    # 最终落盘 & 恢复 WORLD_SIZE
    dump(res, out_file)
    if ws_bak:
        os.environ['WORLD_SIZE'] = ws_bak

    # 回收子进程
    for p in procs:
        p.join(timeout=0.2)

    return model
