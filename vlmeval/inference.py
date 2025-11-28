import torch
import torch.distributed as dist
from vlmeval.config import supported_VLM
from vlmeval.utils import track_progress_rich
from vlmeval.smp import *
from vlmeval.dataset import DATASET_TYPE

import os, warnings
import torch
import multiprocessing as mp
import queue as py_queue  # for Empty
import time

from tqdm import tqdm

FAIL_MSG = 'Failed to obtain answer via API.'
FAIL_MSGS = [
    'Failed to obtain answer via API.',
    '[ERROR]',
    'Hit max new token.',
    'Failed'
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
def infer_data_api(model, work_dir, model_name, dataset, index_set=None, api_nproc=4, ignore_failed=False):
    rank, world_size = get_rank_and_world_size()
    assert rank == 0 and world_size == 1
    dataset_name = dataset.dataset_name
    data = dataset.data
    if index_set is not None:
        data = data[data['index'].isin(index_set)]

    model = supported_VLM[model_name]() if isinstance(model, str) else model
    assert getattr(model, 'is_api', False)
    if hasattr(model, 'set_dump_image'):
        model.set_dump_image(dataset.dump_image)

    lt, indices = len(data), list(data['index'])

    structs = []
    for i in range(lt):
        item = data.iloc[i]
        if hasattr(model, 'use_custom_prompt') and model.use_custom_prompt(dataset_name):
            assert hasattr(model, 'build_prompt')
            struct = model.build_prompt(item, dataset=dataset_name)
        else:
            struct = dataset.build_prompt(item)
        structs.append(struct)

    out_file = f'{work_dir}/{model_name}_{dataset_name}_supp.pkl'

    # To reuse records in MMBench_V11
    if dataset_name in ['MMBench', 'MMBench_CN']:
        pred_format = get_pred_file_format()
        v11_pred = f'{work_dir}/{model_name}_{dataset_name}_V11.{pred_format}'
        if osp.exists(v11_pred):
            try:
                reuse_inds = load('http://opencompass.openxlab.space/utils/mmb_reuse.pkl')
                data = load(v11_pred)
                ans_map = {x: y for x, y in zip(data['index'], data['prediction']) if x in reuse_inds}
                dump(ans_map, out_file)
            except Exception as err:
                print(type(err), err)

    res = {}
    if osp.exists(out_file):
        res = load(out_file)
        if ignore_failed:
            res = {k: v for k, v in res.items()
                    if not any(msg in str(v) for msg in FAIL_MSGS)}

    structs = [s for i, s in zip(indices, structs) if i not in res]
    indices = [i for i in indices if i not in res]

    gen_func = model.generate
    structs = [dict(message=struct, dataset=dataset_name) for struct in structs]

    if len(structs):
        track_progress_rich(gen_func, structs, nproc=api_nproc, chunksize=api_nproc, save=out_file, keys=indices)

    res = load(out_file)
    if index_set is not None:
        res = {k: v for k, v in res.items() if k in index_set}
    os.remove(out_file)
    return res


def infer_data(model, model_name, work_dir, dataset, out_file, verbose=False, api_nproc=4, use_vllm=False):
    dataset_name = dataset.dataset_name
    prev_file = f'{work_dir}/{model_name}_{dataset_name}_PREV.pkl'
    res = load(prev_file) if osp.exists(prev_file) else {}
    if osp.exists(out_file):
        res.update(load(out_file))

    rank, world_size = get_rank_and_world_size()
    sheet_indices = list(range(rank, len(dataset), world_size))
    lt = len(sheet_indices)
    data = dataset.data.iloc[sheet_indices]
    data_indices = [i for i in data['index']]

    # If finished, will exit without building the model
    all_finished = True
    for i in range(lt):
        idx = data.iloc[i]['index']
        if idx not in res:
            all_finished = False
    if all_finished:
        res = {k: res[k] for k in data_indices}
        dump(res, out_file)
        return model

    # Data need to be inferred
    data = data[~data['index'].isin(res)]
    lt = len(data)

    kwargs = {}
    if model_name is not None and (
        'Llama-4' in model_name
        or 'Qwen2-VL' in model_name
        or 'Qwen2.5-VL' in model_name
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
        lt, indices = len(data), list(data['index'])
        supp = infer_data_api(
            model=model,
            work_dir=work_dir,
            model_name=model_name,
            dataset=dataset,
            index_set=set(indices),
            api_nproc=api_nproc)
        for idx in indices:
            assert idx in supp
        res.update(supp)
        res = {k: res[k] for k in data_indices}
        dump(res, out_file)
        return model
    else:
        model.set_dump_image(dataset.dump_image)

    for i in tqdm(range(lt), desc=f'Infer {model_name}/{dataset_name}, Rank {rank}/{world_size}'):
        idx = data.iloc[i]['index']
        if idx in res:
            continue

        # print("hasattr:", hasattr(model, "use_custom_prompt"))
        # print("dataset_name:", dataset_name)
        # print("dataset_type:", DATASET_TYPE(dataset_name, default=None))
        # print("use_custom_prompt:", model.use_custom_prompt(dataset_name))

        if hasattr(model, 'use_custom_prompt') and model.use_custom_prompt(dataset_name):

            print(111)

            struct = model.build_prompt(data.iloc[i], dataset=dataset_name)
        else:
            print(222)

            struct = dataset.build_prompt(data.iloc[i])

        # If `SKIP_ERR` flag is set, the model will skip the generation if error is encountered
        if os.environ.get('SKIP_ERR', False) == '1':
            FAIL_MSG = 'Failed to obtain answer'
            try:
                response = model.generate(message=struct, dataset=dataset_name)
            except RuntimeError as err:
                torch.cuda.synchronize()
                warnings.warn(f'{type(err)} {str(err)}')
                response = f'{FAIL_MSG}: {type(err)} {str(err)}'
        else:
            response = model.generate(message=struct, dataset=dataset_name)
        torch.cuda.empty_cache()

        if verbose:
            print(response, flush=True)

        res[idx] = response
        if (i + 1) % 10 == 0:
            dump(res, out_file)

    res = {k: res[k] for k in data_indices}
    dump(res, out_file)
    return model


# A wrapper for infer_data, do the pre & post processing
def infer_data_job(
    model, work_dir, model_name, dataset, verbose=False, api_nproc=4, ignore_failed=False, use_vllm=False
):
    rank, world_size = get_rank_and_world_size()
    dataset_name = dataset.dataset_name
    # 使用环境变量控制的文件格式
    result_file = get_pred_file_path(work_dir, model_name, dataset_name, use_env_format=True)

    prev_file = f'{work_dir}/{model_name}_{dataset_name}_PREV.pkl'
    if osp.exists(result_file):
        if rank == 0:
            data = load(result_file)
            # breakpoint()
            results = {k: v for k, v in zip(data['index'], data['prediction'])}
            if not ignore_failed:
                # results = {k: v for k, v in results.items() if FAIL_MSG not in str(v)}

                results = {k: v for k, v in results.items()
                    if not any(msg in str(v) for msg in FAIL_MSGS)}

            dump(results, prev_file)
        if world_size > 1:
            dist.barrier()

    tmpl = osp.join(work_dir, '{}' + f'{world_size}_{dataset_name}.pkl')
    out_file = tmpl.format(rank)

    print(f"tmpl path : {out_file}")
    replicas_per_device = int(os.environ.get('replicas_per_device', 1))

    if 'ddp' in model_name.lower() or 'single' in model_name.lower():
        model = infer_data_unified(
            model=model_name,                # 字符串 key，来自 supported_VLM
            model_name=model_name,
            work_dir=work_dir,
            dataset=dataset,
            out_file=out_file,
            devices=list(range(torch.cuda.device_count())),
            replicas_per_device=replicas_per_device,           # 想要每卡放几个就写几
            worker_multiplier=4,             # 每副本 in-flight 任务个数系数，2~4 之间调
            verbose=verbose
        )
    else:
        model = infer_data(
            model=model, work_dir=work_dir, model_name=model_name, dataset=dataset,
            out_file=out_file, verbose=verbose, api_nproc=api_nproc, use_vllm=use_vllm)

    if world_size > 1:
        dist.barrier()

    if rank == 0:
        data_all = {}
        for i in range(world_size):
            data_all.update(load(tmpl.format(i)))

        data = dataset.data
        for x in data['index']:
            assert x in data_all
        if os.getenv('SPLIT_THINK', False):
            prediction = [str(data_all[x]) for x in data['index']]

            def split_thinking(s):
                if '</think>' in s:
                    splits = s.split('</think>')
                    prediction = splits[-1].strip()
                    if len(splits) == 2 and '<think>' in splits[0]:
                        thinking = splits[0].split('<think>')[1].strip()
                    else:
                        thinking = '</think>'.join(splits[:-1])
                        thinking += '</think>'
                        warnings.warn('Failed to parse thinking, multiple </think> tags or missing <think> tag.')
                else:
                    thinking = ''
                    prediction = s
                return (prediction, thinking)
            split_func = model.split_thinking if hasattr(model, 'split_thinking') else split_thinking
            print(f'Prediction format: {os.getenv("SPLIT_THINK")},splitting func: {split_func}')
            tups = [split_func(x) for x in prediction]
            data['prediction'] = [x[0] for x in tups]
            data['thinking'] = [x[1] for x in tups]
        else:
            data['prediction'] = [str(data_all[x]) for x in data['index']]
        if 'image' in data:
            data.pop('image')

        dump(data, result_file)
        for i in range(world_size):
            os.remove(tmpl.format(i))
    if world_size > 1:
        dist.barrier()
    return model



def make_registry_builder(model_name: str, extra_kwargs: dict):
    ctor = supported_VLM[model_name]
    def _builder(*, device_id=None, force_single_device=True, **kw):
        merged = {**extra_kwargs, **kw}
        return ctor(device_id=device_id, force_single_device=force_single_device, **merged)
    return _builder


def _worker_loop(device_id, replica_id, in_q, out_q, model_name: str, extra_kwargs: dict):
    import os, sys, warnings, traceback
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):
        os.environ.pop(k, None)

    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if pkg_root not in sys.path:
        sys.path.insert(0, pkg_root)

    import torch

    kw = dict(extra_kwargs or {})
    kw["force_single_device"] = True
    kw["device_id"] = device_id
    kw["replica_id"] = replica_id

    builder = make_registry_builder(model_name, {})

    model = None
    try:
        model = builder(**kw)
    except Exception as e:
        warnings.warn(f"[worker {device_id}-{replica_id}] model init failed: {type(e)} {e}")

    while True:
        item = in_q.get()

        if item is None:
            break

        idx, struct, dataset_name = item
        try:
            if model is None:
                resp = f'Failed: model_init_failed on worker {device_id}-{replica_id}'
            else:
                resp = model.generate(message=struct, dataset=dataset_name)
        except RuntimeError as err:
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
            warnings.warn(f'{type(err)} {str(err)}')
            resp = f'Failed: {type(err)} {str(err)}'
        except Exception as err:
            resp = f'Failed: {type(err)} {str(err)} | {traceback.format_exc(limit=1)}'

        out_q.put((idx, resp))
        logger.info(f"--- queue put response: {str(resp)[:1000]}")


def infer_data_unified(
    model, model_name, work_dir, dataset, out_file,
    verbose=False, api_nproc=4, use_vllm=False,
    devices=None, replicas_per_device=None, worker_multiplier=4
):
    dataset_name = dataset.dataset_name
    prev_file = f'{work_dir}/{model_name}_{dataset_name}_PREV.pkl'
    res = load(prev_file) if os.path.exists(prev_file) else {}
    if os.path.exists(out_file):
        res.update(load(out_file))

    rank, world_size = get_rank_and_world_size()
    sheet_indices = list(range(rank, len(dataset), world_size))
    data = dataset.data.iloc[sheet_indices]
    data_indices = [i for i in data['index']]

    # 已完成直接写回
    if all(idx in res for idx in data_indices):
        dump({k: res[k] for k in data_indices}, out_file)
        return model

    data = data[~data['index'].isin(res)]
    lt = len(data)
    if lt == 0:
        dump({k: res[k] for k in data_indices}, out_file)
        return model

    extra_kwargs = {}
    if model_name and any(k in model_name for k in ['Llama-4','Qwen2-VL','Qwen2.5-VL']):
        extra_kwargs['use_vllm'] = use_vllm

    # 规避 TP
    ws_bak = os.environ.pop('WORLD_SIZE', None)

    # ========== 仅在“模型名为字符串”的情况下，使用多进程多卡调度 ==========
    if isinstance(model, str):
        if devices is None:
            devices = list(range(torch.cuda.device_count()))
        assert len(devices) >= 1, "No CUDA devices found."

        # 队列与进程（每卡一个进程）
        replicas = replicas_per_device or 1  # ← 新增：每卡开多少份
        ctx = mp.get_context("spawn")
        inqs, procs = [], []
        outq = ctx.Queue(maxsize=int(len(data_indices) * 2))
        # outq = ctx.SimpleQueue()

        for did in devices:
            for r in range(replicas):
                # iq = ctx.SimpleQueue()
                iq = ctx.Queue(maxsize=int(len(data_indices)))
                p = ctx.Process(
                    target=_worker_loop,
                    args=(did, r, iq, outq, model_name, extra_kwargs)
                    # 不要 daemon=True
                )
                p.start()
                inqs.append(iq)
                procs.append(p)

        # ===== 先预构建 prompts =====
        tasks = []
        for i in tqdm(range(lt), desc="Building prompt..."):
            row = data.iloc[i]
            idx = row['index']
            if idx in res:
                continue
            struct = (dataset.build_prompt(row)
                    if hasattr(dataset, 'build_prompt') else row['question'])
            tasks.append((idx, struct, dataset_name))

        want = len(tasks)
        print(f"-----------------------Build prompt done! want={want} ---------------------------", flush=True)
        logger.info(f"----------------------------------Build prompt done! want={want}-----------------------, procs={len(procs)}")
        assert want > 0, "No tasks submitted to workers on this rank."

        # ===== 再分发任务 =====
        rr = 0
        for item in tasks:
            inqs[rr % len(inqs)].put(item)
            rr += 1

        print(f" ---------------- split mission -----------------------")

        # 关闭输入队列
        for iq in inqs:
            iq.put(None)

        print(f" ---------------- closed queue -----------------------")

        # 收集结果（带 timeout，避免永久卡死）
        completed = 0
        DUMP_EVERY = 10
        DUMP_EVERY_SEC = 180
        last_dump_t = time.time()

        time.sleep(60 * 2)

        # 收集结果
        try:
            while completed < want:
                try:
                    # logger.info(f"Start to get...")
                    idx, response = outq.get(timeout=30.0)
                    # logger.info(f"Success get!!!")
                except py_queue.Empty:
                    # 心跳：看看子进程还活着吗
                    alive = sum(p.is_alive() for p in procs)
                    logger.info(f"[collector] still waiting... completed={completed}/{want}, alive_procs={alive}")
                    # 如果全部死光了但还没凑够 want，说明前面哪里出问题了，跳出或 raise
                    if alive == 0:
                        raise RuntimeError(f"All workers exited early: completed={completed}, want={want}")
                    continue

                res[idx] = response
                logger.info(f"--- queue get response idx: {idx} {str(response)[:2000]}")  # 截断避免日志太长
                completed += 1

                if (completed % DUMP_EVERY == 0) or (time.time() - last_dump_t >= DUMP_EVERY_SEC):
                    dump({k: res[k] for k in data_indices if k in res}, out_file)
                    last_dump_t = time.time()
                    if verbose:
                        logger.info(f"📀 checkpoint saved at {completed}/{want}")

        except Exception as e:
            logger.exception(f"[collector] fatal: {e}")
            dump({k: res[k] for k in data_indices if k in res}, out_file)
            raise
        finally:
            dump({k: res[k] for k in data_indices if k in res}, out_file)


        dump({k: res[k] for k in data_indices if k in res}, out_file)

        if ws_bak:
            os.environ['WORLD_SIZE'] = ws_bak

        for p in procs:
            p.join(timeout=0.2)

        return model
