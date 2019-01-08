import argparse
from contextlib import ExitStack
import cv2
import json
import pandas as pd
from PIL import Image
import tqdm

import tensorflow as tf
from tensorpack import *
from tensorpack.dataflow import MapDataComponent
from tensorpack.utils.utils import get_tqdm_kwargs

from common import DataFromListOfDict
from config import config as cfg
from config import finalize_configs
from eval import DetectionResult, detect_one_image, print_coco_metrics
from train import ResNetC4Model, ResNetFPNModel

from coco import COCODetection, COCOMeta

def _offline_evaluate(pred_config, output_file):
    num_gpu = cfg.TRAIN.NUM_GPUS
    graph_funcs = MultiTowerOfflinePredictor(
        pred_config, list(range(num_gpu))).get_predictors()
    predictors = []
    dataflows = []
    for k in range(num_gpu):
        predictors.append(lambda img,
                          pred=graph_funcs[k]: detect_one_image(img, pred))
        dataflows.append(_get_eval_dataflow(shard=k, num_shards=num_gpu))
    if num_gpu > 1:
        all_results = _multithread_eval_coco(dataflows, predictors)
    else:
        all_results = _eval_coco(dataflows[0], predictors[0])

    with open(output_file + '.json', 'w+') as f:
        json.dump(all_results, f)
    print_coco_metrics(output_file + '.json')

    records = []
    for res in all_results:
        width, height = Image.open(res['image_path']).size
        # left, top, right, bottom = [float(item) for item in res['bbox']]
        x1, y1, w, h = [float(item) for item in res['bbox']]
        left, top, right, bottom = [x1, y1, x1 + w, y1 + h]
        records.append([res['image_path'], 0, width, height, left, top, right, bottom, res['score'], res['label']])

    records_df = pd.DataFrame.from_records(records, columns=['path', 'timestamp', 'width', 'height', 'left', 'top', 'right', 'bottom', 'score', 'class'])
    records_df.to_csv(output_file + '.csv', index=False, header=False)

def _get_eval_dataflow(shard=0, num_shards=1):
    """
    Args:
        shard, num_shards: to get subset of evaluation data
    """
    roidbs = COCODetection.load_many(cfg.DATA.BASEDIR, cfg.DATA.VAL, add_gt=False)
    num_imgs = len(roidbs)
    img_per_shard = num_imgs // num_shards
    img_range = (shard * img_per_shard, (shard + 1) * img_per_shard if shard + 1 < num_shards else num_imgs)

    # no filter for training
    ds = DataFromListOfDict(roidbs[img_range[0]: img_range[1]], ['file_name', 'id', 'file_name'])

    def f(fname):
        im = cv2.imread(fname, cv2.IMREAD_COLOR)
        assert im is not None, fname
        return im
    ds = MapDataComponent(ds, f, 0)
    # Evaluation itself may be multi-threaded, therefore don't add prefetch here.
    return ds

# Note(xin): overwrite _eval_coco
def _eval_coco(df, detect_func, tqdm_bar=None):
    df.reset_state()
    all_results = []
    # tqdm is not quite thread-safe: https://github.com/tqdm/tqdm/issues/323
    with ExitStack() as stack:
        if tqdm_bar is None:
            tqdm_bar = stack.enter_context(
                tqdm.tqdm(total=df.size(), **get_tqdm_kwargs()))
        for img, img_id, img_path in df:
            results = detect_func(img)
            for r in results:
                box = r.box
                cat_id = COCOMeta.class_id_to_category_id[r.class_id]
                box[2] -= box[0]
                box[3] -= box[1]

                res = {
                    'image_path': img_path,
                    'image_id': img_id,
                    'category_id': cat_id,
                    'label': cfg.DATA.CLASS_NAMES[cat_id],
                    'bbox': list(map(lambda x: round(float(x), 3), box)),
                    'score': round(float(r.score), 4),
                }

                # also append segmentation to results
                if r.mask is not None:
                    rle = cocomask.encode(
                        np.array(r.mask[:, :, None], order='F'))[0]
                    rle['counts'] = rle['counts'].decode('ascii')
                    res['segmentation'] = rle
                all_results.append(res)
            tqdm_bar.update(1)
    return all_results


def _multithread_eval_coco(dataflows, detect_funcs):
    num_worker = len(dataflows)
    assert len(dataflows) == len(detect_funcs)
    with ThreadPoolExecutor(max_workers=num_worker, thread_name_prefix='EvalWorker') as executor, \
            tqdm.tqdm(total=sum([df.size() for df in dataflows])) as pbar:
        futures = []
        for dataflow, pred in zip(dataflows, detect_funcs):
            futures.append(executor.submit(_eval_coco, dataflow, pred, pbar))
        all_results = list(itertools.chain(*[fut.result() for fut in futures]))
        return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', required=True, help='load a model for evaluation or training. Can overwrite BACKBONE.WEIGHTS')
    # parser.add_argument('--logdir', help='log directory', default='train_log/maskrcnn')
    # parser.add_argument('--visualize', action='store_true', help='visualize intermediate results')
    parser.add_argument('--evaluate', default='xin', help="Run evaluation on COCO. "
                                           "This argument is the name of the output csv/json evaluation file")
    # parser.add_argument('--predict', help="Run prediction on a given image. "
                                          # "This argument is the path to the input image file")
    parser.add_argument('--config', help="A list of KEY=VALUE to overwrite those defined in config.py",
                        nargs='+')

    args = parser.parse_args()
    if args.config:
        cfg.update_args(args.config)

    MODEL = ResNetFPNModel() if cfg.MODE_FPN else ResNetC4Model()

    finalize_configs(is_training=False)

    predcfg = PredictConfig(
        model=MODEL,
        session_init=get_model_loader(args.load),
        input_names=MODEL.get_inference_tensor_names()[0],
        output_names=MODEL.get_inference_tensor_names()[1])

    _offline_evaluate(predcfg, args.evaluate)
