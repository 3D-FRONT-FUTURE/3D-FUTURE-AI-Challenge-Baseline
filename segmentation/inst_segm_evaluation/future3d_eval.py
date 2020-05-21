import logging
import os
from collections import OrderedDict
import pdb

def do_future3d_evaluation(
    coco_data,
    pred_file_path,
    iou_types=["segm",],
    expected_results=(),
    expected_results_sigma_tol=4,
):
    logger = logging.getLogger("future3d.evaluation")
    # inline
    prepare_for_coco_segmentation(coco_data.dataset)

    results = COCOResults(*iou_types)
    logger.info("Evaluating predictions")
    for iou_type in iou_types:
        res = evaluate_predictions_on_coco(
            coco_data, pred_file_path, iou_type
        )
        results.update(res)
    logger.info(results)
    check_expected_results(results, expected_results, expected_results_sigma_tol)
    return results

def prepare_for_coco_segmentation(dataset):
    if "annotations" in dataset.keys():
        for ann_i in range(len(dataset["annotations"])):
            dataset["annotations"][ann_i]["iscrowd"] = 0
    return dataset

def evaluate_predictions_on_coco(
    coco_gt, json_result_file, iou_type="bbox"
):
    import json

    #from pycocotools.coco import COCO
    #from pycocotools.cocoeval import COCOeval
    from coco import COCO
    from cocoeval import COCOeval

    coco_dt = coco_gt.loadEntireRes(str(json_result_file)) if json_result_file else COCO()

    # coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval


class COCOResults(object):
    METRICS = {
        # "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
        "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
    }

    def __init__(self, *iou_types):
        allowed_types = ("bbox", "segm")
        assert all(iou_type in allowed_types for iou_type in iou_types)
        results = OrderedDict()
        for iou_type in iou_types:
            results[iou_type] = OrderedDict(
                [(metric, -1) for metric in COCOResults.METRICS[iou_type]]
            )
        self.results = results

    def update(self, coco_eval):
        if coco_eval is None:
            return
        #from pycocotools.cocoeval import COCOeval
        from cocoeval import COCOeval

        assert isinstance(coco_eval, COCOeval)
        s = coco_eval.stats
        iou_type = coco_eval.params.iouType
        res = self.results[iou_type]
        metrics = COCOResults.METRICS[iou_type]
        for idx, metric in enumerate(metrics):
            res[metric] = s[idx]

    def __repr__(self):
        results = '\n'
        for task, metrics in self.results.items():
            results += 'Task: {}\n'.format(task)
            metric_names = metrics.keys()
            metric_vals = ['{:.4f}'.format(v) for v in metrics.values()]
            results += (', '.join(metric_names) + '\n')
            results += (', '.join(metric_vals) + '\n')
        return results


def check_expected_results(results, expected_results, sigma_tol):
    if not expected_results:
        return

    logger = logging.getLogger("future3d.evaluation")
    for task, metric, (mean, std) in expected_results:
        actual_val = results.results[task][metric]
        lo = mean - sigma_tol * std
        hi = mean + sigma_tol * std
        ok = (lo < actual_val) and (actual_val < hi)
        msg = (
            "{} > {} sanity check (actual vs. expected): "
            "{:.3f} vs. mean={:.4f}, std={:.4}, range=({:.4f}, {:.4f})"
        ).format(task, metric, actual_val, mean, std, lo, hi)
        if not ok:
            msg = "FAIL: " + msg
            logger.error(msg)
        else:
            msg = "PASS: " + msg
            logger.info(msg)
