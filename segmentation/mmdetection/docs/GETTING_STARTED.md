# Getting Started

This page provides basic tutorials about the usage of MMDetection.
For installation instructions, please see [INSTALL.md](INSTALL.md).

## Inference with pretrained models

We provide testing scripts to evaluate a whole dataset (COCO, PASCAL VOC, Cityscapes, etc.),
and also some high-level apis for easier integration to other projects.

### Test a dataset

- [x] single GPU testing
- [x] multiple GPU testing
- [x] visualize detection results

You can use the following commands to test a dataset.

```shell
# single-gpu testing
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}] [--show]

# multi-gpu testing
./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}]
```

Optional arguments:
- `RESULT_FILE`: Filename of the output results in pickle format. If not specified, the results will not be saved to a file.
- `EVAL_METRICS`: Items to be evaluated on the results. Allowed values depend on the dataset, e.g., `bbox`, `segm` are available for FUTURE3D.
- `--show`: If specified, detection results will be plotted on the images and shown in a new window. It is only applicable to single GPU testing and used for debugging and visualization. Please make sure that GUI is available in your environment, otherwise you may encounter the error like `cannot connect to X server`.

If you would like to evaluate the dataset, do not specify `--show` at the same time.

Examples:

Assume that you have already downloaded the checkpoints to the directory `checkpoints/`.

1. Test Cascade Mask R-CNN and visualize the results. Press any key for the next image.

```shell
python tools/test.py configs/future3d_cascade_mask_rcnn_x101_64x4d_fpn_1x.py \
    checkpoints/future3d_cascade_mask_rcnn_x101_64x4d_fpn_1x-7487f640.pth \
    --show
```

2. Test Cascade Mask R-CNN on FUTURE3D (without saving the test results) and evaluate the mAP.

```shell
python tools/test.py configs/future3d_cascade_mask_rcnn_x101_64x4d_fpn_1x.py \
    checkpoints/future3d_cascade_mask_rcnn_x101_64x4d_fpn_1x-7487f640.pth \
    --eval segm
```

3. Test Cascade Mask R-CNN with 8 GPUs, and evaluate the bbox and mask AP.

```shell
./tools/dist_test.sh configs/future3d_cascade_mask_rcnn_x101_64x4d_fpn_1x.py \
    checkpoints/future3d_cascade_mask_rcnn_x101_64x4d_fpn_1x-7487f640.pth \
    8 --out results.pkl --eval segm
```

4. Test Cascade Mask R-CNN on FUTURE3D test with 8 GPUs, and generate the json file to be submit to the official evaluation server.

```shell
./tools/dist_test.sh configs/future3d_cascade_mask_rcnn_x101_64x4d_fpn_1x.py \
    checkpoints/future3d_cascade_mask_rcnn_x101_64x4d_fpn_1x-7487f640.pth \
    8 --format_only --options "jsonfile_prefix=./cascade_mask_rcnn_test_results"
```

You will get two json files `cascade_mask_rcnn_test_results.bbox.json` and `cascade_mask_rcnn_test_results.segm.json`.


## Train a model

MMDetection implements distributed training and non-distributed training,
which uses `MMDistributedDataParallel` and `MMDataParallel` respectively.

All outputs (log files and checkpoints) will be saved to the working directory,
which is specified by `work_dir` in the config file.

By default we evaluate the model on the validation set after each epoch, you can change the evaluation interval by adding the interval argument in the training config.
```python
evaluation = dict(interval=12)  # This evaluate the model per 12 epoch.
```

**\*Important\***: The default learning rate in config files is for 8 GPUs and 2 img/gpu (batch size = 8*2 = 16).
According to the [Linear Scaling Rule](https://arxiv.org/abs/1706.02677), you need to set the learning rate proportional to the batch size if you use different GPUs or images per GPU, e.g., lr=0.01 for 4 GPUs * 2 img/gpu and lr=0.08 for 16 GPUs * 4 img/gpu.

### Train with a single GPU

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

If you want to specify the working directory in the command, you can add an argument `--work_dir ${YOUR_WORK_DIR}`.

### Train with multiple GPUs

```shell
./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```

Optional arguments are:

- `--validate` (**strongly recommended**): Perform evaluation at every k (default value is 1, which can be modified like [this](https://github.com/open-mmlab/mmdetection/blob/master/configs/mask_rcnn_r50_fpn_1x.py#L174)) epochs during the training.
- `--work_dir ${WORK_DIR}`: Override the working directory specified in the config file.
- `--resume_from ${CHECKPOINT_FILE}`: Resume from a previous checkpoint file.

Difference between `resume_from` and `load_from`:
`resume_from` loads both the model weights and optimizer status, and the epoch is also inherited from the specified checkpoint. It is usually used for resuming the training process that is interrupted accidentally.
`load_from` only loads the model weights and the training epoch starts from 0. It is usually used for finetuning.

### Publish a model

Before you upload a model to AWS, you may want to
(1) convert model weights to CPU tensors, (2) delete the optimizer states and
(3) compute the hash of the checkpoint file and append the hash id to the filename.

```shell
python tools/publish_model.py ${INPUT_FILENAME} ${OUTPUT_FILENAME}
```

E.g.,

```shell
python tools/publish_model.py work_dirs/faster_rcnn/latest.pth faster_rcnn_r50_fpn_1x_20190801.pth
```

The final output filename will be `faster_rcnn_r50_fpn_1x_20190801-{hash id}.pth`.
