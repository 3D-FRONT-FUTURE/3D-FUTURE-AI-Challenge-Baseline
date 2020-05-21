# Python evaluation script for IJCAI-PRICAI 2020 3D AI Challenge: Instance Segmentation
In the IJCAI-PRICAI 2020 3D AI Challenge, participants are required to label each foreground pixel with the appropriate object and instance. The evaluation metric is Mask Average Precision (mAP). This is a evaluation script for instance segmentation based on mAP metrics.

## Requirements:
* Numpy 
* Pycocotools

## Usage:
```
python evaluate.py /path/to/your/prediction /path/to/your/GT /path/to/your/output.json
```

## File Format:
### 1. Ground-truth Format
```
GT.json:
{
    categories: [
        {
            id: int, 
            category_name: str, 
            fine-grained category name: str, 
        },
        ...
    ]
    images: [
        {
            id: int, 
            width: int, 
            height: int, 
            file_name: str, 
        },
        ...
    ], 
    annotations: [
        {
            id: int, 
            image_id: int, 
            category_id: int, 
            segmentation: RLE or [polygon], 
            area: float, 
            bbox: [x,y,width,height], 
            model_id: str, 
            texture_id: str,
            pose: list
            fov: float,
            style: str,
            theme: str,
            material: str
        },
        ...
    ], 
}
```

### 2. Prediction Format
```
prediction.json
{
    annotations: [
        {
            image_id: int, 
            category_id: int, 
            segmentation: RLE or [polygon], 
            bbox: [x,y,width,height], 
            score: float
        },
        ...
    ], 
}
```
We have released our baseline results "**cascade_mask_rcnn_test_results.segm.json**". You can check the prediction format in this json file.