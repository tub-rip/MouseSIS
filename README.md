# MouseSIS: Space-Time Instance Segmentation of Mice

[![Paper](https://img.shields.io/badge/arXiv-2409.03358-b31b1b.svg)](https://arxiv.org/abs/2409.03358)
[![Dataset](https://img.shields.io/badge/Dataset-GoogleDrive-4285F4.svg)](https://drive.google.com/drive/folders/1TQns9-WZw-n26FaUE3gqdAhGgrlRUzCp?usp=sharing)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

This is the official repository for [**MouseSIS: A Frames-and-Events Dataset for Space-Time Instance Segmentation of Mice**](https://arxiv.org/pdf/2409.03358), accepted at the **Workshop on Neuromorphic Vision** in conjunction with **ECCV 2024** by [Friedhelm Hamann](https://friedhelmhamann.github.io/), [Hanxiong Li](), [Paul Mieske](https://scholar.google.de/citations?user=wQPmm6kAAAAJ&hl=de), [Lars Lewejohann](https://www.vetmed.fu-berlin.de/einrichtungen/vph/we11/mitarbeitende/lewejohann_lars3/index.html) and [Guillermo Gallego](https://sites.google.com/view/guillermogallego).

👀 **Currently, the test set of this dataset is not available in preparation of a challenge (see split in the paper Tab. 2). You can still run our baseline method on the validation set and we'll soon provide access to an evaluation server. Stay tuned or in case of questions contact us!**

<p align="center">
  <img src="./image/visualization_seq12_0003.jpg" alt="MouseSIS Visualization" width="600"/>
</p>

## Table of Contents
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Evaluation](#evaluation)
    - [Evaluation of ModelMixSORT](#evaluation-of-modelmixsort)
    - [Evaluating Your Own Method](#evaluating-your-own-method)
- [Training](#training)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)
- [Additional Resources](#additional-resources)
- [License](#license)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/tub-rip/MouseSIS_dev.git
   cd MouseSIS_dev
   ```

2. Set up the environment:
   ```bash
   conda create --name MouseSIS python=3.8
   conda activate MouseSIS
   ```

3. Install PyTorch (choose a command compatible with your CUDA version from the [PyTorch website](https://pytorch.org/get-started/locally/)):
   ```bash
   conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
   ```

4. Install other dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Data Preparation
1. Create a folder for the original data

    ```bash
    cd <project-root>
    mkdir -p data/orig
    ```

2. [Download the data and annotation](https://drive.google.com/drive/folders/1amY4kuaZFWdpgHg4RfTrw9Qb-tKrM-8h) and save it in `<project-root>/data/orig`.
The `data/orig` folder should be organized as follows:


    ```txt
    data/orig/
    │
    ├── top/
    │   ├── seq_01.hdf5
    │   ├── seq_02.hdf5
    │   ├── ...
    │   └── seq_33.hdf5
    │
    ├── dataset_info.csv
    └── annotations.json
    ```

    - **`top/`**: This directory contains the frame and event data for the Mouse dataset captured from top view, stored as 33 individual `.hdf5` files, each containing approximately 20 seconds of data (around 600 frames), along with temporally aligned events.
    - **`dataset_info.csv`**: This CSV file contains metadata for each sequence, such as recording dates, providing additional context and details about the dataset.
    - **`annotations.json`**: The annotation file of top view follows a structure similar to MSCOCO's format in JSON, with some modifications.  The definition of json file is:

    ```txt
    {
        "info": {
            "description": "string",  
            "version": "string",  
            "date_created": "string"  
        },
        "videos": [
            {
                "id": "string", // video_id from "01" to "33"
                "width": 1280,  // Width of the video in pixels
                "height": 720,  // Height of the video in pixels
                "length": "int"  // Number of frames in the video
            }
        ],
        "annotations": [
            {
                "id": "int",  // Instance number for the mouse
                "video_id": "string",  // Corresponding video_id from "01" to "33"
                "category_id": 1,  // The category ID for the object
                "segmentations": [
                    {
                        "size": [720, 1280],  // Size of the segmentation mask
                        "counts": "RLE encoded string or null"  // RLE encoded segmentation or null
                    }
                ],
                "areas": [0.0],  // Area of the object (can be null)
                "bboxes": [[0.0, 0.0, 0.0, 0.0]],  // Bounding box for the object [x_min, y_min, width, height]
                "iscrowd": 0  
            }
        ],
        "categories": [
            {
                "id": 1,  
                "name": "mouse", 
                "supercategory": "animal"  
            }
        ]
    }

    ```

3. To evaluate the ModelMixSORT method or train the YOLO model used within it, you first need to convert the original dataset into YOLO format. 
For grayscale frames, Please run the following command.

    ```bash
    python3 scripts/preprocess.py --data_root data/orig --data_format frame
    ```
        
    For reconstructed e2vid images, Please run the following command.

    ```bash
    python3 scripts/preprocess.py --data_root data/orig --data_format e2vid 
    ```

    You can check the preprocessed data under `data/prepocessed`


## Evaluation

### Evaluation of ModelMixSORT

1. Download the [model weights](https://drive.google.com/drive/folders/1-P1HN4FZEy3ETn5rrQiMoDQx3378HLQW?usp=drive_link):
   ```bash
   mkdir models
   # Download yolo_e2vid.pt, yolo_frame.pt, and XMem.pth from the provided link
   # and place them in the models directory
   ```

2. Run inference:

    ```bash
    python3 scripts/inference.py --config configs/predict/combined.yaml
    ```
   We provide several config files in `configs/predict` for the different inference settings.
   The inference script produces per sequence predictions and visualizations.
All predictions are summarized in `final_results.json` . Each prediction follows this structure:

    ```txt
    [
        {
            "video_id": int, 
            "category_id": int, 
            "segmentations": [
                    {
                        "size": [int, int],
                        "counts": "RLE encoded string or null"
                    },
                    ...
                ],
            "score": float
        },
        ...
    ]
    ```

    The `final_results.json` file is also saved under the `src/TrackEval/data/trackers` folder for use with the TrackEval evaluation tool.

3. Evaluate the results (based on [TrackEval](https://github.com/JonathonLuiten/TrackEval)). The general command is:
   ```bash
   python src/TrackEval/run_mouse_eval.py --TRACKERS_TO_EVAL <tracker_name> --SPLIT_TO_EVAL <split_name>
   ```
   So, if you run inference with `configs/predict/combined.yaml`, the command looks like this:
   ```bash
   python src/TrackEval/run_mouse_eval.py --TRACKERS_TO_EVAL combined_0.1 --SPLIT_TO_EVAL test_wo17
   ```
   The provided result [in the paper](https://arxiv.org/pdf/2409.03358) is Tab. 4 line 3 (w/o 1 & 7).

### Evaluating Your Own Method

To evaluate your own method, please generate the output in JSON format, following the structure of `final_result.json` as described in the evaluation section. Place this JSON file in `src/TrackEval/data/trackers/<your_tracker_name>/test`, where **your_tracker_name** should be replaced with the name of your own tracker. Then, run the evaluation using the command:

```bash
python src/TrackEval/run_mouse_eval.py --TRACKERS_TO_EVAL <your_tracker_name> --SPLIT_TO_EVAL <split_name>
```

## Training

To train the yolo models used in ModelMixSORT using preprocessed grayscale mice data, please run:

    python scripts/train.py --config configs/train/frame.yaml

To train the yolo model using preprocessed e2vid mice data, please run:

    python scripts/train.py --config configs/train/e2vid.yaml

## Acknowledgements

We greatfully appreciate the following repositories and thank the authors for their excellent work:

- [XMem](https://github.com/hkchengrex/XMem)
- [TrackEval](https://github.com/JonathonLuiten/TrackEval)

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@inproceedings{hamann2024mousesis,
  title={MouseSIS: A Frames-and-Events Dataset for Space-Time Instance Segmentation of Mice},
  author={Hamann, Friedhelm and Li, Hanxiong and Mieske, Paul and Lewejohann, Lars and Gallego, Guillermo},
  booktitle={Proceedings of the European Conference on Computer Vision Workshops (ECCVW)},
  year={2024}
}
```

## Additional Resources

- [Recording Software (CoCapture)](https://github.com/tub-rip/CoCapture)
- [TU Berlin, RIP lab Homepage](https://sites.google.com/view/guillermogallego/research/event-based-vision)
- [Science Of Intelligence Homepage](https://www.scienceofintelligence.de/)
- [Event Camera Class at TU Berlin](https://sites.google.com/view/guillermogallego/teaching/event-based-robot-vision)
- [Event-based Vision Survey Paper](http://rpg.ifi.uzh.ch/docs/EventVisionSurvey.pdf)
- [List of Event Vision Resources](https://github.com/uzh-rpg/event-based_vision_resources)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.