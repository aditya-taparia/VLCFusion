# VLCFusion

This project utilizes two datasets for training and evaluation: the **ATR Dataset** and the **Waymo Open Dataset**.

## ATR Dataset 📝
The ATR dataset, specifically the `cegr` (MWIR/Infrared) and `i1co` (Visible) video collections, is crucial for evaluating VLCFusion's multi-modal object detection capabilities.

### Preparation
The following pipeline of Python scripts is provided to process the raw ATR data into a structured format suitable for training and evaluating VLCFusion. These scripts automate the parsing of proprietary annotation files, synchronization of Infrared and Visible video frames based on timestamps, extraction of images, and generation of detailed metadata files (`metadata.jsonl`) which include bounding box annotations.

**Prerequisites for ATR Processing:**
* You must acquire the ATR Dataset from the DSIAC website: [https://dsiac.dtic.mil/databases/atr-algorithm-development-image-database/](https://dsiac.dtic.mil/databases/atr-algorithm-development-image-database/).
* **Important Video Format Note:** The raw video files obtained from DSIAC may not be in `.avi` format. You will need to manually convert them into `.avi` files using an appropriate file reader or conversion utility provided with the dataset or other compatible tools before using them with these processing scripts.
* Ensure you have access to the corresponding annotation (`.agt` files) and the associated metric (`.bbox_met`) files for the video collections you download.
* Python 3.x and the libraries listed in the main `requirements.txt` of this repository.

**Expected Raw Data Structure:**
Before running the processing scripts, organize your downloaded and (video-converted) ATR data into a directory structure similar to the following (adjust paths as per your setup):

```
/path/to/your/ATR_Dataset/
├── i1co/                           # Visible spectrum data
│   ├── arf-video/                  # Contains i1coXXXX_YYYY.avi files (converted from raw)
│   └── agt/                        # Contains i1coXXXX_YYYY.agt files
├── cegr/                           # Mid-Wave Infrared (MWIR) data
│   ├── arf-video/                  # Contains cegrXXXX_YYYY.avi files (converted from raw)
│   └── agt/                        # Contains cegrXXXX_YYYY.agt files
└── Metric/                         # Contains .bbox_met files for precise bounding box info
```

**Processing Scripts:**
We recommend placing the following scripts and the `config_utils.py` file into a dedicated `data_processing/atr/` directory within your local `VLCFusion` repository clone for better organization.

**1. Configuration Utility (`ATR Experiment/ATR data preprocessing/config_utils.py`)**
* **Purpose:** This script is not executed directly but serves as a centralized module for shared constants (e.g., lists of scenarios and classes as defined in the VLCFusion paper, target type mappings) and utility functions (e.g., JSON reading, Julian date conversion, data serialization) that are imported by the other processing scripts.
* **Important:** Please review the `SCENARIOS`, `CLASSES_FILTER`, and `TGTTYPE_TO_ID` / `TGTID_TO_CATEGORIES` mappings within this file to ensure they precisely match the subset of the ATR dataset (e.g., 9 scenarios, 10 target classes) used for the experiments in the VLCFusion paper.

**2. AGT to JSON Conversion (`ATR Experiment/ATR data preprocessing/agt_to_json_parser.py`)**
* **Purpose:** Parses the proprietary `.agt` text files (containing frame-by-frame annotations for ATR videos) and converts them into a more accessible `.json` format.
* **Usage Example:** Run this script for both `cegr` (IR) and `i1co` (Visible) `.agt` files.
    ```bash
    # Process Infrared (cegr) .agt files
    python ATR Experiment/ATR data preprocessing/agt_to_json_parser.py \
        --agt_path "/path/to/your/ATR_Dataset/cegr/agt" \
        --save_path "/path/to/your/processed_atr_output/cegr/agt-json"

    # Process Visible (i1co) .agt files
    python ATR Experiment/ATR data preprocessing/agt_to_json_parser.py \
        --agt_path "/path/to/your/ATR_Dataset/i1co/agt" \
        --save_path "/path/to/your/processed_atr_output/i1co/agt-json"
    ```

**3. Initial DataFrame Preparation (`ATR Experiment/ATR data preprocessing/prepare_initial_dataframes.py`)**
* **Purpose:** Creates initial Pandas DataFrames containing detailed metadata. It filters video files based on scenarios/classes defined in `config_utils.py`, ensures that corresponding IR and Visible video tags exist, and extracts frame-level information from the AGT-JSON files.
* **Usage Example:**
    ```bash
    python ATR Experiment/ATR data preprocessing/prepare_initial_dataframes.py \
        --rgb_video_dir "/path/to/your/ATR_Dataset/i1co/arf-video" \
        --ir_video_dir "/path/to/your/ATR_Dataset/cegr/arf-video" \
        --rgb_agt_json_dir "/path/to/your/processed_atr_output/i1co/agt-json" \
        --ir_agt_json_dir "/path/to/your/processed_atr_output/cegr/agt-json" \
        --output_dir "/path/to/your/processed_atr_output/metadata_csvs"
    ```
* **Outputs:** This script generates `cegr_initial_metadata.csv` (for IR) and `i1co_initial_metadata.csv` (for Visible) in the specified output directory.

**4. IR-RGB Data Combination and Synchronization (`ATR Experiment/ATR data preprocessing/combine_and_synchronize.py`)**
* **Purpose:** Takes the initial metadata DataFrames and synchronizes IR and Visible frames. For each IR frame, it identifies the closest corresponding Visible frame based on timestamps, adhering to a 100ms tolerance window as specified in the VLCFusion paper.
* **Usage Example:**
    ```bash
    python ATR Experiment/ATR data preprocessing/combine_and_synchronize.py \
        --ir_metadata_csv "/path/to/your/processed_atr_output/metadata_csvs/cegr_initial_metadata.csv" \
        --rgb_metadata_csv "/path/to/your/processed_atr_output/metadata_csvs/i1co_initial_metadata.csv" \
        --output_csv "/path/to/your/processed_atr_output/metadata_csvs/combined_synchronized_data.csv" \
        --time_tolerance_ms 100
        # Optional: --ir_train_exclusion_jsonl "path/to/exclude_ir.jsonl"
        # Optional: --rgb_train_exclusion_jsonl "path/to/exclude_rgb.jsonl"
    ```
* **Output:** Produces `combined_synchronized_data.csv`, where each row represents a successfully synchronized IR-Visible frame pair.

**5. Final Image Dataset Construction (`ATR Experiment/ATR data preprocessing/build_image_datasets.py`)**
* **Purpose:** This script constructs the final image datasets ready for use with VLCFusion. It takes the synchronized frame data and splits it into training, validation, and test sets. The split is performed at the video tag level (e.g., all frames from `cegr02003_0001` go into the same set) using a configurable random seed for reproducibility. For each frame pair, it extracts images from the `.avi` files. Bounding boxes for IR images are derived from `.bbox_met` files, while bounding boxes for Visible images are interpolated based on the IR bounding boxes and target center coordinates from the AGT-JSON files. Finally, it generates `metadata.jsonl` annotation files for each data split.
* **Usage Example:**
    ```bash
    python ATR Experiment/ATR data preprocessing/build_image_datasets.py \
        --combined_csv "/path/to/your/processed_atr_output/metadata_csvs/combined_synchronized_data.csv" \
        --base_output_dir "/path/to/your/VLCFusion_ATR_dataset" \
        --ir_video_dir "/path/to/your/ATR_OSU_CTD_Dataset/cegr/arf-video" \
        --rgb_video_dir "/path/to/your/ATR_OSU_CTD_Dataset/i1co/arf-video" \
        --ir_agt_json_dir "/path/to/your/processed_atr_output/cegr/agt-json" \
        --rgb_agt_json_dir "/path/to/your/processed_atr_output/i1co/agt-json" \
        --metric_dir "/path/to/your/ATR_OSU_CTD_Dataset/Metric" \
        --train_ratio 0.7 \
        --val_ratio 0.15 \
        --random_state 42
    ```
* **Output Structure:** The script creates two main dataset directories (e.g., `mwir_vlc_dataset` and `visible_vlc_dataset`) within the `--base_output_dir`. Each of these contains `train/`, `val/`, and `test/` subdirectories, which in turn hold an `images/` folder (with `.png` image files) and a `metadata.jsonl` file. The `metadata.jsonl` contains entries for each image, including its path, dimensions, and a list of annotated objects with their bounding boxes (`[x,y,width,height]`), category names, category IDs, range, and aspect angle information.
    ```
    /path/to/your/VLCFusion_ATR_dataset/
    ├── mwir_vlc_dataset/
    │   ├── train/
    │   │   ├── images/
    │   │   └── metadata.jsonl
    │   ├── val/
    │   └── test/
    └── visible_vlc_dataset/
        ├── train/
        │   ├── images/
        │   └── metadata.jsonl
        ├── val/
        └── test/
    ```

### Training and Evaluation
**1. ATR Condition Generation (VLM-based Scene Attributes)**
If your model uses VLM-generated conditions for feature modulation (as described in the VLCFusion paper), run `create_conditions.py` first.

* **Purpose:** Uses a Vision Language Model (e.g., GPT-4o via OpenAI API) to analyze images from the processed ATR dataset and generate boolean answers to a predefined set of questions about scene conditions. These are saved as JSON files.
* **Location:** `ATR Experiment/create_conditions.py`
* **Usage Example:** (Run for train, validation, and test splits of one modality, e.g., MWIR, as conditions are typically per scene instance).
    ```bash
    # For training set conditions (using MWIR images as input)
    python "ATR Experiment/create_conditions.py" \
        "/path/to/your/VLCFusion_ATR_Formatted_Dataset/mwir_vlc_dataset/train" \
        --metadata_file_name "metadata.jsonl" \
        --questions_file "ATR Experiment/conditions/preprocess/refined_conditions.json" \
        --output_file "ATR Experiment/conditions/seen/vlm_train.json" \
        --api_key "YOUR_OPENAI_API_KEY" # Or ensure OPENAI_API_KEY env var is set
        # Add other args like --model, --max_workers as needed

    # Repeat for validation and test sets, adjusting input/output paths.
    # Example for validation:
    # python "ATR Experiment/create_conditions.py" "/path/to/your/VLCFusion_ATR_Formatted_Dataset/mwir_vlc_dataset/val" --output_file "ATR Experiment/conditions/seen/vlm_val.json" ...
    ```

**2. ATR Training**
Train the VLCFusion model on the prepared ATR dataset using `ensemble_trainer.py`.

* **Location:** `ATR Experiment/ensemble_trainer.py`
* **Purpose:** This script handles the training of the `MultimodalDetr` model (defined in `ATR Experiment/multimodal_detr.py`). It loads the processed IR and Visible datasets, corresponding VLM conditions, and uses Hugging Face `Trainer` for the training loop.
* **Usage Example:**
    ```bash
    python "ATR Experiment/ensemble_trainer.py" \
        --output_dir "ATR Experiment/outputs/VLCFusion_ATR_Run1" \
        --visible_dataset_dir "/path/to/your/VLCFusion_ATR_Formatted_Dataset/visible_vlc_dataset" \
        --ir_dataset_dir "/path/to/your/VLCFusion_ATR_Formatted_Dataset/mwir_vlc_dataset" \
        --train_conditions_file "ATR Experiment/conditions/seen/vlm_train.json" \
        --val_conditions_file "ATR Experiment/conditions/seen/vlm_val.json" \
        --test_conditions_file "ATR Experiment/conditions/seen/vlm_test.json" \
        --condition_indices_to_sample_str "16,13,1,11,15,19,18" \
        --model_dir_1 "path/to/pretrained_ir_component_checkpoint_if_any" \
        --model_dir_2 "path/to/pretrained_visible_component_checkpoint_if_any" \
        --base_model_name "facebook/detr-resnet-50" \
        --ensemble_method "CBAM_FiLM" \
        --image_size 480 \
        --num_classes 10 \
        --per_device_train_batch_size 8 \
        --gradient_accumulation_steps 2 \
        --learning_rate 5e-5 \
        --num_train_epochs 50 \
        --save_strategy "epoch" \
        --eval_strategy "epoch" \
        --load_best_model_at_end True \
        --metric_for_best_model "eval_map" \
        --fp16 True \
        --report_to "wandb" \
        --run_name "VLCFusion_ATR_CBAM_FiLM_Run1" \
        # Add other relevant ModelArguments, DataArguments, TrainingArguments as needed
    ```
* Ensure paths to pretrained components (`model_dir_1`, `model_dir_2`) are correct if you are initializing parts of `MultimodalDetr` from existing checkpoints. If training from scratch using the `base_model_name` for DETR backbones, these might point to the Hugging Face model identifier or be handled within `MultimodalDetr`.

**3. ATR Evaluation**
Evaluate your trained VLCFusion model using the provided Jupyter Notebook or a similar evaluation script.

* **Location:** `ATR Experiment/test_ensemble.ipynb`
* **Purpose:** To assess the trained model's performance on the ATR test set, calculating metrics like mAP.
* **Usage:**
    1.  Open and run the `ATR Experiment/test_ensemble.ipynb` notebook using Jupyter.
    2.  Within the notebook, configure the path to your trained VLCFusion model checkpoint (usually found in the `output_dir` specified during training, e.g., `ATR Experiment/outputs/VLCFusion_ATR_Run1/checkpoint-XXXXX`).
    3.  Set the paths to the preprocessed ATR test dataset (`mwir_vlc_dataset/test` and `visible_vlc_dataset/test`) and the VLM conditions for the test set (`ATR Experiment/conditions/seen/vlm_test.json`).
    4.  Execute the cells to load the model and data, perform inference, and compute evaluation metrics.


## Waymo Open Dataset 📝

The Waymo Open Dataset is a large-scale, high-resolution dataset for autonomous driving research, featuring LiDAR point clouds, camera imagery, and synchronized sensor data.

### Preparation

Preparing the Waymo Open Dataset for use involves several steps:

**A. Download Waymo Open Dataset:**
1.  Visit the [Waymo Open Dataset website](https://waymo.com/open/) and download the sensor data (TFRecord files containing LiDAR, camera, and labels) for the splits you intend to use (e.g., training, validation).
2.  Organize the downloaded TFRecord files. For example:
    ```
    /path/to/your/Waymo_Open_Dataset_Root/
    ├── training/      # Contains training_0000.tfrecord, ..., training_XXXX.tfrecord
    ├── validation/    # Contains validation_0000.tfrecord, ..., validation_XXXX.tfrecord
    ```

**B. Generate Waymo Dataset (`._infos_*.pkl`):**
MMDetection3D relies on intermediate annotation files (typically `.pkl` format) that store structured information about the dataset (metadata, file paths, annotations).

1. **Create GT Data:** Use the `mmdet_data_creation/lidar_dataset_creation.ipynb` script to generate ground truth data for training and evaluation.

**C. Create Ground Truth `.bin` Files for Evaluation (using `gt_bin_creator.py`):**
For official Waymo evaluation, ground truth annotations need to be in a specific `.bin` format.

* **Purpose:** The `gt_bin_creator.py` script processes Waymo TFRecord data (guided by an `_infos_*.pkl` file or by scanning TFRecords for specific variations like 'day') to generate these `gt.bin` files.
* **Location:** `gt_bin_creator.py` (at the root of your VLCFusion project).
* **Usage Example (for validation split, 'day' variation):**
    ```bash
    python gt_bin_creator.py \
        --ann_file "/path/to/your/Waymo_Open_Dataset_Root/infos/waymo_infos_val.pkl" \
        --data_root "/path/to/your/Waymo_Open_Dataset_Root/" \
        --split "validation" \
        --load_interval 1 \
        # The script currently hardcodes variation1='day' and output filename.
        # You might need to modify the script or its parameters for different variations (night, dawn_dusk)
        # or to control the output .bin filename explicitly.
    ```
* **Output:** A file like `gt_day_validation.bin` (name depends on `variation` and `split` in the script) in the current directory or a path specified within the script.

### Training and Evaluation

**1. Waymo Condition Generation:**
Similar to the ATR dataset, if your Waymo experiments in VLCFusion utilize VLM-generated scene conditions for feature modulation, you'll need to generate these conditions.

* **Purpose:** The `get_conditions.py` script (or `get_features.ipynb`) processes images from the Waymo dataset using a VLM to generate condition vectors.
* **Location:** `get_conditions.py` (at the root of your VLCFusion project).
* **Usage Example (adapt based on the refactored `get_conditions.py` arguments and your Waymo data structure):**
    ```bash
    # For training set conditions
    python get_conditions.py \
        "/path/to/your/Waymo_Open_Dataset_Root/" \
        "/path/to/your/Waymo_Open_Dataset_Root/infos/waymo_infos_train.pkl" \
        --questions_file "Waymo Experiment/conditions/preprocess/refined_conditions.json" \
        --output_jsonl_file "Waymo Experiment/conditions/seen/waymo_vlm_train.jsonl" \
        --api_key "YOUR_OPENAI_API_KEY" \
        # Ensure waymo_image_root_path points to the directory structure
        # from which 'img_path' in waymo_infos_train.pkl can be resolved.
        # E.g., if img_path is "training/image_0/XXXX.png", then waymo_image_root_path
        # would be "/path/to/your/Waymo_Open_Dataset_Root/"
    ```
    * Repeat for validation and test sets as needed, adjusting input/output file names. The `get_conditions.py` script expects paths to where images can be found relative to the `img_path` field in the Waymo infos file.

**2. Waymo Training:**
Training for the Waymo dataset uses the `trainer.py` script with an MMLab configuration file tailored for Waymo. Configuration for different fusion strategies (like `CBAM_FiLM`) and VLM conditions can be specified in the configs folder.

* **Location:** `trainer.py` (at the root of your VLCFusion project).
* **Purpose:** This script leverages MMEngine's `Runner` to train your `MultimodalDetr` model (or any other model defined in your MMLab config). The configuration file will specify the Waymo dataset (using the `_infos_*.pkl` files), data augmentation pipelines (potentially including GT-AUG using the database created earlier), model architecture, VLM conditions integration, and training schedules.
* **Usage Example:**
    ```bash
    python trainer.py \
        "configs/your_waymo_experiment_config.py" \
        --work-dir "Waymo Experiment/outputs/VLCFusion_Waymo_Run1" \
        # --resume auto # or path to a checkpoint
    ```
    * The `your_waymo_experiment_config.py` should be configured to:
        * Load the Waymo dataset using the `waymo_infos_train.pkl` and `waymo_infos_val.pkl` files.
        * Specify the path to the GT database if GT-AUG is used.
        * Incorporate VLM conditions if applicable (e.g., by modifying the data loading pipeline to read `waymo_vlm_train.jsonl`).
        * Define the `MultimodalDetr` architecture and its fusion strategies.

**3. Waymo Evaluation:**
For evaluation, use the `evaluation.py` script provided in the Waymo Experiment directory.