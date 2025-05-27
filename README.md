# VLCFusion

## Datasets

This project utilizes two main datasets for training and evaluation: the **ATR Dataset** and the **Waymo Open Dataset**.

### ATR Dataset Preparation 📝

The ATR dataset, specifically the `cegr` (MWIR/Infrared) and `i1co` (Visible) video collections, is crucial for evaluating VLCFusion's multi-modal object detection capabilities.

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

**0. Configuration Utility (`ATR Experiment/ATR data preprocessing/config_utils.py`)**
* **Purpose:** This script is not executed directly but serves as a centralized module for shared constants (e.g., lists of scenarios and classes as defined in the VLCFusion paper, target type mappings) and utility functions (e.g., JSON reading, Julian date conversion, data serialization) that are imported by the other processing scripts.
* **Important:** Please review the `SCENARIOS`, `CLASSES_FILTER`, and `TGTTYPE_TO_ID` / `TGTID_TO_CATEGORIES` mappings within this file to ensure they precisely match the subset of the ATR dataset (e.g., 9 scenarios, 10 target classes) used for the experiments in the VLCFusion paper.

**1. AGT to JSON Conversion (`ATR Experiment/ATR data preprocessing/agt_to_json_parser.py`)**
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

**2. Initial DataFrame Preparation (`ATR Experiment/ATR data preprocessing/prepare_initial_dataframes.py`)**
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

**3. IR-RGB Data Combination and Synchronization (`ATR Experiment/ATR data preprocessing/combine_and_synchronize.py`)**
* **Purpose:** Takes the initial metadata DataFrames (from S2) and synchronizes IR and Visible frames. For each IR frame, it identifies the closest corresponding Visible frame based on timestamps, adhering to a 100ms tolerance window as specified in the VLCFusion paper.
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

**4. Final Image Dataset Construction (`ATR Experiment/ATR data preprocessing/build_image_datasets.py`)**
* **Purpose:** This script constructs the final image datasets ready for use with VLCFusion. It takes the synchronized frame data (from S3) and splits it into training, validation, and test sets. The split is performed at the video tag level (e.g., all frames from `cegr02003_0001` go into the same set) using a configurable random seed for reproducibility. For each frame pair, it extracts images from the `.avi` files. Bounding boxes for IR images are derived from `.bbox_met` files, while bounding boxes for Visible images are interpolated based on the IR bounding boxes and target center coordinates from the AGT-JSON files. Finally, it generates `metadata.jsonl` annotation files for each data split.
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

### Waymo Open Dataset Preparation 📝
