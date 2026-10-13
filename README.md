# X<sup>2</sup>Localizer: Cross-grained Alignment for Progressive Cross-view Video Geo-localization

[![Homepage](https://img.shields.io/badge/Homepage-X2Localizer-14a362)](https://zichaozeng.github.io/X2Localizer/)
[![Paper](https://img.shields.io/badge/Paper-BMVC_Oral-blue)](https://arxiv.org/abs/2608.16658)

X²Localizer progressively localizes ground-level videos in aerial imagery through cross-grained alignment.
This repository is being released incrementally. Model code, frame-extraction code, dataset links, and checkpoints are available; training and evaluation code and long-range data instructions will follow.

## Release Status

- ✅ Preprint
- ✅ Model code
- ✅ Standard dataset links and ground-frame extraction instructions
- ✅ Checkpoints
- ⏳ Long-range dataset and preparation instructions
- ⏳ Training and evaluation code
- ⏳ Training and evaluation instructions

## Data Preparation

### Ground-level videos: BDD100K

Download the ground-level videos from [BDD100K](https://bdd-data.berkeley.edu/).
Frame extraction also requires the per-video GPS JSON files and the daytime split lists used by GAMa. Videos alone are insufficient: the script selects frames using the timestamps in those JSON files.

We revised the GAMa frame-extraction procedure to address issues in the provided script and follow the sampling procedure described in the GAMa paper. As a result, our constructed ground-frame dataset may differ from the versions used by GAMa and GAReT. Use the procedure below when preparing data for this repository.

### Aerial imagery and split files: GAMa

Obtain the aerial data and associated preparation resources from [GAMa](https://www.crcv.ucf.edu/data1/GAMa/).
Prepare the source video directories, GPS JSON directories, and split lists before extracting frames. The examples below use the following layout; all paths can be changed through command-line arguments:

```text
DATA_ROOT/
├── bdd100k/videos/100k/
│   ├── train/<video_id>.mov
│   └── val/<video_id>.mov
├── 100k/
│   ├── train/<video_id>.json
│   └── val/<video_id>.json
└── satellite/list/
    ├── train_day.list
    └── val_day_vid.list
```

Each non-empty line in a split list must begin with a video ID, optionally followed by the video extension. Additional columns are ignored. For each listed ID, the script reads the matching video and JSON file. The JSON must contain `startTime` and a non-empty `locations` array with `timestamp` values, all in milliseconds:

```json
{
  "startTime": 1000000,
  "locations": [
    {"timestamp": 1000000},
    {"timestamp": 1001000}
  ]
}
```

The numbers above are illustrative. Preserve the timestamps and location order from the source metadata. Frame `i` is extracted at `locations[i]["timestamp"] - startTime`; the script does not perform an additional uniform eight-frame sampling step.

### Extract ground frames

Install the frame-extraction dependency:

```bash
pip install opencv-python
```

From the repository root, extract the training frames:

```bash
python extract_frames.py \
  --video_dir /path/to/DATA_ROOT/bdd100k/videos/100k/train \
  --json_dir /path/to/DATA_ROOT/100k/train \
  --split_file /path/to/DATA_ROOT/satellite/list/train_day.list \
  --output_dir /path/to/DATA_ROOT/bdd_frames_216/train
```

Extract the validation frames:

```bash
python extract_frames.py \
  --video_dir /path/to/DATA_ROOT/bdd100k/videos/100k/val \
  --json_dir /path/to/DATA_ROOT/100k/val \
  --split_file /path/to/DATA_ROOT/satellite/list/val_day_vid.list \
  --output_dir /path/to/DATA_ROOT/bdd_frames_216/val
```

Frames are resized to **384 pixels wide × 216 pixels high** and saved as:

```text
bdd_frames_216/
├── train/<video_id>/00.jpg, 01.jpg, ...
└── val/<video_id>/00.jpg, 01.jpg, ...
```

Use `--video_ext .mp4` for MP4 sources, or `--width` and `--height` to change the output size. Run `python extract_frames.py --help` for all options.

Existing files with the same names are overwritten; other files are retained. Use an empty output directory when changing the source metadata. Missing inputs and decoding failures are reported, and the script continues with the remaining videos. It returns a nonzero exit status if any video or frame fails. Failed frames retain their original index gaps, so resolve reported failures before pairing the frames with aerial tiles by index.

## Checkpoints

Download the released weights from [the checkpoint folder](https://drive.google.com/drive/folders/1zJw0T6iUgvAaKJeAGaxa6RyHPrRUVZPt?usp=share_link).

For image-level ground-frame-to-aerial-tile pretraining, we use the pretrained model released by [GAReT](https://github.com/manupillai308/GAReT). This model also serves as the fine-grained retrieval model. Using the same image-level pretrained weights supports a fair comparison with GAReT, while our contribution focuses on cross-grained alignment during adaptation and progressive video localization.

Checkpoint loading instructions will be provided with the training and evaluation code.

## Citation

If you find our work useful, please cite:

```bibtex
@article{zeng2026x,
  title={X$^2$Localizer: Cross-grained Alignment for Progressive Cross-view Video Geo-localization}, 
  author={Zeng, Zichao and Fan, Weijia and Chen, Yufan and Goo, June Moh and Zheng, Junwei and Liu, Ruiping and Peng, Kunyu and Zhang, Jiaming and Stiefelhagen, Rainer and Boehm, Jan},
  journal={arXiv preprint arXiv:2608.16658},
  year={2026}
}
```

## License

This project is released under the [MIT License](LICENSE).
