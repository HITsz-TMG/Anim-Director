# EvalCrafter: Benchmarking and Evaluating Large Video Generation Models (CVPR 2024) 🎥📊

#### [Project Page](http://evalcrafter.github.io) &nbsp; · &nbsp; [Leaderboard](https://huggingface.co/spaces/AILab-CVC/EvalCrafter) &nbsp; · &nbsp;  [Gallery](https://huggingface.co/spaces/RaphaelLiu/EvalCrafter-T2V-Gallery) &nbsp; · &nbsp; [Paper@ArXiv](https://arxiv.org/abs/2310.11440)  &nbsp; · &nbsp;  [EvalBoard](https://github.com/evalcrafter/EvalBoard)
#### [ECTV Dataset](https://huggingface.co/datasets/RaphaelLiu/EvalCrafter_T2V_Dataset) &nbsp; · &nbsp; [Human Feedback](https://drive.google.com/file/d/1hbSoGoqD6DO2yV2R6wn8NKvK0nUxMDqv/view?usp=sharing) &nbsp; · &nbsp;  [Prompt List](https://github.com/evalcrafter/EvalCrafter/blob/master/prompt700.txt) 


<div align="center">
<img src="https://github.com/evalcrafter/evalcrafter/assets/4397546/818c9b0d-35ac-4edf-aafc-ae17e92c6da5" width="250"/>
</div>

Welcome to EvalCrafter, a comprehensive evaluation toolkit for AI-generated videos. Our innovative framework assesses generative models across visual, content, and motion qualities using 17 objective metrics and subjective user opinions, providing a reliable ranking for state-of-the-art text-to-video generation models. Dive into the world of unified and user-centric evaluation with EvalCrafter! 🚀🌍📊

#### 🔥 ~~2023/10/22: Release prompt list at [Prompt List](https://github.com/evalcrafter/EvalCrafter/blob/master/prompt700.txt)! You can generate the resulting video and send it to vinthony@gmail.com for evaluation!~~

#### 🔥 2024/01/10: Code and docker released!

#### 🔥 2024/01/24: EvalCrafter Text-to-Video (ECTV) dataset released, around 10000 AI-generated videos in total, take a look at [here](https://huggingface.co/datasets/RaphaelLiu/EvalCrafter_T2V_Dataset)!

#### 🔥 2024/03/07:  Dive into the exciting world of AI-Created Videos at our [Gallery](https://huggingface.co/spaces/RaphaelLiu/EvalCrafter-T2V-Gallery) 🌌✨!

#### 🔥 2024/04/07:  Release the code of [EvalBoard](https://github.com/evalcrafter/EvalBoard) 🌌✨ for video comparison !

#### 🔥 2024/04/11:  Release the  [Human Feedback](https://drive.google.com/file/d/1hbSoGoqD6DO2yV2R6wn8NKvK0nUxMDqv/view?usp=sharing) data, _**8.6k feedback**_, cover _**5 aspects**_ like visual quality and t2v alignment! ！🚀🔥

#### 🔥 2024/08/01:  Environment setup using Conda is Supported！🛫🤘

![AI-Created Video Gallery](https://github.com/evalcrafter/EvalCrafter/blob/master/Gallery.gif)


#### 🔆 Join our Discord to enjoy free text-to-video generation and more: [![Discord](https://dcbadge.vercel.app/api/server/rrayYqZ4tf?style=flat)](https://discord.gg/rrayYqZ4tf)

#### 🔆 Watch our project for more details and findings.


## Installation 💻

Clone the repository:

   ```bash
   git clone https://github.com/evalcrafter/EvalCrafter
   cd EvalCrafter
   ```

## Data Preparation 📚

Generate videos of your model using the 700 prompts provided in `prompt700.txt` or `./prompts` and organize them in the following structure:

```
/EvalCrafter/videos
├── 0000.mp4
├── 0001.mp4
├── 0002.mp4
├── 0003.mp4
├── 0004.mp4
...
└── 0699.mp4
```

## Pretrained Models 🧠
Please download all checkpoints using 
```
cd checkpoints
bash download.sh
```

Alternatively, you can follow `./checkpoints/README.md` to download pretrained models for specific metrics.

Note: Please organize the pretrained models in this structure: 
```
/EvalCrafter/checkpoints/
├── bert-base-uncased
├── blip2-opt-2.7b
├── ckpt
├── clip-vit-base-patch32
├── Dover
├── FlowNet2_checkpoint.pth.tar
├── pt_inception-2015-12-05-6726825d.pth
├── RAFT
├── stable-diffusion-xl-base-1.0
├── tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb_20220906-2692d16c.pth
├── vgg_face_weights.h5
└── VideoMAE
```

<!-- Alternatively, download all the pretrained models from [Huggingface](https://huggingface.co/RaphaelLiu/EvalCrafter-Models) -->


## Setup 🛠️ 

### Download Docker Image  🐳

   ```
   docker pull bruceliu1/evalcrafter:v1
   ```

### Or, create a conda environment  🎮

   ```
   conda env create -f environment.yml
   conda activate EvalCrafter

   cd ./metrics/RAFT/networks/resample2d_package
   python setup.py install --user
   ```
Note that if you encounter this problem  `RuntimeError: GET was unable to find an engine to execute this computation` when running the evaluation code, you may update Cuda to version Cuda 11.7 (Compatibility issue).



## Usage 🚀

### Running the Whole Pipeline 

1. Run with command line:

   ```
   docker run --runtime=nvidia -it --shm-size "15G" -v $EC_path:$EC_path bruceliu1/evalcrafter:v1 \
      bash -c "source /opt/conda/bin/activate EvalCrafter \
         && bash $bash_file $EC_path $EC_path/videos"
   ```

   🔁 Please replace `$EC_path`, `$bash_file`, and `$dir_videos` with your local path to `EvalCrafter`, `EvalCrafter/start.sh`, and `EvalCrafter/videos`, respectively. 

Alternatively, you can:

2. Enter the Docker container and run:

   ```
   docker run --runtime=nvidia -it --shm-size "15G" -v $EC_path:$EC_path bruceliu1/evalcrafter:v1 bash
   cd $EC_path
   bash start.sh $EC_path $dir_videos
   ```

3. If you use conda, you can directly run the evaluation code by:
   ```
   bash start.sh $EC_path $dir_videos
   ```

### Running a Single Metric

1. 🔧 To test a specific metric, pick out the code for the metric in `start.sh`. For example, to test the Celebrity ID Score:

   ```
   docker run --runtime=nvidia -it --shm-size "15G" -v $EC_path:$EC_path bruceliu1/evalcrafter:v1 bash
   cd $EC_path
   cd /metrics/deepface
   python3 celebrity_id_score.py --dir_videos $dir_videos
   ```

2. If you use conda, you can directly run the evaluation code by:
   ```
   cd $EC_path
   cd /metrics/deepface
   python3 celebrity_id_score.py --dir_videos $dir_videos
   ```

<!-- ### Run with Conda 🍃

1. Create the Conda environment and install dependencies:

   ```
   conda env create -f EvalCrafter_env.yml
   conda activate EvalCrafter
   cd $EC_path$
   ``` -->

## EvalCrafter Gallery🌌
Please ref to [EvalBoard](https://github.com/evalcrafter/EvalBoard) to use it.
1. Clone the repository:

   ```
   git clone https://github.com/evalcrafter/EvalBoard
   ```

2. Navigate to the project directory:

   ```
   cd EvalBoard
   ```

3. Install the required dependencies:

   ```
   pip install -r requirements.txt
   ```

4. Start EvalBoard:
      ```
      python app.py
      ```

## Acknowledgements 🙏

This work is based on the following open-source repositories:

- [deepface](https://github.com/serengil/deepface)
- [DOVER](https://github.com/teowu/DOVER-Dev)
- [mmaction2](https://github.com/open-mmlab/mmaction2)
- [CLIP](https://github.com/openai/CLIP)
- [RAFT](https://github.com/princeton-vl/RAFT)
- [pytorch-gan-metrics](https://github.com/w86763777/pytorch-gan-metrics)
- [SDXL](https://github.com/Stability-AI/generative-models)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [SAM-Track](https://github.com/z-x-yang/Segment-and-Track-Anything)
- [BILIP2](https://github.com/salesforce/LAVIS/tree/main/projects/blip2)
- [HRS-Bench](https://github.com/eslambakr/HRS_benchmark)
- [fast_blind_video_consistency](https://github.com/phoenix104104/fast_blind_video_consistency)

## Citation
If you find this repository helpful, please consider citing it in your research:

   ```
   @article{liu2023evalcrafter,
  title={Evalcrafter: Benchmarking and evaluating large video generation models},
  author={Liu, Yaofang and Cun, Xiaodong and Liu, Xuebo and Wang, Xintao and Zhang, Yong and Chen, Haoxin and Liu, Yang and Zeng, Tieyong and Chan, Raymond and Shan, Ying},
  journal={arXiv preprint arXiv:2310.11440},
  year={2023}
   }
   ```


## Know More About Video Generation at:

- [VideoCrafter1: Open Diffusion Models for High-Quality Video Generation](https://github.com/AILab-CVC/VideoCrafter)
- [ScaleCrafter: Tuning-free Higher-Resolution Visual Generation with Diffusion Models](https://github.com/YingqingHe/ScaleCrafter)
- [TaleCrafter: Interactive Story Visualization with Multiple Characters](https://github.com/AILab-CVC/TaleCrafter)

