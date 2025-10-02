# Video, Image-to-Video, World & Motion Generation Evaluation Metrics

In this section we collect **metrics, paper links, and code repos** for evaluating video generation across:  
- **Text-to-Video**  
- **Image-to-Video**  
- **World Generation**

Each entry includes:  
- **Paper link**  
- **Axes / Scores**  
- **Method summary**  
- **Code / Repo** 
---

## Text to Video Generation Metrics

### DEVIL (Dynamics-Centric Protocol)
- **Paper**: *Evaluation of Text-to-Video Generation Models: A Dynamics Perspective* (Liao et al., 2024) [arXiv](https://arxiv.org/pdf/2407.01094)  
- **Axes**:  
  - Dynamics Range  
  - Dynamics Controllability  
  - Dynamics-based Quality  
  - Naturalness (via Gemini)  
- **Summary**: Dynamics-focused benchmark; computes inter-frame, inter-segment, full-video scores; adds Gemini “naturalness” score.  
- **Repo**: [![Stars](https://img.shields.io/github/stars/MingXiangL/DEVIL.svg?style=social&label=DEVIL)](https://github.com/MingXiangL/DEVIL)

---

### JEDi (JEPA Embedding Distance)  
- **Paper**: *Beyond FVD: Enhanced Evaluation Metrics for Video Generation* (Voynov et al., 2024) [arXiv](https://arxiv.org/abs/2410.05203)  
- **Axes**:  
  - Distributional fidelity (single score)  
- **Summary**: Uses JEPA embeddings and computes distances (Wasserstein, MMD) between real vs. generated distributions.  
- **Repo**: [![Stars](https://img.shields.io/github/stars/oooolga/JEDi.svg?style=social&label=JEDi)](https://github.com/oooolga/JEDi)

---

### FVMD (Fréchet Video Motion Distance)  
- **Paper**: *Fréchet Video Motion Distance: A Metric for Evaluating Motion Consistency in Videos* (Liu et al., 2024) [arXiv](https://arxiv.org/abs/2407.16124)  
- **Axes**:  
  - Motion consistency  
  - Temporal realism  
- **Summary**: Motion features (optical flow, trajectories) → embedding → Fréchet distance.  
- **Repo**: [![Stars](https://img.shields.io/github/stars/ljh0v0/FMD-frechet-motion-distance.svg?style=social&label=FVMD)](https://github.com/ljh0v0/FMD-frechet-motion-distance)

---

### VAMP (Visual + Physics Metric)  
- **Paper**: *What You See Is What Matters* (Wang et al., 2024) [arXiv](https://arxiv.org/abs/2411.13609)  
- **Axes**:  
  - Appearance Score (consistency: color, texture, shape)  
  - Motion Plausibility Score (physics-based realism)  
- **Summary**: Combines frame-level visual consistency with physical plausibility of motion.  
- **Repo**:

---

### FVD (Fréchet Video Distance)  
- **Paper**: *Towards Accurate Generative Models of Video* (Unterthiner et al., 2018) [arXiv](https://arxiv.org/abs/1812.01717)  
- **Axes**:  
  - Quality  
  - Temporal coherence  
  - Diversity  
- **Summary**: I3D embeddings → Fréchet distance between real vs. generated feature distributions.  
- **Repo**: [![Stars](https://img.shields.io/github/stars/bioinf-jku/TTUR.svg?style=social&label=TTUR)](https://github.com/bioinf-jku/TTUR)

---

### EvalCrafter  
- **Paper**: *EvalCrafter* (Liu et al., CVPR 2024) [Paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Liu_EvalCrafter_Benchmarking_and_Evaluating_Large_Video_Generation_Models_CVPR_2024_paper.pdf)  
- **Axes**:  
  - Visual Quality  
  - Content Accuracy  
  - Motion Quality  
  - Text Alignment  
- **Summary**: 700-prompt benchmark, 17 metrics aligned to human evals.  
- **Repo**: [![Stars](https://img.shields.io/github/stars/evalcrafter/EvalCrafter.svg?style=social&label=EvalCrafter)](https://github.com/evalcrafter/EvalCrafter)

---

### VBench  
- **Paper**: *VBench* (Huang et al., CVPR 2024) [arXiv](https://arxiv.org/abs/2403.12962)  
- **Axes**:  
  - 16 fine-grained dimensions (subject consistency, motion smoothness, flicker, etc.)  
- **Summary**: Disentangled benchmark across multiple axes; leaderboard provided.  
- **Repo**: [![Stars](https://img.shields.io/github/stars/Vchitect/VBench.svg?style=social&label=VBench)](https://github.com/Vchitect/VBench)

---

### FETV  
- **Paper**: *FETV* (Liu et al., NeurIPS 2023) [OpenReview](https://openreview.net/forum?id=yWpY5I3XyX)  
- **Axes**:  
  - Content alignment  
  - Attribute control  
  - Temporal aspects  
- **Summary**: Fine-grained categories; human eval + metric validation.  
- **Repo**: [![Stars](https://img.shields.io/github/stars/llyx97/FETV.svg?style=social&label=FETV)](https://github.com/llyx97/FETV)

---

### T2V-CompBench  
- **Paper**: *T2V-CompBench* (Sun et al., CVPR 2025) [Project](https://t2v-compbench.github.io/)  
- **Axes**:  
  - Object compositionality  
  - Attribute binding  
  - Spatial relations  
  - Actions / motions  
  - Numeracy  
- **Summary**: 7 compositional categories; evaluated via multimodal LLMs, detectors, trackers.  
- **Repo**: 

---

### VideoScore  
- **Paper**: *VideoScore* (Pan et al., EMNLP 2024) [arXiv](https://arxiv.org/abs/2405.12345)  
- **Axes**:  
  - Quality  
  - Motion  
  - Temporal   
  - Alignment  
- **Summary**: Regression-based metric trained on 37.6K human-labeled videos.  
- **Repo**: [![Stars](https://img.shields.io/github/stars/TIGER-AI-Lab/VideoScore.svg?style=social&label=VideoScore)](https://github.com/TIGER-AI-Lab/VideoScore)

---

### T2VScore  
- **Paper**: *T2VScore* (Wang et al., 2024) [arXiv](https://arxiv.org/abs/2401.07781)  
- **Axes**:  
  - Text-video alignment  
  - Visual quality  
- **Summary**: Mixture-of-experts scorer (text-video similarity + quality discriminator).  
- **Repo**: [![Stars](https://img.shields.io/github/stars/showlab/T2VScore.svg?style=social&label=T2VScore)](https://github.com/showlab/T2VScore)

---

### LOVE  
- **Paper**: *LOVE: Large-scale Open Video Evaluator* (Zhang et al., 2025) [arXiv](https://arxiv.org/abs/2505.12098)  
- **Axes**:  
  - Perceptual Quality  
  - Text-Video Correspondence  
  - Task-specific Accuracy  
- **Summary**: LMM evaluator trained on AIGVE-60K (58.5K videos, 120K MOS).  
- **Repo**: (Dataset only) [AIGVE-60K](https://huggingface.co/datasets/LOVE/AIGVE-60K)

---

### GRADEO  
- **Paper**: *GRADEO* (Sun et al., 2025) [arXiv](https://arxiv.org/abs/2503.16867)  
- **Axes**:  
  - Multi-step reasoning evaluation  
  - Explainable text alignment  
- **Summary**: GPT-based evaluator trained on GRADEO-Instruct dataset; produces step-by-step judgments.  
- **Repo**: 

---
### LiFT (Reward / Human-Feedback Alignmen
- **Paper / Source**: *LiFT: Leveraging Human Feedback for Text-to-Video Model Alignment* (Wang et al., 2024) [arXiv](https://arxiv.org/abs/2412.04814)
- **Axes / Scores Measured**:
  - Semantic Consistency  
  - Motion Smoothness  
  - Video Fidelity  
- **Method Summary**:
  1. They build a human annotated dataset **LiFT-HRA** (~10k videos), with ratings + reasoning.  
  2. They train a reward model **LiFT-Critic** to predict scores given (video, prompt).  
  3. They fine-tune a T2V model (e.g. CogVideoX) via reward-weighted likelihood to better align output to human preferences.
- **Code / Repo / Stars**: [![Stars](https://img.shields.io/github/stars/CodeGoat24/LiFT.svg?style=social&label=LiFT)](https://github.com/CodeGoat24/LiFT)  



---

##  Image-to-Video Generation Metrics

### AnimateBench  
- **Paper**: *AnimateBench* (Zhang et al., CVPR 2024) [arXiv](https://arxiv.org/abs/2309.08526)  
- **Axes**:  
  - Image alignment (input image vs. generated frames)  
  - Text alignment (motion text vs. frames)  
- **Summary**: CLIP similarity-based metrics for fidelity vs. motion.  
- **Repo**: (No official repo yet)

### AIGCBench (I2V)  
- **Paper**: *AIGCBench* (Fan et al., 2024) [arXi](https://arxiv.org/pdf/2401.01651)  
- **Axes**:  
  - Control alignment  
  - Motion effects  
  - Temporal consistency  
  - Video quality  
- **Summary**: 11 metrics across 4 axes it is validated against human judgment.  
- **Repo**: [![Stars](https://img.shields.io/github/stars/BenchCouncil/AIGCBench.svg?style=social&label=AIGCBench)](https://github.com/BenchCouncil/AIGCBench)


### VBench  
- **Paper**: *VBench* (Huang et al., CVPR 2024) [arXiv](https://arxiv.org/pdf/2411.13503)  
- **Axes**:  
   - Video–Image Consistency (how well generated frames match the input image)
   - Video–Text Consistency (if prompt is used in I2V)
   - Temporal / Motion Metrics (motion smoothness, temporal flickering, dynamic degree)
   - Imaging Quality (frame-level visual fidelity)
- **Summary**: VBench++ introduces a high-quality Image Suite (adaptive aspect ratio) to  evaluate I2V models, they reuse the 16 dimensions of VBench for video but adapts metrics to measure video–image alignment and consistency 
- **Repo**: [![Stars](https://img.shields.io/github/stars/Vchitect/VBench.svg?style=social&label=VBench)](https://github.com/Vchitect/VBench/tree/master/VBench-2.0)




---

## 🌍 World Generation Metrics

### WorldScore  
- **Paper**: *WorldScore* (Duan et al., ICCV 2025) [arXiv](https://arxiv.org/abs/2501.01234)  
- **Axes**:  
  - Controllability (camera, placement, alignment)  
  - Quality (3D / photometric / style / subjective)  
  - Dynamics (accuracy, magnitude, smoothness)  
- **Summary**: Unified world generation evaluation benchmark across 10 metrics grouped in 3  axes.  
- **Repo**: [![Stars](https://img.shields.io/github/stars/haoyi-duan/WorldScore.svg?style=social&label=WorldScore)](https://github.com/haoyi-duan/WorldScore)

---

## Leaderboards for Video Generation & Emerging Benchmarks

This section tracks current and emerging leaderboards for video generation and reward evaluation.

| Leaderboard | Description                                                                                                                                                                         | Links |
|---|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---|
| **VBench Leaderboard** | Leaderboard for text-to-video & long-video models, with various scores.                                                                                                             | ⭐ [GitHub](https://github.com/Vchitect/VBench) · 🤗 [HuggingFace Space](https://huggingface.co/spaces/Vchitect/VBench_Leaderboard) |
| **Video-Bench Leaderboard** | Human-aligned benchmark with axes: video quality, aesthetics, temporal consistency, motion, alignment.                                                                              | ⭐ [GitHub](https://github.com/PKU-YuanGroup/Video-Bench) · 🤗 [HuggingFace Space](https://huggingface.co/spaces/LanguageBind/Video-Bench) |
| **VideoScore Leaderboard** | Leaderboard of models scored by VideoScore metric across T2V datasets.                                                                                                              | 🤗 [HuggingFace Space / Leaderboard](https://huggingface.co/spaces/TIGER-Lab/VideoScore-Leaderboard) |
| **VideoGen-RewardBench** | Preference-based benchmark evaluating video reward models across Visual Quality (VQ), Motion Quality (MQ), Temporal Alignment (TA), Overall. Derived from pairwise human judgments. | ⭐ [GitHub / VideoReward](https://github.com/KwaiVGI/VideoAlign) · 🤗 [HF Space / Leaderboard](https://huggingface.co/spaces/KwaiVGI/VideoGen-RewardBench) |
| **MJ-Bench / MJ-VIDEO** | A fine-grained video preference benchmark across 5 aspects: Alignment, Safety, Fineness, Coherence & Consistency, Bias & Fairness.                                                  | 🌐 [Project / Data](https://aiming-lab.github.io/MJ-VIDEO.github.io/) · ⭐ [GitHub](https://github.com/aiming-lab/MJ-Video) · 🤗 [HuggingFace / MJ-Bench Team](https://huggingface.co/MJ-Bench) |
| **ArtificialAnalysis Arena** | Crowd-voting / preference-based leaderboard for T2V models.                                                                                                                         | 🤗 [HuggingFace Space](https://huggingface.co/spaces/ArtificialAnalysis/Video-Generation-Arena-Leaderboard) |
| **Labelbox Video Generation Leaderboard** | Industry evaluation platform’s video generation leaderboard.                                                                                                                        | 🌐 [Labelbox Video Leaderboards](https://labelbox.com/leaderboards/video-generation/) |
| **MMBench-Video Leaderboard** | Leaderboard for long-form video understanding tasks (video QA, reasoning) within MMBench-Video.                                                                                     | 🌐 [MMBench-Video site](https://mmbench-video.github.io/) · OpenVLM Video Leaderboard link in the repo.

---
