import os
import torch
import pickle
import numpy as np
import torchaudio
import warnings
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from torchvision import models, transforms
import torch.nn as nn

# 忽略 torchvision 版本警告
warnings.filterwarnings('ignore')

# ---------------- 配置区域 (已按你要求修改) ----------------
# 你的原始数据路径 (包含 rawframes 和 audios)
# 注意：脚本在 Local 目录下运行，所以上一级是 ../data
RAW_DATA_DIR = Path("../data/ucf101") 

# 输出路径 (匹配 dataloader.py 的读取路径)
OUTPUT_DIR = Path("./results")
AUDIO_OUT_DIR = OUTPUT_DIR / "feature" / "audio" / "mfcc" / "ucf101"
VIDEO_OUT_DIR = OUTPUT_DIR / "feature" / "video" / "mobilenet_v2" / "ucf101"
# -------------------------------------------------------

class FeatureExtractor:
    def __init__(self):
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # 1. 初始化 Video 模型 (MobileNetV2)
        try:
            self.video_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        except:
            # 兼容旧版 torchvision
            self.video_model = models.mobilenet_v2(pretrained=True)
            
        # 移除分类层，只保留特征提取
        self.video_model.classifier = self.video_model.classifier[:-1]
        self.video_model = self.video_model.to(self.device)
        self.video_model.eval()
        
        self.img_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extract_video_features(self, video_dir):
        # 排序确保顺序一致
        rawframes = sorted(list(video_dir.glob("*.jpg")))
        if not rawframes:
            return None
            
        # [核心优化]：每 25 帧取 1 帧 (极速模式)
        rawframes = rawframes[::25]
        
        input_data_list = []
        for frame_path in rawframes:
            try:
                input_image = Image.open(frame_path).convert('RGB')
                input_tensor = self.img_transform(input_image)
                input_data_list.append(input_tensor.numpy())
            except Exception as e:
                print(f"Error reading {frame_path}: {e}")

        if not input_data_list:
            return None

        with torch.no_grad():
            input_data = torch.tensor(np.array(input_data_list)).to(self.device)
            features = self.video_model(input_data)
            features = features.detach().cpu().numpy()
            
        return features

    def extract_audio_features(self, audio_path):
        try:
            audio, sr = torchaudio.load(str(audio_path))
            
            if audio.shape[0] != 1:
                audio = torch.mean(audio, dim=0).unsqueeze(0)
            
            if sr != 16000:
                transform_model = torchaudio.transforms.Resample(sr, 16000)
                audio = transform_model(audio)
            
            features = torchaudio.compliance.kaldi.fbank(
                waveform=audio,
                frame_length=40,
                frame_shift=20,
                num_mel_bins=80,
                window_type="hamming"
            )
            
            features = features.detach().cpu().numpy()
            
            if features.shape[0] > 0:
                features = (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-5)
                
            return features
            
        except Exception as e:
            return None

def process_all():
    extractor = FeatureExtractor()
    
    # ----------------- 处理视频 -----------------
    print("\n" + "="*50)
    print("开始提取视频特征 (MobileNetV2 - 降采样 1/25)")
    print("="*50)
    
    video_root = RAW_DATA_DIR / "rawframes"
    video_dict = {}
    
    if video_root.exists():
        classes = sorted([d.name for d in video_root.iterdir() if d.is_dir()])
        
        # 外层循环：遍历类别
        for cls in tqdm(classes, desc="总体进度 (类别)", position=0):
            cls_dir = video_root / cls
            videos = sorted([d.name for d in cls_dir.iterdir() if d.is_dir()])
            
            # 内层循环：遍历视频 (显示当前类别的具体进度)
            # leave=False 表示跑完一个类别后，这行进度条会消失，保持界面清爽
            for vid in tqdm(videos, desc=f"正在处理 {cls}", leave=False, position=1):
                vid_path = cls_dir / vid
                feats = extractor.extract_video_features(vid_path)
                
                if feats is not None:
                    key = f"{cls}/{vid}"
                    video_dict[key] = feats
        
        VIDEO_OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = VIDEO_OUT_DIR / "feature.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(video_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"\n✅ 视频特征提取完成！已保存至: {out_path}")
        print(f"共处理 {len(video_dict)} 个视频")
    else:
        print(f"❌ 错误: 找不到目录 {video_root}")
        return

    # ----------------- 处理音频 -----------------
    print("\n" + "="*50)
    print("开始提取音频特征 (MFCC)")
    print("="*50)
    
    audio_root = RAW_DATA_DIR / "audios"
    audio_dict = {}
    
    if audio_root.exists():
        classes = sorted([d.name for d in audio_root.iterdir() if d.is_dir()])
        
        for cls in tqdm(classes, desc="总体进度 (类别)", position=0):
            cls_dir = audio_root / cls
            wavs = sorted(list(cls_dir.glob("*.wav")))
            
            for wav_path in tqdm(wavs, desc=f"正在处理 {cls}", leave=False, position=1):
                feats = extractor.extract_audio_features(wav_path)
                
                if feats is not None:
                    key = f"{cls}/{wav_path.stem}"
                    audio_dict[key] = feats
                    
        AUDIO_OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = AUDIO_OUT_DIR / "feature.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(audio_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"\n✅ 音频特征提取完成！已保存至: {out_path}")
        print(f"共处理 {len(audio_dict)} 个音频")
    else:
        print(f"❌ 错误: 找不到目录 {audio_root}")

if __name__ == "__main__":
    process_all()         