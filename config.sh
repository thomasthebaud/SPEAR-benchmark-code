seamless_data_dir="/home/tthebau1/SPEAR/SPEARBench/data/seamless_interaction/datasets/"
seamless_assets_dir="data/seamless_assets"


protocol="seamless_2t_2s_questions"
data_dir="data/$protocol"

# LLMS to evaluate
llm_model="gpt-4o-audio-preview-2025-06-03" #deprecated
llm_model="gpt-audio-1.5"
llm_model="gpt-realtime-2"
# llm_model="Qwen3-Omni-30B-A3B-Instruct"
llm_model="Qwen2.5-Omni-7B"
llm_model="mini-omni" #https://huggingface.co/gpt-omni/mini-omni
# llm_model="original"

eval_models=("gpt-audio-1.5" "gpt-realtime-2" "Qwen3-Omni-30B-A3B-Instruct" "Qwen2.5-Omni-7B" "mini-omni")
# eval_models=($llm_model)

# Inference models
# 03_transcribe
asr_models=('Qwen3-ASR-0.6B' 'whisper-large-v3')
# 10_compute_base_metrics.sh: for latency and UTMOS, we compute for all models.
UTMOS_model="SpeechMOS/utmos22_strong"
VAD_model="silero_vad"
# 11_language_dialect_id.sh
language_id_model=facebook/mms-lid-126
dialect_id_model=tiantiaf/voxlect-english-dialect-whisper-large-v3

# 20/21/22 emotional naturalness
sbert_model="sentence-transformers/all-MiniLM-L6-v2"
ser_model='tiantiaf/whisper-large-v3-msp-podcast-emotion-dim'
emo_naturalness_checkpoint='models/naturalness/last_model.pt' 
# 30/31 stances
stance_llm_model="gpt-audio-1.5"


# For reporting
# Statistical test used for report p-values. Options: "Welch t-test", "Mann-Whitney U Test", "Wilcoxon Signed-Rank Test"
statistical_test="Mann-Whitney U Test"
# Explainable feature columns ignored by scripts 41 and 51.
ignored_explainable_features=("f0_n_voiced_frames" "f0_total_duration_s" "f0_mean_raw")

