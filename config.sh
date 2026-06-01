seamless_data_dir="/home/tthebau1/SPEAR/SPEARBench/data/seamless_interaction/datasets/"
seamless_assets_dir="data/seamless_assets"


protocol="seamless_2t_2s_questions"
data_dir="data/$protocol"

llm_model="gpt-4o-audio-preview-2025-06-03" #deprecated
llm_model="gpt-audio-1.5"
llm_model="gpt-realtime-2"

language_id_model=facebook/mms-lid-126
dialect_id_model=tiantiaf/voxlect-english-dialect-whisper-large-v3
asr="whisper-large-v3"
# proper evaluation would select a different one... that's on the TODO list
stance_llm_model="gpt-audio-1.5"

# Statistical test used for report p-values. Options: "Welch t-test", "Mann-Whitney U Test", "Wilcoxon Signed-Rank Test"
statistical_test="Mann-Whitney U Test"
# Explainable feature columns ignored by scripts 41 and 51.
ignored_explainable_features=("f0_p10" "f0_p25" "f0_p90" "f0_std_raw" "f0_voiced_ratio" "f0_min_raw")

