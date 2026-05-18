seamless_data_dir="/home/tthebau1/SPEAR/SPEARBench/data/seamless_interaction/datasets/"
seamless_assets_dir="data/seamless_assets"


protocol="seamless_1t_1s_questions"
data_dir="data/$protocol"
llm_model="gpt-4o-audio-preview-2025-06-03"
asr="whisper-large-v3"
# proper evaluation would select a different one... that's on the TODO list
stance_llm_model=$llm_model

# Statistical test used for report p-values. Options: "Welch t-test", "Mann-Whitney U Test", "Wilcoxon Signed-Rank Test"
statistical_test="Mann-Whitney U Test"
