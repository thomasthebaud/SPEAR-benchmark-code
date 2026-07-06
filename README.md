# SPEARBench

SPEARBench is a benchmark repository created for the SPEAR project, funded by Amazon and developed at Johns Hopkins University.

The benchmark evaluates speech-to-speech language models in conversational settings. It prepares question-answer dialogue clips from Seamless Interaction, runs speech-to-speech LLM inference, transcribes answers, computes audio/text metrics, scores naturalness and STANCE behavior, extracts explainable baseline features, and generates reports.

## Data Setup

SPEARBench builds evaluation clips from the Seamless Interaction dataset. Download the `dev` and `test` splits for both labels used by the benchmark:

- `improvised/dev`
- `improvised/test`
- `naturalistic/dev`
- `naturalistic/test`

You can download Seamless Interaction with the official repository:

```bash
git clone https://github.com/facebookresearch/seamless_interaction
cd seamless_interaction
pip install -e .
```

Then download the required splits with the official `SeamlessInteractionFS` API. The example below downloads one batch/archive; increase the batch/archive coverage as needed for your run.

```python
from pathlib import Path
from seamless_interaction.fs import SeamlessInteractionFS, DatasetConfig

root = Path("/path/to/seamless_interaction/datasets")

for label in ["improvised", "naturalistic"]:
    for split in ["dev", "test"]:
        config = DatasetConfig(
            label=label,
            split=split,
            local_dir=root,
            preferred_vendors_only=True,
        )
        fs = SeamlessInteractionFS(config=config)
        fs.download_batch_from_hf(batch_idx=0, archive_list=[0])
```

The benchmark expects the downloaded data to be formatted like this:

```text
seamless_interaction/
└── datasets/
    ├── assets/
    │   ├── dyad_lookup.csv
    │   ├── filelist.csv
    │   ├── interactions.csv
    │   ├── interactions_role.csv
    │   ├── interactions_role_ABmapped.csv
    │   ├── participants.csv
    │   └── relationships.csv
    ├── improvised/
    │   ├── dev/0000/0000/<file_id>.wav
    │   └── test/0000/0000/<file_id>.wav
    └── naturalistic/
        ├── dev/0000/0000/<file_id>.wav
        └── test/0000/0000/<file_id>.wav
```

Each participant file should have the matching annotation JSON next to the audio:

```text
<file_id>.wav
<file_id>.json
```

The benchmark scripts read the dataset through `config.sh`:

```bash
seamless_data_dir="/path/to/seamless_interaction/datasets/"
seamless_assets_dir="data/seamless_assets"
```

If your data lives outside this repository, either point `seamless_data_dir` to that location or create a symlink.

## Installation

The benchmark uses the conda environment defined in `environment.yml`.

From this directory:

```bash
conda env create -f environment.yml
conda activate spearbench
```

If you update `environment.yml` later, refresh the environment with:

```bash
conda env update -f environment.yml --prune
```

The environment currently uses Python 3.11 and CUDA 11.8 PyTorch wheels, including:

```text
torch==2.7.1+cu118
torchaudio==2.7.1+cu118
torchvision==0.22.1+cu118
```

It also installs packages used by the benchmark pipeline, including `pandas`, `scipy`, `scikit-learn`, `librosa`, `matplotlib`, `seaborn`, `soundfile`, `qwen-asr`, `silero-vad`, `sentence-transformers`, and the VoxLect/Whisper dependencies.

`Qwen3-ASR-*` can require a newer `qwen-asr`/`transformers` stack than the VoxLect-oriented `spearbench` environment. On the cluster, keep a separate ASR environment such as `spearbenchASR` if needed, and install packages with the environment's Python explicitly:

```bash
conda activate spearbenchASR
python -m pip install -U qwen-asr==0.0.6 transformers==4.57.6
```

When debugging ASR imports, prefer `python -m pip show qwen-asr transformers torch torchvision` over bare `pip`, so you are checking the active conda environment.

## Configuration

Before running the benchmark, edit `config.sh` so it points to your local data, model, and protocol settings.

Important fields:

- `seamless_data_dir`: path to the Seamless Interaction dataset directory.
- `seamless_assets_dir`: path to the benchmark assets directory, usually `data/seamless_assets`.
- `protocol`: name of the benchmark protocol to create and evaluate. The current default is `seamless_2t_2s_questions`.
- `data_dir`: derived output data directory for the selected protocol.
- `llm_model`: speech-to-speech LLM used by single-model stages and ad hoc runs. Current proxy files include `gpt-audio-1.5`, `gpt-realtime-2`, `Qwen3-Omni-30B-A3B-Instruct`, `Qwen2.5-Omni-7B`, `mini-omni`, `gemini-2.5-flash-native-audio-preview`, and `gemini-3.1-flash-live-preview`.
- `eval_models`: model list used by multi-model stages, including LLM inference, transcription, base metrics, language/dialect analysis, naturalness scoring, STANCE metric aggregation, explainable features, missing-file checks, report generation, benchmark-table merging, and article graph generation. Set it to the full model list for a complete benchmark, or to `($llm_model)` for one-model runs. Some scripts compare against `original` internally, while others only process the models listed here.
- `language_id_model`: Hugging Face audio language ID model, currently `facebook/mms-lid-126`.
- `dialect_id_model`: VoxLect dialect model used by the language/dialect stage.
- `asr_models`: ASR model names used for transcription and report labels. Transcription currently runs both `Qwen3-ASR-0.6B` and `whisper-large-v3`.
- `stance_llm_model`: LLM used as the STANCE judge.
- `statistical_test`: statistical test used by report p-values.

OpenAI-based scripts read credentials from `openai_keys.sh`. Create or edit that file before running LLM inference or STANCE scoring:

```bash
openai_api_key="YOUR_API_KEY"
org="YOUR_ORG_ID"
gemini_api_key="GEMINI_KEY"
```

## Command Helpers

Pipeline scripts source `cmd.sh` to build Slurm launch commands in a consistent way. The main helper is:

```bash
$(python_cmd JOB_NAME --cpu)
$(python_cmd JOB_NAME --gpu)
```

`JOB_NAME` is the Slurm job name, and the flag selects the scheduler resources. For example:

```bash
$(python_cmd SB10-S1 --gpu) bin/base_metrics.py --metadata metadata.csv --outputs base_metrics.csv
$(python_cmd SB51-S3 --cpu) bin/reports/generate_html_report.py --protocol "$protocol" --model "$model" --report-dir "$report_dir"
```

With the current `cmd.sh`, these expand to commands like:

```bash
srun -p gpu --gpus 1 --job-name SB10-S1 python3
srun -p cpu --cpus-per-task 4 --job-name SB51-S3 python3
```

Use `python_cmd` for Python stages. `srun_cmd JOB_NAME --cpu|--gpu` is also available when a script needs the Slurm prefix without automatically appending `python3`. Edit `cmd.sh` if the cluster partition, GPU count, CPU count, or excluded nodes need to change. The `exclude` variable is currently empty; set it in `cmd.sh` if some nodes should be avoided.

## Model Proxies

`02_run_LLM_inference.sh` selects model proxies from `eval_models`. Each speech-to-speech backend should have a module at:

```text
benchmark/bin/llm_proxies/${llm_model}.py
```

The proxy must define `get_reply_with_audio(...)` and return the answer audio bytes, answer transcript, finish reason, success flag, and answer start time. `run_LLM_inference.py` saves only the answer audio, not the full question+answer conversation.

## Data Layout

`01_prepare_data_from_seamless.sh` writes question clips under:

```text
data/$protocol/inputs/$split/$subset/
```

and original answer clips under:

```text
data/$protocol/outputs/original/$split/$subset/
```

LLM-generated answer clips are written under:

```text
data/$protocol/outputs/$model/$split/$subset/
```

Each metadata CSV uses the current question-answer schema. The diagram below illustrates how the fields created by `01_prepare_data_from_seamless.sh` relate to the question and answer audio:

![Metadata fields created by script 01](metadata_fields.png)

- `audio_path`: question/context audio clip.
- `answer_audio_path`: answer-only audio clip.
- `context_end_time`: time in `audio_path` where the final context ends and the question turn begins.
- `question_end_time`: time in `audio_path` where the question ends.
- `answer_duration`: duration of the answer-only audio.
- `speakers`: pipe-separated speaker IDs.
- `conversation_id`: source Seamless conversation ID.
- `transcript_question`: text for the context and question.
- `transcript_answer`: text for the answer.
- `answer_start_time`: answer onset relative to the end of the question. `0` means immediate answer, positive values mean a pause, and negative values mean the answer starts before the question audio has fully ended.
- `finish_reason`: LLM proxy finish status, present for generated outputs.

This split question/answer layout is important: most metric scripts analyze `answer_audio_path`, while naturalness and STANCE reconstruct the two-turn interaction when they need the question and answer together.

## Pipeline Scripts

Run scripts from the `benchmark` directory. The scripts are numbered in the intended pipeline order.

### `01_prepare_data_from_seamless.sh`

Creates the benchmark data from Seamless Interaction. It selects two-speaker dialogues with at least two turns where a speaker asks a question and the next speaker answers. It exports:

- question/context stereo clips to `data/$protocol/inputs/$split/$subset/audios/`;
- original answer-only stereo clips to `data/$protocol/outputs/original/$split/$subset/audios/`;
- metadata with `context_end_time`, `question_end_time`, `answer_start_time`, and answer transcript fields.

After preparation, `bin/data_prep_summary.py` prints a compact summary for the original-answer outputs.

### `02_run_LLM_inference.sh`

Runs speech-to-speech LLM inference on the prepared question clips. The current script loops over every model in `eval_models`, sends each `audio_path` from the input metadata to the corresponding proxy, and writes generated answer-only audio plus metadata under `data/$protocol/outputs/$model/$split/$subset/`.

The runner keeps the original question metadata and adds or updates:

- `answer_audio_path`
- `transcript_answer`
- `answer_start_time`, stored relative to `question_end_time`
- `answer_duration`
- `finish_reason`

Existing output metadata causes the Python runner to skip that split/subset.

### `02b_run_parallel_LLM_inference.sh`

Runs sharded LLM inference for the single model named by `llm_model`. This is useful for GPU-heavy local proxies where one split/subset should be divided across multiple one-GPU jobs:

```bash
bash 02b_run_parallel_LLM_inference.sh --num-gpus 4
bash 02b_run_parallel_LLM_inference.sh --num-gpus 4 --gpu-a100
```

You can also set `N_GPUS=4`. For each split/subset, the script launches `N` shards, writes `metadata_shard_<idx>_of_<N>.csv`, and merges the shards into the normal output file:

```text
data/$protocol/outputs/$llm_model/$split/$subset/metadata.csv
```

### `03_transcribe.sh`

Runs ASR over answer audio for both `dev` and `test`, both subsets, and every model in `eval_models`. If you also need original-answer transcripts for comparison runs, include `original` in the loop or run the transcriber directly on `data/$protocol/outputs/original`. It currently runs the ASR systems configured in `asr_models`, usually:

- `Qwen3-ASR-0.6B`
- `whisper-large-v3`

Transcript CSVs are written next to each metadata file, for example:

```text
data/$protocol/outputs/$model/$split/$subset/Qwen3-ASR-0.6B_transcripts.csv
data/$protocol/outputs/$model/$split/$subset/whisper-large-v3_transcripts.csv
```

### `04_verify_file_integrity.sh`

Verifies generated LLM output metadata against the corresponding input metadata for every split, subset, and model in `eval_models`. It checks that referenced audio files exist and writes a verified copy next to each generated metadata file:

```text
data/$protocol/outputs/$model/$split/$subset/metadata_verified.csv
```

The script currently does not overwrite `metadata.csv` automatically; the promotion block is left commented out so you can inspect `metadata_verified.csv` first. The current `10_compute_base_metrics.sh` reads `metadata.csv`, so promote or copy the verified metadata yourself if you want downstream stages to use it.

### `10_compute_base_metrics.sh`

Computes answer-level base metrics for every split/subset/model combination in `eval_models` and writes:

```text
results/$protocol/$model/$split/$subset/base_metrics.csv
```

Current metrics include WER/CER when ASR transcripts are present, response latency from VAD, interruption count, interrupted time in milliseconds, and UTMOS speech quality. These metrics operate on `answer_audio_path` and use `answer_start_time` for turn-timing measures.

Latency is VAD-aware. For each row, `bin/base_metrics.py` measures the trailing silence between the last voiced segment in `audio_path` and the end of the question clip, measures the leading silence before the first voiced segment in `answer_audio_path`, and defines latency as:

```text
question trailing silence + answer leading silence + max(answer_start_time, 0)
```

When `answer_start_time` is negative, the answer VAD is computed only on `answer_audio_path[-answer_start_time:]`, so speech that overlaps the question is not counted as leading silence. Negative `answer_start_time` values still indicate interruptions and are used by the interruption metrics. The current shell script reads `metadata.csv` from each output directory; use `04_verify_file_integrity.sh` first when you want to inspect or promote verified metadata before metric computation.

### `11_language_dialect.sh`

Runs language and dialect analysis on answer audio. With no arguments it runs all stages:

```bash
bash 11_language_dialect.sh
```

Optional stages:

```bash
bash 11_language_dialect.sh --language
bash 11_language_dialect.sh --dialect
bash 11_language_dialect.sh --summary
bash 11_language_dialect.sh --all
```

Stage 1 writes language predictions to:

```text
results/$protocol/$model/$split/$subset/language_id.csv
```

Stage 2 writes dialect predictions to:

```text
results/$protocol/$model/$split/$subset/dialect_id.csv
```

Dialect prediction uses language predictions to decide whether an utterance should be sent to the English VoxLect model. The language and dialect stages run over `eval_models`; the summary helper currently prints aggregate counts for `original` and `$llm_model`.

### `12_dialect_loglogits.sh`

Projects VoxLect dialect log-logits for `original` and every model in `eval_models`. Plot labels use the same display cleanup as the report graphs: Gemini `-preview` suffixes are removed and model names are capitalized consistently. It writes tabular outputs under:

```text
results/$protocol/dialect_logits/
```

and graph files rooted at:

```text
graphs/dialect_logits*
```

### `20_naturalness_feats.sh`

Extracts VoxProfile-style Whisper emotion features used by the naturalness model for `dev` and `test`, both subsets, and `original` plus every model in `eval_models`. With no flags it runs both stages; use `--extract`/`--stage1` for only feature extraction and `--aggregate`/`--stage2` for only SER_AVD aggregation.

Feature extraction accepts variable chunk sizes:

```bash
bash 20_naturalness_feats.sh --extract --chunk-size 3.0 --hop-size 1.0
bash 20_naturalness_feats.sh --extract --chunk-size 0
```

`--chunk-size 0` tells `bin/naturalness/extract_features.py` to process each full question turn and each full answer turn as one chunk. Without CLI flags, script `20` uses `naturalness_chunk_size=${naturalness_chunk_size:-3.0}` and `naturalness_chunk_hop_size=${naturalness_chunk_hop_size:-1.0}`, so those can also be set in the environment or `config.sh`.

`bin/naturalness/extract_features.py` handles the split question/answer layout. For each metadata row it:

- loads the question from `audio_path`, using `context_end_time` through `question_end_time`;
- loads the full answer from `answer_audio_path`;
- chunks each turn separately with the configured sliding window, or as one full-turn chunk when `--win-sec 0`;
- computes question embeddings and answer embeddings separately;
- concatenates the resulting embeddings and saves them under a single base key, so downstream scoring sees one sequence for the row;
- skips rows where all expected embedding chunks already exist;
- skips questions shorter than `--min-len-question`, currently set to `1.0` by the shell script.

Features are saved under:

```text
data/$protocol/outputs/$model/$split/$subset/naturalness/voxprofile_features/
```

`bin/naturalness/extract_features.py` also stores full-turn VoxProfile `turn_arousal`, `turn_dominance`, and `turn_valence` values for the whole question and whole answer. `bin/naturalness/aggregate_AVD_emotions.py` exports those full-turn values to:

```text
results/$protocol/$model/$split/$subset/SER_AVD.csv
```

### `21_extract_relations_context.sh`

Builds text embedding caches used by naturalness scoring from input metadata and Seamless asset files. It writes:

```text
data/$protocol/inputs/$split/$subset/context_hf_cache.pkl
data/$protocol/inputs/$split/$subset/relationship_hf_cache.pkl
```

### `22_score_naturalness.sh`

Scores utterances with the trained naturalness model. It combines precomputed audio embeddings with cached context and relationship embeddings and writes outputs under:

```text
results/$protocol/$model/$split/$subset/
```

The current shell script runs this scoring stage for both `test` and `dev`, across both subsets, for `original` and every model in `eval_models`. It launches `bin/naturalness/score.py` with `$(python_cmd 'SB22' --gpu)`. The scorer defaults to `--device auto`, which uses CUDA when a GPU is visible, honors `LOCAL_RANK` when set, and falls back to CPU when CUDA is unavailable. You can override this manually with `--device cpu`, `--device cuda`, or `--device cuda:0` when invoking `score.py` directly.

Scripts `20`, `21`, and `22` are based on the TRACE emotional naturalness pipeline. See [References](#references) for the paper citation and full implementation.

### `30_run_LLM_inference_STANCEs.sh`

Builds STANCE question CSVs and uses an LLM judge to score stance/tone/style dimensions. It currently runs STANCE question indices `0` through `9` for `dev` and `test`, only on the `improvised` subset, for every model in `eval_models`. The naturalistic subset is intentionally skipped because the STANCE setup depends on ground-truth stance role metadata available for improvised interactions. Include or process `original` separately when you need fresh original STANCE question files or judge scores.

For each question index, `bin/STANCE/make_questions.py` filters rows by role/category and writes:

```text
data/$protocol/outputs/$model/$split/improvised/stance_questions_Q<idx>.csv
```

`bin/STANCE/score.py` then sends a patched two-turn audio file to the judge model. It loads the question from `audio_path`, the answer from `answer_audio_path`, and reconstructs their timing using `answer_start_time`:

- `answer_start_time == 0`: answer starts immediately after the question.
- `answer_start_time > 0`: silence is inserted before the answer.
- `answer_start_time < 0`: the answer overlaps the end of the question, representing an interruption.

The temporary patched WAV is deleted after the request. STANCE metrics are written under:

```text
results/$protocol/$model/$split/improvised/
```

### `31_compute_STANCE_metrics.sh`

Merges STANCE outputs from `original` and each model in `eval_models` for `dev` and `test`, improvised only. It expects original STANCE outputs to already exist under `results/$protocol/original/...`. The merged output is:

```text
results/$protocol/$model/$split/improvised/merged_stances.csv
```

Scripts `30` and `31` are based on the StanceBench audio LLM interpersonal stance evaluation setup. See [References](#references) for the paper citation and full implementation.

### `40_preprocess_audios.sh`

Creates word-level transcription and VAD JSON files used by the explainable baseline feature extractor. It runs WhisperX alignment plus Silero VAD through `bin/distrib_baselines/preprocess_audio_alignments.py` and writes Seamless-style JSON files under each output directory:

```text
data/$protocol/outputs/$model/$split/$subset/baseline_prepreprocess/
```

Each JSON contains:

- `metadata:transcript`: word-level transcript segments with `word`, `start`, `end`, and optional `score`;
- `metadata:vad`: speech activity segments with `start` and `end`.

The script can preprocess answer audio from `answer_audio_path` and question/context audio from `audio_path`, depending on the active block in the shell script. Script `41` will use these JSON files when they exist and fall back to metadata-derived estimates when they do not.

### `41_extract_explainable_features.sh`

Extracts and normalizes explainable distributional baseline features. With no arguments, both stages run; stages can also be selected explicitly:

```bash
bash 41_extract_explainable_features.sh --extract
bash 41_extract_explainable_features.sh --normalize
bash 41_extract_explainable_features.sh --all
```

Stage 1, `--extract`, computes prosodic, lexical, temporal, and relationship-aware features from metadata and audio. For lexical and temporal features, `bin/distrib_baselines/extract_features.py` first looks for matching JSON files from script `40` in `baseline_prepreprocess/`. If a JSON exists, lexical text comes from `metadata:transcript`, temporal word timings come from `metadata:transcript[*].words`, and VAD comes from `metadata:vad`. If no JSON exists, the extractor falls back to transcript and duration estimates from `metadata.csv`.

Answer-side features are written as:

```text
results/$protocol/$model/$split/$subset/distrib_baselines_features.csv
```

For the `original` system, the script also extracts question-side features with `--questions`, using metadata `audio_path` instead of `answer_audio_path`, and writes:

```text
results/$protocol/original/$split/$subset/distrib_baselines_features_q.csv
```

Stage 2, `--normalize`, normalizes f0 features by the mean `f0_mean_raw` of the speaker who asks the final question. It also renames `f0_total_duration_s`, `f0_voiced_duration_s`, `f0_voiced_ratio`, and `f0_n_voiced_frames` by removing the `f0_` prefix. Normalized answer-side features are written as:

```text
results/$protocol/$model/$split/$subset/distrib_baselines_features_normalized.csv
```

For original question-side features, the normalizer is called with `--questions`, matches rows to metadata by `audio_path`, and writes:

```text
results/$protocol/original/$split/$subset/distrib_baselines_features_normalized_q.csv
```

### `42_use_features_for_baseline.sh`

Uses the normalized explainable features from script `41`. It has two optional stages:

```bash
bash 42_use_features_for_baseline.sh --scores
bash 42_use_features_for_baseline.sh --clusters
bash 42_use_features_for_baseline.sh --all
```

If no option is passed, both stages are run.

Stage 1, `--scores`, trains dev-set per-feature baselines and scores test utterances for every model in `eval_models`. It writes:

```text
results/$protocol/$model/test/$subset/distrib_baselines_feature_scores.csv
results/$protocol/$model/distrib_baselines_summary.csv
```

Stage 2, `--clusters`, loads both `original` and each evaluated model's explainable features before computing correlations. It normalizes from the combined dev set, computes Spearman correlations, clusters features at the configured threshold, validates the groups on test, and fits PCA features. It writes the same cluster definitions and transformed test values back into each model directory:

```text
results/$protocol/$model/test/$subset/correlation_feature_groups_rho0p8.csv
results/$protocol/$model/test/$subset/correlation_cluster_features_rho0p8.csv
```

The cluster feature CSV includes one PCA feature per correlated group plus a `general_explainable_feature_rho0p8` PCA feature across all groups. Features listed in `ignored_explainable_features` in `config.sh` are excluded from both stages and from reports.

Scripts `40`, `41`, and `42` are based on the Distributional Baselines for conversational prosody and rhythm. See [References](#references) for the paper citation and full implementation.

### `60_check_missing_files.sh`

Checks whether the expected output files from scripts `01` through `42` and scripts `61` through `63` have been produced. It sources `config.sh`, then verifies the expected metadata, transcript, metric, language/dialect, naturalness, STANCE, explainable-feature, baseline-score, correlation-cluster, report, benchmark-table, and article-graph files for every relevant split, subset, and model.

Run it from the benchmark directory with:

```bash
bash 60_check_missing_files.sh
```

By default it prints every expected file as either `found` or `missing`, plus an `all computed` summary for groups where every expected file exists. To print only missing files, use:

```bash
bash 60_check_missing_files.sh --missing-only
```

Example default output:

```text
[X] script 20   - model original        - subset test/improvised        - found: results/.../SER_AVD.csv
[ ] script 20   - model mini-omni       - subset test/improvised        - missing: results/.../SER_AVD.csv
```

The script exits with status `0` if all expected files exist and `1` if any file is missing.

### `61_generate_report.sh`

Generates human-readable reports from pipeline outputs. It has three optional stages:

```bash
bash 61_generate_report.sh --short
bash 61_generate_report.sh --graphs
bash 61_generate_report.sh --long
bash 61_generate_report.sh --all
```

If no option is passed, the script runs all stages.

The short report loops over every model in `eval_models`. It reads base metrics, naturalness scores, merged STANCE metrics, and normalized explainable-feature outputs from script `41`. The explainable-feature report is restricted to `total_duration_s`, `voiced_duration_s`, `voiced_ratio`, and columns whose names start with `f0_p`. It writes:

```text
reports/$protocol/$model/report.txt
reports/$protocol/$model/metrics-improvised.csv
reports/$protocol/$model/metrics-naturalistic.csv
```

The graph stage writes plots under:

```text
reports/$protocol/$model/graphs/
```

It includes `emotion_scatter.png`, a 2x3 grid comparing full-question vs full-answer Arousal, Dominance, and Valence from `SER_AVD.csv`. Columns are Arousal, Dominance, and Valence; rows are improvised and naturalistic. It also includes `emo_naturalness_by_relationship.png`, a single horizontal violin plot comparing original and model emotional-naturalness logits across naturalistic relationship categories, plus `cluster_explainables.png` and `general_explainable.png` for the correlation-cluster PCA features created by script `42`.

The long report stage writes:

```text
reports/$protocol/$model/detailed_report.html
```

Report generation uses `statistical_test` from `config.sh`, defaulting to Welch t-test if the variable is unset. Explainable features listed in `ignored_explainable_features` are omitted where applicable. The current short-report and graph explainable sections use the normalized script `41` feature files and only report `total_duration_s`, `voiced_duration_s`, `voiced_ratio`, and `f0_p*` columns.

### `62_benchmark.sh`

Generates compact benchmark CSV rows, a merged benchmark table, and a LaTeX table from the outputs of scripts `10`, `11`, `22`, `31`, and `41`. It has three optional stages:

```bash
bash 62_benchmark.sh --lines
bash 62_benchmark.sh --merge
bash 62_benchmark.sh --latex
bash 62_benchmark.sh --all
```

Aliases `--stage1`, `--stage2`, and `--stage3` are also supported. If no option is passed, all three stages are run.

Stage 1, `--lines`, writes one two-line CSV per evaluated model under:

```text
reports/$protocol/benchmark/$model.csv
```

Each row aggregates test-set metrics across `improvised` and `naturalistic`, including latency, UTMOS, WER, ASR-to-ASR WER variation, interruption rate and timing, language and dialect percentages, emotional naturalness, STANCE shifts, and the general explainable-feature PCA value. Negative latency values are excluded before computing average latency. Interruption metrics are intentionally hidden in the benchmark table for non-streaming or half-duplex systems where they are not comparable (`gpt-audio-1.5`, `mini-omni`, and `Qwen2.5-Omni-7B`).

STANCE polarity follows the article-facing positive/negative orientation. In particular, `aggression`, `inhibition`, `callousness`, and `disorganization` are reported as the inverted positive labels `Calmness`, `Disinhibition`, `Empathy`, and `Organization`, and the positive-vs-negative criterion is flipped consistently for those dimensions. Numeric values are rounded with `--n-digits`, which currently defaults to `3` in `62_benchmark.sh`. Add `--use-std` to include standard-deviation columns for mean-valued metrics:

```bash
bash 62_benchmark.sh --lines --n-digits 3 --use-std
```

Stage 2, `--merge`, concatenates the per-model CSV files into one protocol-level table:

```text
reports/$protocol/benchmark.csv
```

Stage 3, `--latex`, converts the merged benchmark CSV into:

```text
reports/$protocol/benchmark.tex
```

### `63_generate_article_graphs.sh`

Generates protocol-level article figures that compare all evaluated systems in single figures. The script activates the `spearbench` conda environment, submits each requested graph with `python_cmd`, and writes PNGs under:

```text
graphs/
```

With no arguments, all eight stages run. Stage flags are:

```bash
bash 63_generate_article_graphs.sh --stage1  # Intelligibility and Speech Quality
bash 63_generate_article_graphs.sh --stage2  # Interruptions and Latency
bash 63_generate_article_graphs.sh --stage3  # Dialects
bash 63_generate_article_graphs.sh --stage4  # Emotional Naturalness
bash 63_generate_article_graphs.sh --stage5  # AVD consistency
bash 63_generate_article_graphs.sh --stage6  # Stances
bash 63_generate_article_graphs.sh --stage7  # EXplainable features
bash 63_generate_article_graphs.sh --stage8  # Turn-taking naturalness histogram
bash 63_generate_article_graphs.sh --all
```

Rendered article figures:

![Stage 1: Intelligibility and Speech Quality](graphs/stage1_article_intelligibility_speech_quality.png)

![Stage 2: Interruptions and Latency](graphs/stage2_article_interruptions_latency.png)

![Stage 3: Dialects](graphs/stage3_article_dialects_point.png)

![Stage 4: Emotional Naturalness](graphs/stage4_article_emotional_naturalness_histogram.png)

![Stage 5: AVD consistency](graphs/stage5_article_avd_consistency.png)

![Stage 6: Stances](graphs/stage6_article_stances_spider.png)

![Stage 7: Explainable features](graphs/stage7_article_explainable_features_point_combined.png)

![Stage 8: Turn-taking naturalness histogram](graphs/stage8_article_turntaking_naturalness_histogram.png)


## Outputs

The main benchmark outputs are written under:

```text
results/$protocol/$model/$split/$subset/
```

Typical outputs include:

```text
metadata_verified.csv
base_metrics.csv
language_id.csv
dialect_id.csv
naturalness_scores.csv
naturalness_predictions_raw.csv
SER_AVD.csv
stance_metrics_Q<idx>.csv
merged_stances.csv
distrib_baselines_features.csv
distrib_baselines_features_normalized.csv
distrib_baselines_features_q.csv
distrib_baselines_features_normalized_q.csv
distrib_baselines_feature_scores.csv
```

Per-model report artifacts are written under:

```text
reports/$protocol/$model/
```

Typical per-model report outputs include:

```text
report.txt
metrics-improvised.csv
metrics-naturalistic.csv
graphs/
detailed_report.html
```

Benchmark-table artifacts from script `62` are written under:

```text
reports/$protocol/benchmark/$model.csv
reports/$protocol/benchmark.csv
reports/$protocol/benchmark.tex
```

Article figures from script `63` are written under:

```text
graphs/stage*_article_*.png
graphs/stage*_article_*_point.png
```

Model labels in report graphs are normalized for readability: Gemini `-preview` suffixes are removed where appropriate, and model names shown in legends are capitalized consistently.

## Current Benchmark Summary

The article-ready LaTeX table is generated at `reports/seamless_2t_2s_questions/benchmark.tex`, with values aggregated over the `seamless_2t_2s_questions` protocol. The summary table is generated from `reports/$protocol/benchmark.csv`. The website leaderboard adds two display columns before the metric columns: `inference_mode` and `open_weights`. Current model metadata used by the website is:

| Model | Inference mode | Open weights |
| --- | --- | --- |
| Human/original | Human | - |
| GPT-audio-1.5 | Non-streaming | no |
| GPT-realtime-2 | Full-duplex | no |
| Qwen3-Omni-30B-Instruct | Full-duplex | yes |
| Qwen2.5-Omni-7B | Half-duplex | yes |
| Gemini-2.5-flash-native-audio | Full-duplex | no |
| Gemini-3.1-flash-live | Full-duplex | no |
| Mini-omni | Half-duplex | yes |

For the current table, interruption columns are shown as `-` for GPT-audio-1.5, Mini-omni, and Qwen2.5-Omni-7B, and negative latency rows are excluded before the average latency is computed.

## References

The benchmark reuses or adapts feature pipelines from the following related work.

### TRACE Naturalness Features

Used by scripts `20_naturalness_feats.sh`, `21_extract_relations_context.sh`, and `22_score_naturalness.sh`.

- Paper: `TRACE: Temporal Relationship-Aware Conversational Entrainment Detection in Dyadic Speech`
- Code: [github.com/SathvikNapa/NaturalnessPrediction](https://github.com/SathvikNapa/NaturalnessPrediction)

```bibtex
@misc{TRACE,
      title={TRACE: Temporal Relationship-Aware Conversational Entrainment Detection in Dyadic Speech}, 
      author={Sathvik Manikantan Napa Ugandhar and Hao Zhang and Alison Gunzler and Yuzhe Wang and Thomas Thebaud and Georgi Tinchev and Venkatesh Ravichandran and Laureano Moro-Velázquez},
      year={2026},
      eprint={2606.30543},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2606.30543}, 
}
```

### StanceBench STANCE Features

Used by scripts `30_run_LLM_inference_STANCEs.sh` and `31_compute_STANCE_metrics.sh`.

- Paper: `StanceBench: A Benchmark for Audio LLM-Based Interpersonal Stance Evaluation from Speech`
- Code: [github.com/YuzheWangjhu/SPEAR_fine_grained_benchmark](https://github.com/YuzheWangjhu/SPEAR_fine_grained_benchmark)

```bibtex
@article{STANCE,
  title={StanceBench: A Benchmark for Audio LLM-Based Interpersonal Stance Evaluation from Speech},
  author={Yuzhe Wang and Thomas Thebaud and Jennifer Hu and Jes{\'u}s Villalba-Lopez and Venkatesh Ravichandran and Georgi Tinchev and Najim Dehak and Laureano Moro-Vel{\'a}zquez},
  journal={arXiv preprint arXiv:2506.10827},
  year={2026}
}
```

### Distributional Baselines

Used by scripts `40_preprocess_audios.sh`, `41_extract_explainable_features.sh`, and `42_use_features_for_baseline.sh`.

- Paper: `Reference-Based Prosody and Rhythm Evaluation for Spoken Dialogue Systems`
- Code: [github.com/Ashish-Hallur/SPEAR-Metrics](https://github.com/Ashish-Hallur/SPEAR-Metrics)

```bibtex
@misc{DISTRIB,
      title={Reference-Based Prosody and Rhythm Evaluation for Spoken Dialogue Systems}, 
      author={Ashish Hallur and Thomas Thebaud and Georgi Tinchev and Venkatesh Ravichandran and Laureano Moro-Velazquez},
      year={2026},
      eprint={2606.31055},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2606.31055}, 
}
```

### Turn Taking Surprisal

Used by script `50_turn_taking_inference.sh` and the turn-taking naturalness graph/report stages.

- Paper: `TurnNat: Automatic Evaluation of Turn-Taking Naturalness in Dyadic Spoken Dialogue`
- Code: [github.com/TedZhangHao/turn-taking-naturalness](https://github.com/TedZhangHao/turn-taking-naturalness)

```bibtex
@misc{TURNS,
      title={TurnNat: Automatic Evaluation of Turn-Taking Naturalness in Dyadic Spoken Dialogue}, 
      author={Hao Zhang and Thomas Thebaud and Georgi Tinchev and Venkatesh Ravichandran and Laureano Moro-Velazquez},
      year={2026},
      eprint={2607.01345},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2607.01345}, 
}
```

## Contact

For questions, please contact:

**Thomas Thebaud**  
_Assistant Research Professor_ <br>
ECE department, Johns Hopkins University
`tthebau1@jhu.edu`

## License

This benchmark code is released under the [MIT License](LICENSE). Third-party
datasets, model checkpoints, and external tools used with the benchmark remain
subject to their own licenses and terms.

## Citation

If you use this benchmark, please cite the SPEAR article.

ArXiv: `<ARXIV_LINK_PLACEHOLDER>`

```bibtex
@article{spearbench2026,
  title   = {SPEAR: Evaluating Speech-to-Speech Language Models in Conversational Settings},
  author  = {Thebaud, Thomas and others},
  journal = {arXiv preprint},
  year    = {2026},
  url     = {ARXIV_LINK_PLACEHOLDER}
}
```
