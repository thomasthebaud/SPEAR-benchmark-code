# SPEARBench

SPEARBench is a benchmark repository created for the SPEAR project, funded by Amazon and developed at Johns Hopkins University.

The goal of the project is to evaluate the naturalness of speech-to-speech large language models in conversational settings. This repository contains code to create an evaluation dataset, simulate conversations with an LLM, analyze the generated answers, and produce evaluation reports.

## Data Setup

SPEARBench builds its evaluation clips from the Seamless Interaction dataset. Download the `dev` and `test` splits for both labels used by the benchmark:

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

If your data lives outside this repository, either point `seamless_data_dir` to that location or create a symlink:

```bash
ln -s /path/to/seamless_interaction SPEARBench/data/seamless_interaction
```

The preparation script `01_prepare_data_from_seamless.sh` will then convert the downloaded Seamless Interaction files into the SPEARBench metadata and audio layout under `benchmark/data/$protocol`.

## Installation

The benchmark uses a conda environment defined in `environment.yml`.

From this directory:

```bash
conda env create -f environment.yml
conda activate spearbench
```

If you update `environment.yml` later, refresh the environment with:

```bash
conda env update -f environment.yml --prune
```

The environment installs PyTorch with CUDA 11.8 wheels through pip, including:

```text
torch==2.7.1+cu118
torchaudio==2.7.1+cu118
```

It also includes the analysis and plotting packages used by the reporting pipeline, including `pandas`, `scipy`, `scikit-learn`, `matplotlib`, and `seaborn`.

## Configuration

Before running the benchmark, edit `config.sh` so it points to your local data, model, and protocol settings.

Important fields:

- `seamless_data_dir`: path to the Seamless Interaction dataset directory.
- `seamless_assets_dir`: path to the benchmark assets directory, usually `data/seamless_assets`.
- `protocol`: name of the benchmark protocol to create and evaluate.
- `data_dir`: derived output data directory for the selected protocol.
- `llm_model`: speech-to-speech LLM used to generate answers.
- `asr`: ASR model used to transcribe generated audio.
- `stance_llm_model`: LLM used to score STANCE outputs.

OpenAI-based scripts also read credentials from `openai_keys.sh`. Create or edit that file with your API credentials before running inference or STANCE scoring.

Example:

```bash
openai_api_key="YOUR_API_KEY"
org="YOUR_ORG_ID"
```

## Pipeline Scripts

Run scripts from the `benchmark` directory. The scripts are numbered in the intended pipeline order.

### `01_prepare_data_from_seamless.sh`

Creates the benchmark data from the Seamless Interaction dataset. It selects conversations ending in a question, writes input question clips, writes original answer clips, and saves metadata for `test` and `dev` splits across `improvised` and `naturalistic` subsets.

### `02_run_LLM_inference.sh`

Runs speech-to-speech LLM inference on the prepared input audio clips. It sends each input audio file to the configured `llm_model`, saves the generated audio response, and writes output metadata.

### `03_transcribe.sh`

Runs ASR over generated and original answer audio with Whisper and Qwen ASR models. It writes model-specific transcript CSV files next to each metadata file instead of modifying the original metadata, for example:

```text
data/$protocol/outputs/$model/$split/$subset/whisper-large-v3_transcripts.csv
data/$protocol/outputs/$model/$split/$subset/Qwen3-ASR-0.6B_transcripts.csv
```

If you need to pre-download the Qwen ASR model weights, run:

```bash
huggingface-cli download Qwen/Qwen3-ASR-0.6B --local-dir models/Qwen3-ASR-0.6B
```

### `10_compute_base_metrics.sh`

Computes base answer metrics for the test set, including ASR text error metrics, response latency, and UTMOS-style speech quality scores.

### `20_naturalness_feats.sh`

Extracts audio features used by the naturalness model. It processes each utterance with the VoxProfile-style Whisper emotion feature extractor and saves features under each metadata directory.

### `21_extract_relations_context.sh`

Builds context and relationship text embedding caches for naturalness scoring. These caches are created from the input metadata and Seamless asset files.

### `22_score_naturalness.sh`

Scores each utterance with the trained naturalness model. It combines extracted audio features with cached context and relationship embeddings, then writes naturalness predictions and scores to `results/`.

### `30_run_LLM_inference_STANCEs.sh`

Builds STANCE question CSVs and uses an LLM judge to score conversational stance dimensions. The script runs across the configured STANCE question indices and role sets.

### `31_compute_STANCE_metrics.sh`

Merges STANCE outputs from original and LLM-generated answers into a combined CSV for comparison.

### `40_extract_explainable_features.sh`

Extracts explainable distributional baseline features for each utterance. It computes prosodic/F0, lexical, and temporal-style features from metadata and audio, then writes:

```text
results/$protocol/$model/$split/$subset/distrib_baselines_features.csv
```

The script processes the `dev` and `test` splits, both `improvised` and `naturalistic` subsets, and both `original` and the configured `$llm_model`.

### `41_use_features_for_baseline.sh`

Uses the explainable features extracted by `40_extract_explainable_features.sh` to build per-feature distributional baselines. For each subset, it trains a linear discriminant classifier on the `dev` split to distinguish `original` utterances from `$llm_model` utterances, then scores every test utterance.

It writes per-utterance scores under:

```text
results/$protocol/$llm_model/test/$subset/distrib_baselines_feature_scores.csv
```

It also writes a summary of per-feature classification accuracy and AUROC:

```text
results/$protocol/$llm_model/distrib_baselines_summary.csv
```

### `50_generate_report.sh`

Generates human-readable reports from the outputs of the benchmark pipeline. The script has three optional stages:

```bash
bash 50_generate_report.sh --short
bash 50_generate_report.sh --graphs
bash 50_generate_report.sh --long
bash 50_generate_report.sh --all
```

If no option is passed, the script runs `--short`.

#### Stage 1: Short Report

Reads the outputs of `10_compute_base_metrics.sh`, `22_score_naturalness.sh`, `31_compute_STANCE_metrics.sh`, and `41_use_features_for_baseline.sh`. It writes:

```text
reports/$llm_model/report.txt
reports/$llm_model/metrics-improvised.csv
reports/$llm_model/metrics-naturalistic.csv
```

The text report contains sections for:

- setup information, including the protocol, ASR model, evaluated LLM, STANCE inference LLM, and SBERT model used for naturalness prediction;
- data statistics;
- basic metrics from script `10`, such as WER/CER, UTMOS, and latency when available;
- emotional naturalness comparisons from `naturalness_scores.csv`;
- STANCE and explainable-feature summaries.

The metrics CSV files use one row per metric and columns for mean difference, standard deviation, p-value, AUROC, accuracy, and additional detail. STANCE metrics are reported for `improvised` only.

#### Stage 2: Graphs

Generates plots with `matplotlib` and `seaborn`, saved under:

```text
reports/$llm_model/graphs/
```

Main graph outputs include:

```text
reports/$llm_model/graphs/basic_metric_graphs/<metric>.png
reports/$llm_model/graphs/basic_metrics.png
reports/$llm_model/graphs/feat_graphs/<feature_name>.png
reports/$llm_model/graphs/stances.png
reports/$llm_model/graphs/emo_naturalness.png
reports/$llm_model/graphs/explainables.png
```

The explainable-feature overview uses normalized horizontal violin plots, with separate panels for `improvised` and `naturalistic`. Feature labels are marked with significance stars based on the test-set p-value.

#### Stage 3: Detailed HTML Report

Builds a browser-readable report from the short report, metrics CSVs, and generated graphs:

```text
reports/$llm_model/detailed_report.html
```

The HTML report includes rendered tables, sectioned metric summaries, and graph galleries for basic metrics, emotional naturalness, STANCEs, and explainable features.

## Outputs

The main benchmark outputs are written under:

```text
results/$protocol/$model/$split/$subset/
```

Typical outputs include base metrics, naturalness scores, STANCE metrics, merged STANCE outputs, explainable baseline features, and per-feature baseline scores.

The final report artifacts are written under:

```text
reports/$llm_model/
```

Typical report outputs include:

```text
report.txt
metrics-improvised.csv
metrics-naturalistic.csv
graphs/
detailed_report.html
```

## Contact

For questions, please contact:

**Thomas Thebaud**  
_Assistant Research Professor_ <br>
ECE department, Johns Hopkins University
`tthebau1@jhu.edu`

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
