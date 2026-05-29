# OMGs

Publication release for the OMGs ovarian-tumour MDT decision-support pipeline.
This package contains the standalone orchestration code, prompt configuration
and one de-identified runnable example.

## Environment

```bash
conda create -n OMGs python=3.10 -y
conda activate OMGs
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.lock.txt
pip install -e ../omgs_engine
```

## Configuration

```bash
cp .env.example .env
source .env
```

Fill provider credentials in `.env`. Raw prompts and raw model responses are
not persisted by default.

## Minimal example

```bash
make smoke
make run AGENT=omgs PROVIDER=azure MODEL=gpt-5.1
```

The example uses `examples/case001.deidentified.example.jsonl` and matching
de-identified report fixtures in `files/`.

For input-matched ablations:

```bash
python main.py \
  --input_path ./examples/case001.deidentified.example.jsonl \
  --agent omgs \
  --provider azure \
  --model gpt-5.1 \
  --num_samples 1 \
  --omgs_ablation_outputs
```

## Evidence boundary

The evaluated evidence runtime is local and versioned separately. Licensed,
restricted or institution-specific source assets are not redistributed here.
Relevant companion repositories are:

- <https://github.com/pigudog/omgs_engine>
- <https://github.com/pigudog/omgs_external_evidence>
- <https://github.com/pigudog/omgs_trial>
- <https://github.com/pigudog/omgs_guideline>
- <https://github.com/pigudog/omgs_nccn>

Public demonstration site: <https://www.omgsmdt.com>.
