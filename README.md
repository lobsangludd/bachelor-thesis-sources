# Folktale detection in *Kmetijske in rokodelske novice*

Data and scripts to regenerate the numbers reported in the thesis.

```
gold/gold.json         2,860 gold articles: urn, issue_urn, label, annotator verdict, genre, provenance, flags, note
lora/                  the frozen training set of the four reported adapters (2,398 train / 228 val rows)
eval/                  model outputs: 36 gold runs the thesis tables are computed from, plus the teacher's full corpus run
scripts/
  extract_articles.py  sPeriodika archive -> articles.json
  classify.py          run a model over the gold or full set with one of the three prompts
  evaluate.py          score runs against the gold, paired McNemar test
  train_lora.py        LoRA / QLoRA distillation of a local base on lora/
```

## Data files

**articles.json** is built from the sPeriodika 1.0 JSON distribution (http://hdl.handle.net/11356/1881, `sPeriodika.1.0.json.tgz`, 13.5 GB): every document whose `periodical_name` is *Kmetijske in rokodelske novice*, with the `text_csmtised_splitfixed` page texts joined. Total of 3,400 issues and 28,409 articles. This file is created when running scripts/extract_articles.py with sPeridika archive as the argument.

**gold/gold.json** is the annotation export collapsed to one entry per article. The binary flag that is used everywhere is `label`: `folk`, `not` or `unsure` (`unsure` counts as `not` when scoring). A raw `folk` (`annotator_verdict`) verdict becomes `not` when the tale is only embedded in a longer article, when the article talks about a tale instead of telling one, or when the genre is `other`, `anekdota` or `not_applicable`. Fields `annotator_verdict`, `genre`,`provenance`, the flags and the annotator's `note` are kept so every decision is transparent (29 folk, 18 unsure, 2,813 not).

**lora/** contains the exact rows the reported adapters were trained on: the bare prompt over a 16,000 + 6,000 character window as the user turn, the teacher's (`gpt-5.6-sol`, detailed prompt) `{reasoning, is_folktale}` as the target. Positives are oversampled (3 times) in the train split.

**eval/** contains one `gold_<prompt>_<model>.jsonl` per reported run (8 models - 3 prompts, 4 adapters - 3 prompts) and `full_detailed_gpt-5.6-sol.jsonl`, the teacher's verdicts over the whole corpus.

## Steps

Workstation requirements: Python 3.10+, `pip install openai tqdm`.

```bash
python scripts/extract_articles.py /path/to/sPeriodika.1.0.json.tgz
python scripts/evaluate.py eval/gold_*.jsonl
python scripts/evaluate.py eval/gold_detailed_gpt-5.6-sol.jsonl eval/gold_bare_gpt-5.6-sol.jsonl # model pair test
```

`evaluate.py` prints N, the confusion matrix, recall and precision with exact Clopper-Pearson 95 % intervals, F1, MCC, MCC with the unsure articles excluded, and the number of recovered malformed-JSON replies. 

With two files it adds the exact two-sided McNemar test over the conflicting articles. 

`classify.py` does the model article classification: 2,860 calls for `gold`, 28,409 for `full`
- first argument is the extent: gold or full
- second argument is the prompt: base, simple, detailed
- third argument is the model string
- fourth argument is the vllm address, if running against a local model

Frontier runs (set `OPENAI_API_KEY` env variable, Responses API, reasoning effort none, temperature 0):

```bash
python scripts/classify.py gold detailed gpt-5.6-sol
python scripts/classify.py full detailed gpt-5.6-sol
```

Local models, on a GPU box (the study used one L40S, 48 GB) with vLLM 0.27.1 and transformers 5.15.1 (5.14.1 for Gemma 4).

```bash
vllm serve cjvt/GaMS3-12B-Instruct --max-model-len 40960 --gpu-memory-utilization 0.90 &
python scripts/classify.py gold bare cjvt/GaMS3-12B-Instruct http://127.0.0.1:8000/v1
```

Qwen 3.5 additionally needs `--default-chat-template-kwargs '{"enable_thinking":false}' --skip-mm-profiling`. 

Every run outputs to `eval/<scope>_<prompt>_<model>.jsonl` and has an option to resume if interrupted.

LoRA (peft 0.20, bitsandbytes for `nf4` - about 3h per adapter on an L40S):

```bash
python scripts/train_lora.py cjvt/GaMS3-12B-Instruct outputs/gams3-bf16 bf16
python scripts/train_lora.py cjvt/GaMS3-12B-Instruct outputs/gams3-nf4  nf4
python scripts/train_lora.py google/gemma-4-12B-it   outputs/gemma4-bf16 bf16
python scripts/train_lora.py google/gemma-4-12B-it   outputs/gemma4-nf4  nf4
```

Run with parameters: r 32, alpha 32, dropout 0, all attention and MLP projections, lr 1e-4 cosine with 20 warm-up steps, 2 epochs (600 steps), batch 1 x grad-accum 8, sequence 8,192, bf16, seed 0, eval and checkpoint every 100 steps. Loss on the assistant turn only. The reported adapters are the checkpoints at step 600. Serve the adapter and evaluate it with the unchanged model:

```bash
vllm serve cjvt/GaMS3-12B-Instruct --enable-lora --lora-modules adapter=outputs/gams3-bf16 \
    --max-lora-rank 32 --max-model-len 40960 &
python scripts/classify.py gold bare adapter http://127.0.0.1:8000/v1
```

## Limits

- Frontier rows are not bit-reproducible because the API does not return model snapshot id.
- Local rows are greedy at temperature 0, but vLLM batching can flip single articles between runs.
- The gold sample and the annotation are human work and are shipped.
- The LoRA training set is shipped for the same reason. It was drawn from the verdicts on articles flagged by earlier local runs.
