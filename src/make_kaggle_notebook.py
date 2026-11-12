"""Generate notebooks/lexsi_pipeline.ipynb: baseline eval -> AlignTune LoRA fine-tune -> tuned eval on a Kaggle GPU.

The notebook reads the Hugging Face token from a Kaggle secret named HF_TOKEN (never embedded), downloads the
private dataset repo, writes the project scripts, and runs the three stages. Results land in /kaggle/working.

Run: python src/make_kaggle_notebook.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "notebooks" / "lexsi_pipeline.ipynb"
USER = "iamvisheshsrivastava"
DATASET_REPO = f"{USER}/lexsi-extraction-data"
DATASET_REPO_V2 = f"{USER}/lexsi-extraction-data-v2"
MODEL = "CohereLabs/tiny-aya-global"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(keepends=True)}


def script(name):
    return code(f"%%writefile /kaggle/working/{name}\n" + (HERE / name).read_text(encoding="utf-8"))


def main():
    cells = [
        md("# Structured extraction with Tiny Aya: baseline, fine-tune, re-evaluate\n\n"
           "Pipeline: CuratorKIT-curated data -> AlignTune LoRA SFT -> field-level evaluation.\n"
           "Needs a GPU, internet, and a Kaggle secret named `HF_TOKEN` (a Hugging Face token whose account has "
           "accepted the Tiny Aya license)."),
        code("!pip install -q aligntune peft bitsandbytes accelerate jsonschema huggingface_hub 2>&1 | tail -2"),
        code("import os\nfrom kaggle_secrets import UserSecretsClient\n"
             "from huggingface_hub import HfApi, login, snapshot_download\n"
             "token = UserSecretsClient().get_secret('HF_TOKEN').strip().strip('\"').strip(\"'\")\n"
             "os.environ['HF_TOKEN'] = token\n"
             "login(token=token)  # also saved on disk for the training and evaluation subprocesses\n"
             "print('logged in as', HfApi().whoami(token=token)['name'])\n"
             f"snapshot_download('{DATASET_REPO}', repo_type='dataset', local_dir='/kaggle/working/data/final', token=token)\n"
             "print(os.listdir('/kaggle/working/data/final'))"),
        script("schema.py"),
        script("eval_extraction.py"),
        script("eval_model.py"),
        script("train_sft.py"),
        md("## 1. Baseline (untuned model)"),
        code("%cd /kaggle/working\n"
             f"!python eval_model.py --model {MODEL} --data data/final --out results/baseline.json"),
        md("## 2. Fine-tune with AlignTune (LoRA)"),
        code(f"!python train_sft.py --model {MODEL} --data data/final/train.jsonl --out out"),
        md("## 3. Re-evaluate with the adapter"),
        code(f"!python eval_model.py --model {MODEL} --adapter out/adapter --data data/final --out results/tuned.json"),
        code("import json\nfor name in ('baseline', 'tuned'):\n"
             "    print(name, json.dumps(json.load(open(f'results/{name}.json'))['overall'], indent=1))"),
    ]
    nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}}}
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print("wrote", OUT)
    baseline_cells = cells[:6] + [
        md("## Baseline: untuned model with the full schema prompt\n\nThe pipeline notebook evaluated the untuned model with the "
           "minimal prompt used for fine-tuning, which never names the JSON keys. This run gives the untuned model the full "
           "schema instruction, for a fair comparison."),
        code("%cd /kaggle/working\n"
             f"!python eval_model.py --model {MODEL} --data data/final --prompt full --out results/baseline_full.json"),
        code("import json\nprint(json.dumps(json.load(open('results/baseline_full.json'))['overall'], indent=1))")]
    nb2 = dict(nb, cells=baseline_cells)
    OUT2 = OUT.with_name("lexsi_baseline_full.ipynb")
    OUT2.write_text(json.dumps(nb2, indent=1), encoding="utf-8")
    print("wrote", OUT2)

    # v2: bigger training set, a 120-document harder test set (test) and the original 24-document test set (test_v1)
    login = cells[2]
    login_v2 = code("".join(login["source"]).replace(DATASET_REPO, DATASET_REPO_V2))
    cells_v2 = [
        md("# Structured extraction with Tiny Aya, v2\n\n"
           "Runs, on the v2 data (a larger training set, a 120-document harder test set `test` and the original 24-document `test_v1`):\n"
           "1. the untuned model with the minimal prompt and with the full schema prompt,\n"
           "2. AlignTune LoRA fine-tuning on the larger training set (2 epochs, about 2 hours on a T4),\n"
           "3. the tuned model with the minimal prompt on both test sets.\n\n"
           "All evaluations stop at the closing brace. Needs a GPU, internet and the `HF_TOKEN` secret."),
        cells[1], login_v2, cells[3], cells[4], cells[5], cells[6],
        md("## 1. Untuned model, two prompts, two test sets"),
        code("%cd /kaggle/working\n"
             f"!python eval_model.py --model {MODEL} --data data/final --split test --prompt short --out results/v2_baseline_short.json\n"
             f"!python eval_model.py --model {MODEL} --data data/final --split test --prompt full --out results/v2_baseline_full.json\n"
             f"!python eval_model.py --model {MODEL} --data data/final --split test_v1 --prompt full --out results/v1test_baseline_full.json"),
        md("## 2. Fine-tune with AlignTune (LoRA, 2 epochs)"),
        code(f"!python train_sft.py --model {MODEL} --data data/final/train.jsonl --out out --epochs 2"),
        md("## 3. Tuned model on both test sets"),
        code("%cd /kaggle/working\n"
             f"!python eval_model.py --model {MODEL} --adapter out/adapter --data data/final --split test --out results/v2_tuned.json\n"
             f"!python eval_model.py --model {MODEL} --adapter out/adapter --data data/final --split test_v1 --out results/v1test_tuned.json"),
        code("import json, glob\nfor path in sorted(glob.glob('results/*.json')):\n"
             "    if path.endswith('predictions.json'):\n        continue\n"
             "    d = json.load(open(path))\n"
             "    print(path, {k: round(v, 3) for k, v in d['overall'].items()}, 'at token limit:', d.get('outputs_at_token_limit'),"
             " 'seconds:', round(d['seconds']))")]
    OUT3 = OUT.with_name("lexsi_pipeline_v2.ipynb")
    OUT3.write_text(json.dumps(dict(nb, cells=cells_v2), indent=1), encoding="utf-8")
    print("wrote", OUT3)


if __name__ == "__main__":
    main()
