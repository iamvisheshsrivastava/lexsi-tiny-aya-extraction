"""LoRA supervised fine-tuning of a Tiny Aya model on the curated extraction data with AlignTune.

Meant for a GPU session (Kaggle / Colab). Needs `pip install aligntune` and a Hugging Face token that has
accepted the model's license.

Run: python train_sft.py --model CohereLabs/tiny-aya-global --data data/final/train.jsonl --out out
"""
import argparse

from aligntune.core.backend_factory import create_sft_trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", default="data/final/train.jsonl")
    parser.add_argument("--out", default="out")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    trainer = create_sft_trainer(
        model_name=args.model,
        dataset_name=args.data,
        backend="trl",
        output_dir=args.out,
        num_epochs=args.epochs,
        batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        max_seq_length=1024,
        use_peft=True,
        lora_r=16,
        lora_alpha=32,
        quantization={"load_in_4bit": True},
        task_type="sft",
        format_type="alpaca",
    )
    trainer.train()
    print("saved to", trainer.save_model(args.out + "/adapter"))


if __name__ == "__main__":
    main()
