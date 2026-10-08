from datasets import load_dataset
from torch.utils.data import DataLoader
import torch

from shortgpt import ShortGPT

MODEL_PATH = "/acpl-ssd32/google/gemma-7b"
N_PRUNE_LAYERS = 6
OUTPUT_PATH = f"/acpl-ssd30/shortgpt/gemma-7b-{N_PRUNE_LAYERS}layers"

data = load_dataset("emozilla/pg19", split="validation")

dataloader = DataLoader(
    data,
    batch_size=1,
    shuffle=True,
    pin_memory=True,
    generator=torch.Generator(),
)

short_model = ShortGPT(
    model_name=MODEL_PATH,
    layers_path="model.layers",
    n_prune_layers=N_PRUNE_LAYERS,
)

for batch in dataloader:
    short_model.eval_importance(
        prompts=batch["text"],
        stride=256,
        max_gen_len=0,
    )

prune_order = sorted(
    range(len(short_model.importances)),
    key=lambda i: short_model.importances[i],
)
print("Pruning order:", ", ".join(map(str, prune_order)))

removed_layers = short_model.remove_layers()
print("Removed layers:", removed_layers)

for layer_idx, module in enumerate(short_model.layers):
    module.self_attn.layer_idx = layer_idx

short_model.model.config.num_hidden_layers = len(short_model.layers)

short_model.model.save_pretrained(
    OUTPUT_PATH,
    safe_serialization=True,
    max_shard_size="5GB",
)
short_model.tokenizer.save_pretrained(OUTPUT_PATH)

print(f"Saved pruned model to: {OUTPUT_PATH}")