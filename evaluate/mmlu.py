# eval_shortgpt_mmlu.py

from mmengine.config import read_base
from opencompass.models import HuggingFaceBaseModel


# ============================================================
# Model paths
# ============================================================

BASE_MODEL = "/acpl-ssd32/meta-llama/Llama-2-7b-hf"

PRUNED_MODEL = (
    "/home/devewha/K_ShortGPT/"
    "outputs/llama2-7b-shortgpt-pruned-9layers"
)


# ============================================================
# MMLU
#
# ShortGPT paper:
#   - multiple-choice evaluation
#   - PPL-based answer selection
#
# OpenCompass mmlu_ppl:
#   - 57 MMLU subjects
#   - 5-shot (dev examples 0~4)
#   - PPLInferencer
# ============================================================

with read_base():
    from opencompass.configs.datasets.mmlu.mmlu_ppl import mmlu_datasets
    from opencompass.configs.summarizers.groups.mmlu import mmlu_summary_groups


datasets = [*mmlu_datasets]


# ============================================================
# Models
# ============================================================

models = [
    # --------------------------------------------------------
    # Original Llama2-7B
    # --------------------------------------------------------
    dict(
        type=HuggingFaceBaseModel,

        path=BASE_MODEL,
        tokenizer_path=BASE_MODEL,

        abbr="llama2-7b-base",

        # Llama2 context length
        max_seq_len=4096,

        # Not important for PPL evaluation,
        # but kept as a normal OpenCompass model setting.
        max_out_len=1024,

        batch_size=1,

        run_cfg=dict(
            num_gpus=1,
        ),
    ),

    # --------------------------------------------------------
    # ShortGPT pruned Llama2-7B
    # --------------------------------------------------------
    dict(
        type=HuggingFaceBaseModel,

        path=PRUNED_MODEL,

        # IMPORTANT:
        # Use exactly the same tokenizer as the original model.
        tokenizer_path=BASE_MODEL,

        abbr="llama2-7b-shortgpt-pruned-9layers",

        max_seq_len=4096,
        max_out_len=1024,

        batch_size=1,

        run_cfg=dict(
            num_gpus=1,
        ),
    ),
]


# ============================================================
# Summary
#
# Paper-style single MMLU score:
# mean accuracy across the 57 MMLU subjects.
# ============================================================

summarizer = dict(
    dataset_abbrs=[
        ["mmlu", "naive_average"],
    ],
    summary_groups=mmlu_summary_groups,
)


# ============================================================
# Output directory
# ============================================================

work_dir = "outputs/shortgpt_mmlu"