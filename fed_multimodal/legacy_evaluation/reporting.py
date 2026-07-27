import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


def seed_evaluation(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluation_metadata(args, data):
    return {
        "partition": data.selected_partition,
        "client_id": args.client_id,
        "alpha": args.alpha,
        "fold": args.fold,
        "batch_size": args.batch_size,
        "num_batches": args.num_batches,
        "seed": args.seed,
        "teacher_checkpoint": args.model_path,
    }


def write_analysis_report(output_dir, payload):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path = output_dir / f"analysis_results_{timestamp}.json"
    with output_path.open("w") as handle:
        json.dump(payload, handle, indent=2)
    return output_path
