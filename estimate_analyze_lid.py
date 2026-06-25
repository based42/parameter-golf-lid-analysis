import torch
import glob
import numpy as np
import torch.nn.functional as F
from pathlib import Path
from measure_variance_ratio import load_module_from_path
from measure_variance_ratio import build_model
from train_gpt import load_data_shard

def forward_pass_get_representations(model, input_ids):
    with torch.no_grad():
        x = model.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips = []

        # First half stores skips; second half reuses them in reverse order.
        for i in range(model.num_encoder_layers):
            x = model.blocks[i](x, x0)
            skips.append(x)
        for i in range(model.num_decoder_layers):
            if skips:
                x = x + model.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            x = model.blocks[model.num_encoder_layers + i](x, x0)

        return model.final_norm(x).reshape(-1, x.size(-1))

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mod = load_module_from_path("train_gpt.py")
    args = mod.Hyperparameters()
    model = build_model(mod, device, "standard")


    train_files = sorted(glob.glob(args.train_files))
    val_files = sorted(glob.glob(args.val_files))

    train_tokens = load_data_shard(Path(train_files[0]))
    val_tokens = load_data_shard(Path(val_files[0]))

    # probe size
    train_tokens = train_tokens[:8192]
    val_tokens = val_tokens[:8192]

if __name__ == "__main__":
    main()