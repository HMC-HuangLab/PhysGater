import matplotlib.pyplot as plt


import argparse

def _parse_set_overrides():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE",
        help="Override config/global values from CLI, e.g. --set csv_path=/data/train.csv"
    )
    args, _ = parser.parse_known_args()
    out = {}
    for item in args.set:
        if "=" not in item:
            continue
        k, v = item.split("=", 1)
        out[k.strip()] = v.strip()
    return out

def _apply_overrides_to_class(cls, overrides):
    for k, v in overrides.items():
        if hasattr(cls, k):
            cur = getattr(cls, k)
            try:
                if isinstance(cur, bool):
                    val = v.lower() in {"1", "true", "yes", "y", "on"}
                elif isinstance(cur, int):
                    val = int(v)
                elif isinstance(cur, float):
                    val = float(v)
                else:
                    val = v
            except Exception:
                val = v
            setattr(cls, k, val)

def _apply_overrides_to_globals(ns, overrides):
    for k, v in overrides.items():
        if k in ns:
            cur = ns[k]
            try:
                if isinstance(cur, bool):
                    val = v.lower() in {"1", "true", "yes", "y", "on"}
                elif isinstance(cur, int):
                    val = int(v)
                elif isinstance(cur, float):
                    val = float(v)
                else:
                    val = v
            except Exception:
                val = v
            ns[k] = val
import matplotlib.image as mpimg
import os

# English comment: see script logic.
# English comment: see script logic.
IMG_DIR = "/mnt/data/fpdetec_V2/bench_results/modality_plots/"
OUTPUT_NAME = "/mnt/data/fpdetec_V2/bench_results/modality_plots/Figure_2_Y_10Fold_Importance.png"

# English comment: see script logic.
# English comment: see script logic.
IMG_FILES = [f"fold{i}_modality_importance_labeled.png" for i in range(1, 11)]

# English comment: see script logic.
def merge_images():
    # English comment: see script logic.
    # English comment: see script logic.
    # English comment: see script logic.
    # English comment: see script logic.
    fig, axes = plt.subplots(5, 2, figsize=(16, 30))
    axes = axes.flatten()

    print("Merging images...")
    for i, ax in enumerate(axes):
        if i < len(IMG_FILES):
            img_path = os.path.join(IMG_DIR, IMG_FILES[i])
            
            if os.path.exists(img_path):
                img = mpimg.imread(img_path)
                ax.imshow(img)
                ax.axis('off')  # English comment removed for consistency.
            else:
                print(f"Warning: {img_path} not found!")
                ax.axis('off')
        else:
            ax.axis('off')  # English comment removed for consistency.
    
    # English comment: see script logic.
    # English comment: see script logic.
    # English comment: see script logic.
    plt.subplots_adjust(wspace=0.0, hspace=0.0, left=0.01, right=0.99, bottom=0.01, top=0.99)
    
    # English comment: see script logic.
    # English comment: see script logic.
    # English comment: see script logic.
    plt.savefig(OUTPUT_NAME, dpi=800, bbox_inches='tight', pad_inches=0.05)
    print(f"✅ Merged image saved to: {OUTPUT_NAME}")

if __name__ == "__main__":
    _ovr = _parse_set_overrides()
    _apply_overrides_to_globals(globals(), _ovr)
    merge_images()