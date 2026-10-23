"""
predict_image_v5.py
-------------------
Manual single-image inference script for ElectronicComponentClassifier using the
trained V5-Spatial model checkpoint (checkpoints/v5_spatial_best_model.pth).

Hardware & Architecture Reference:
  - Architecture : ComponentCNNSpatial (model_v5_spatial.py) — 40,682 parameters
  - Pooling      : AdaptiveAvgPool2d((2, 2)) -> 256 spatial features (4 quadrants)
  - Classifier   : Linear(256, 64) -> ReLU -> Linear(64, 10)
  - Input        : 64x64 RGB
  - Normalization: mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5) => range [-1.0, 1.0]
  - Checkpoint   : checkpoints/v5_spatial_best_model.pth (Epoch 49, val F1 0.6308)
  - Post-Process : Mask transistor logit at index 9 to -infinity before prediction
                   (prediction constrained strictly to 9 permitted component classes)

Usage:
  python predict_image_v5.py "path\\to\\component_image.jpg"
  python predict_image_v5.py "path\\to\\component_image.jpg" --no-display
"""

import argparse
from pathlib import Path
import sys
import time
from typing import List, Tuple

from PIL import Image
import torch
import torchvision.transforms as transforms

# ── Architecture Import ────────────────────────────────────────────────────
from model_v5_spatial import ComponentCNNSpatial, CLASS_NAMES

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = PROJECT_DIR / "checkpoints" / "v5_spatial_best_model.pth"
DEFAULT_PREVIEW_OUT = PROJECT_DIR / "prediction_preview_v5.png"

# Normalization constants used during V5 training
NORM_MEAN = (0.5, 0.5, 0.5)
NORM_STD = (0.5, 0.5, 0.5)
INPUT_SIZE = 64
TRANSISTOR_INDEX = 9


def parse_arguments() -> argparse.Namespace:
    # Print user-friendly guidance if invoked without arguments
    if len(sys.argv) == 1 or (len(sys.argv) == 2 and sys.argv[1] in ("-h", "--help")):
        if len(sys.argv) == 2:
            pass
        else:
            print("=" * 76)
            print("  ComponentCNNSpatial (V5) — Single-Image Inference")
            print("=" * 76)
            print("  Usage:")
            print("    python predict_image_v5.py \"path\\to\\component_image.jpg\"")
            print()
            print("  Optional Flags:")
            print("    --no-display     Suppress interactive window (headless / automation)")
            print("    --save-preview   Save prediction plot (default: prediction_preview_v5.png)")
            print("    --checkpoint     Path to custom checkpoint file")
            print("    --device         Force compute device ('cpu' or 'cuda')")
            print("=" * 76)
            sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Manual single-image inference for electronic components using V5-Spatial ComponentCNN."
    )
    parser.add_argument(
        "image_path",
        type=str,
        help="Path to the input image file (JPEG, PNG, BMP, etc.).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CHECKPOINT),
        help=f"Path to V5 checkpoint file (default: {DEFAULT_CHECKPOINT})",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda"],
        help="Inference device ('cpu' or 'cuda'). Defaults to CUDA if available, else CPU.",
    )
    parser.add_argument(
        "--save-preview",
        type=str,
        default=str(DEFAULT_PREVIEW_OUT),
        help="Path to save preview visualization figure (default: prediction_preview_v5.png)",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Suppress graphical window display (useful for headless environments).",
    )
    return parser.parse_args()


def load_v5_model(
    checkpoint_path: Path, device: torch.device
) -> Tuple[ComponentCNNSpatial, List[str], List[str]]:
    """
    Loads ComponentCNNSpatial from model_v5_spatial.py, loads weights from V5 checkpoint,
    verifies class metadata and transistor index, and sets model to eval mode.
    """
    if not checkpoint_path.exists():
        print(f"\nERROR: Checkpoint file not found: '{checkpoint_path}'")
        print("Please verify that checkpoints/v5_spatial_best_model.pth exists.\n")
        sys.exit(1)

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except Exception as exc:
        print(f"\nERROR: Failed to read checkpoint '{checkpoint_path}'.")
        print(f"Details: {exc}\n")
        sys.exit(1)

    # Validate checkpoint metadata
    class_names = checkpoint.get("class_names")
    if not class_names or not isinstance(class_names, list):
        print(f"\nERROR: Checkpoint missing valid 'class_names' list.\n")
        sys.exit(1)

    if len(class_names) != 10:
        print(
            f"\nERROR: Expected 10 classes in V5 checkpoint metadata, found {len(class_names)}: {class_names}\n"
        )
        sys.exit(1)

    if class_names[TRANSISTOR_INDEX] != "transistor":
        print(
            f"\nERROR: Checkpoint class mapping mismatch! Expected 'transistor' at index {TRANSISTOR_INDEX}, "
            f"found '{class_names[TRANSISTOR_INDEX]}'.\n"
        )
        sys.exit(1)

    # 9 permitted classes (indices 0..8)
    permitted_classes = class_names[:TRANSISTOR_INDEX]

    # Instantiate model and load state dictionary
    model = ComponentCNNSpatial(num_classes=len(class_names))
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()  # Set model to evaluation mode

    return model, class_names, permitted_classes


def load_and_preprocess_image(
    image_path: Path,
) -> Tuple[Image.Image, torch.Tensor]:
    """
    Validates file existence, loads original image using Pillow, converts to RGB,
    resizes to 64x64, converts to tensor, and applies training normalization:
    mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5).
    """
    if not image_path.exists():
        print(f"\nERROR: Image file not found: '{image_path}'")
        print("Please check the path and filename and try again.\n")
        sys.exit(1)

    if not image_path.is_file():
        print(f"\nERROR: Specified path is a directory, not an image file: '{image_path}'")
        print("Please provide a path to a specific image file.\n")
        sys.exit(1)

    try:
        with Image.open(image_path) as img:
            img.load()
            original_img = img.copy()
    except Exception as exc:
        print(f"\nERROR: Unable to open or parse '{image_path}' as an image file.")
        print(f"Details: {exc}")
        print("Supported formats include: JPEG (.jpg, .jpeg), PNG (.png), BMP (.bmp), WEBP (.webp).\n")
        sys.exit(1)

    # Convert to RGB (handles RGBA, grayscale, indexed palette)
    original_rgb = original_img.convert("RGB")

    # Resize to 64x64 using Bilinear interpolation matching training pipeline
    resized_rgb = original_rgb.resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.BILINEAR)

    # Convert to tensor and apply normalization: mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
    transform_pipeline = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])

    tensor = transform_pipeline(resized_rgb).unsqueeze(0)  # Shape: (1, 3, 64, 64)
    return original_rgb, tensor


def render_display(
    original_img: Image.Image,
    image_path: Path,
    predicted_class: str,
    predicted_confidence: float,
    top3_predictions: List[Tuple[str, float]],
    inference_time_ms: float,
    save_preview_path: Path,
    no_display: bool,
) -> None:
    """
    Renders visual display with original image and confidence bar chart.
    Saves preview image to prediction_preview_v5.png and optionally displays interactive GUI window.
    """
    try:
        import matplotlib
        if no_display:
            matplotlib.use("Agg")  # Non-interactive headless backend
        import matplotlib.pyplot as plt

        fig, (ax_img, ax_bar) = plt.subplots(
            1, 2, figsize=(10, 4.8), gridspec_kw={"width_ratios": [1, 1.25]}
        )
        fig.suptitle(
            "ComponentCNNSpatial (V5) — Single-Image Inference",
            fontsize=13,
            fontweight="bold",
            y=0.97,
        )

        # Left panel: Original input image
        ax_img.imshow(original_img)
        ax_img.set_title(
            f"Input: {image_path.name}\n"
            f"Original Size: {original_img.width}x{original_img.height} px (RGB)\n\n"
            f"Predicted: {predicted_class.upper()} ({predicted_confidence * 100:.1f}%)",
            fontsize=10,
            pad=10,
        )
        ax_img.axis("off")

        # Right panel: Horizontal bar chart for Top 3 permitted classes
        class_labels = [item[0] for item in reversed(top3_predictions)]
        scores_pct = [item[1] * 100.0 for item in reversed(top3_predictions)]
        colors = [
            "#2ecc71" if i == len(class_labels) - 1 else "#3498db"
            for i in range(len(class_labels))
        ]

        bars = ax_bar.barh(class_labels, scores_pct, color=colors, height=0.5)
        ax_bar.set_xlim(0, 105)
        ax_bar.set_xlabel("Model Softmax Confidence (%)", fontsize=10)
        ax_bar.set_title(
            f"Top 3 Permitted Classes (Transistor Masked)\nForward Pass Time: {inference_time_ms:.2f} ms",
            fontsize=10,
            pad=10,
        )
        ax_bar.grid(axis="x", linestyle="--", alpha=0.5)

        for bar, pct in zip(bars, scores_pct):
            ax_bar.text(
                pct + 1.5,
                bar.get_y() + bar.get_height() / 2,
                f"{pct:.2f}%",
                va="center",
                ha="left",
                fontsize=10,
                fontweight="bold",
            )

        # Footer disclaimer
        fig.text(
            0.5,
            0.02,
            "Notice: Softmax scores represent relative model confidence within the 9 permitted classes, "
            "not guaranteed real-world correctness.",
            ha="center",
            fontsize=8,
            style="italic",
            color="#555555",
        )

        plt.tight_layout(rect=[0, 0.05, 1, 0.94])

        if save_preview_path:
            plt.savefig(save_preview_path, dpi=150, bbox_inches="tight")
            print(f"  Visual Preview Saved : {save_preview_path.resolve()}")

        if not no_display:
            try:
                plt.show()
            except Exception as disp_exc:
                print(f"  [Note] Interactive display skipped: {disp_exc}")

        plt.close(fig)

    except Exception as exc:
        print(f"  [Note] Matplotlib visualization notice: {exc}")


def main() -> None:
    args = parse_arguments()

    image_path = Path(args.image_path)
    checkpoint_path = Path(args.checkpoint)
    save_preview_path = Path(args.save_preview) if args.save_preview else None

    # Determine device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device_name = f"{device.type.upper()}"
    if device.type == "cuda":
        device_name += f" ({torch.cuda.get_device_name(0)})"

    # Step 1: Load trained V5-Spatial checkpoint and model
    model, class_names, permitted_classes = load_v5_model(checkpoint_path, device)

    # Step 2: Load and preprocess image
    original_img, input_tensor = load_and_preprocess_image(image_path)

    # Step 3: Run inference using torch.inference_mode()
    with torch.inference_mode():
        tensor = input_tensor.to(device)

        # Accurate timing of model forward pass
        if device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        logits = model(tensor)  # Shape: (1, 10)

        if device.type == "cuda":
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        inference_time_ms = (end_time - start_time) * 1000.0

        # Mask transistor logit at index 9 to -infinity
        masked_logits = logits.clone()
        masked_logits[:, TRANSISTOR_INDEX] = -float("inf")

        # Softmax normalized over the 9 permitted classes (indices 0..8)
        permitted_logits = masked_logits[:, :TRANSISTOR_INDEX]
        softmax_scores = torch.softmax(permitted_logits, dim=-1)[0]  # Shape: (9,)

        top3_scores, top3_indices = torch.topk(softmax_scores, k=3)

    # Build predictions list
    top3_predictions = [
        (permitted_classes[idx.item()], score.item())
        for idx, score in zip(top3_indices, top3_scores)
    ]
    predicted_class, predicted_confidence = top3_predictions[0]

    # Display terminal summary and confidence notices
    print("=" * 76)
    print("       ComponentCNNSpatial (V5) Single-Image Inference Report")
    print("=" * 76)
    print(f"  Input Image File     : {image_path.resolve()}")
    print(f"  Original Resolution  : {original_img.width} x {original_img.height} pixels (RGB)")
    print(f"  Model Input Tensor   : 1 x 3 x {INPUT_SIZE} x {INPUT_SIZE} (Normalized mean={NORM_MEAN[0]}, std={NORM_STD[0]})")
    print(f"  Architecture         : ComponentCNNSpatial (40,682 params, 2x2 spatial pool)")
    print(f"  Compute Device       : {device_name}")
    print(f"  Model Checkpoint     : {checkpoint_path.name}")
    print(f"  Forward Pass Time    : {inference_time_ms:.2f} ms")
    print("-" * 76)
    print(f"  PREDICTED COMPONENT  : {predicted_class.upper()}")
    print(f"  TOP CONFIDENCE SCORE : {predicted_confidence * 100:.2f}%")
    print("-" * 76)
    print("  Top 3 Permitted Class Predictions (Transistor Masked):")
    for rank, (name, conf) in enumerate(top3_predictions, start=1):
        pct = conf * 100.0
        bar = "=" * int(round(pct / 4.0))
        print(f"    Rank {rank}: {name:<20} {pct:6.2f}%  [{bar:<25}]")
    print("-" * 76)
    print("  Deployment Masking Configuration:")
    print(f"    [Index {TRANSISTOR_INDEX}] {class_names[TRANSISTOR_INDEX]} -> Logit masked to -inf (EXCLUDED)")
    print("    Prediction domain constrained strictly to 9 permitted classes:")
    print("    (" + ", ".join(permitted_classes) + ")")
    print("=" * 76)
    print("  CONFIDENCE INTERPRETATION NOTICE:")
    print("  * Softmax scores represent relative model confidence within the 9")
    print("    permitted component classes.")
    print("  * They are NOT guaranteed mathematical probabilities of real-world correctness.")
    print("  * Out-of-distribution images, unlisted components, or distorted inputs")
    print("    will still receive high confidence scores for the closest trained class.")
    print("=" * 76)

    # Render Matplotlib display and save preview image
    render_display(
        original_img=original_img,
        image_path=image_path,
        predicted_class=predicted_class,
        predicted_confidence=predicted_confidence,
        top3_predictions=top3_predictions,
        inference_time_ms=inference_time_ms,
        save_preview_path=save_preview_path,
        no_display=args.no_display,
    )


if __name__ == "__main__":
    main()
