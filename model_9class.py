"""
model_9class.py
---------------
ComponentCNN modified for 9-class electronic component classification,
completely excluding the transistor class.

Classes (9, alphabetical):
  0: capacitor
  1: diode
  2: fuse
  3: inductor
  4: integrated_circuit
  5: led
  6: potentiometer
  7: push_button
  8: resistor

Architecture
------------
  Input : RGB 64x64
  Conv block 1 : Conv2d(3->16, 3, pad=1) -> ReLU -> MaxPool2d(2)   => 16 x 32 x 32 (448 params)
  Conv block 2 : Conv2d(16->32, 3, pad=1) -> ReLU -> MaxPool2d(2)  => 32 x 16 x 16 (4,640 params)
  Conv block 3 : Conv2d(32->64, 3, pad=1) -> ReLU -> MaxPool2d(2)  => 64 x 8 x 8   (18,496 params)
  AdaptiveAvgPool2d(1)                                               => 64 x 1 x 1
  Flatten -> Linear(64, 9)                                           => 9 logits     (585 params)

Total Parameters: 24,169
Model Size (FP32 weights): ~94.4 KB
"""

from typing import Dict, List, Tuple
import torch
import torch.nn as nn

# ── Fixed 9-class alphabetical order (transistor excluded) ─────────────────
CLASS_NAMES_9: List[str] = [
    "capacitor",
    "diode",
    "fuse",
    "inductor",
    "integrated_circuit",
    "led",
    "potentiometer",
    "push_button",
    "resistor",
]

NUM_CLASSES_9: int = len(CLASS_NAMES_9)   # 9
INPUT_SIZE: int    = 64                  # H = W = 64

# Fixed class-to-index mapping
CLASS_TO_IDX_9: Dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES_9)}
IDX_TO_CLASS_9: Dict[int, str] = {idx: name for idx, name in enumerate(CLASS_NAMES_9)}


class ComponentCNN9(nn.Module):
    """Lightweight 3-block CNN for 9-class electronic component classification."""

    def __init__(self, num_classes: int = NUM_CLASSES_9) -> None:
        super().__init__()

        # ── Feature extractor ──────────────────────────────────────────────
        self.features = nn.Sequential(
            # Block 1: 3 -> 16, spatial 64 -> 32
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Block 2: 16 -> 32, spatial 32 -> 16
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Block 3: 32 -> 64, spatial 16 -> 8
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        # ── Pooling -> classifier ──────────────────────────────────────────
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor of shape (B, 3, 64, 64)

        Returns
        -------
        Tensor of shape (B, num_classes) — raw logits.
        """
        x = self.features(x)       # (B, 64, 8, 8)
        x = self.pool(x)           # (B, 64, 1, 1)
        x = torch.flatten(x, 1)    # (B, 64)
        x = self.classifier(x)     # (B, 9)
        return x


def count_parameters(model: nn.Module) -> Tuple[int, float]:
    """
    Returns (trainable_parameter_count, size_in_kilobytes_fp32).
    """
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_kb = (total_params * 4) / 1024.0  # FP32: 4 bytes per float
    return total_params, size_kb


# ══════════════════════════════════════════════════════════════════════════
# Verification Self-Test
# ══════════════════════════════════════════════════════════════════════════
def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("  ComponentCNN-9 (Nine-Class Architecture)")
    print("=" * 70)
    print(f"Device        : {device}")

    model = ComponentCNN9().to(device)
    params, size_kb = count_parameters(model)

    print(f"Model Class   : {model.__class__.__name__}")
    print(f"Parameters    : {params:,}")
    print(f"Size (FP32)   : {size_kb:.2f} KB")
    print(f"Num Classes   : {NUM_CLASSES_9}")
    print(f"Input Shape   : (B, 3, {INPUT_SIZE}, {INPUT_SIZE})")
    print("-" * 70)
    print("Layer Details :")
    for name, param in model.named_parameters():
        print(f"  {name:25s} : shape={list(param.shape)!s:18s} params={param.numel():,}")
    print("-" * 70)

    # Smoke test forward pass with dummy tensor
    batch_size = 4
    dummy = torch.randn(batch_size, 3, INPUT_SIZE, INPUT_SIZE, device=device)
    with torch.no_grad():
        logits = model(dummy)

    expected_shape = (batch_size, NUM_CLASSES_9)
    actual_shape = tuple(logits.shape)
    print(f"Dummy Input   : {tuple(dummy.shape)}")
    print(f"Output Logits : {actual_shape}")
    assert actual_shape == expected_shape, f"Expected {expected_shape}, got {actual_shape}"
    print("PASS: Forward pass smoke test successful!")

    print("\nFixed 9-Class Mapping:")
    for idx, name in enumerate(CLASS_NAMES_9):
        print(f"  [{idx}] {name}")
    print("=" * 70)


if __name__ == "__main__":
    main()
