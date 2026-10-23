"""
model.py
--------
Small, FPGA-friendly CNN for 10-class electronic component classification.

Architecture
------------
  Input : RGB 64x64
  Conv block 1 : Conv2d(3->16, 3, pad=1) -> ReLU -> MaxPool2d(2)   => 16 x 32 x 32
  Conv block 2 : Conv2d(16->32, 3, pad=1) -> ReLU -> MaxPool2d(2)  => 32 x 16 x 16
  Conv block 3 : Conv2d(32->64, 3, pad=1) -> ReLU -> MaxPool2d(2)  => 64 x 8 x 8
  AdaptiveAvgPool2d(1)                                               => 64 x 1 x 1
  Flatten -> Linear(64, 10)                                          => 10 logits

Design notes
------------
- No BatchNorm, no Dropout, no skip connections — keeps the HLS
  synthesis straightforward for PYNQ-Z2 deployment.
- Output is raw logits (no softmax).  Use nn.CrossEntropyLoss for
  training, which fuses log-softmax + NLL internally.
- AdaptiveAvgPool reduces the spatial dims to 1x1 regardless of small
  input-size changes, but the target is fixed 64x64.
"""

import torch
import torch.nn as nn

# ── Stable class ordering (alphabetical, matches ImageFolder) ─────────────
CLASS_NAMES = [
    "capacitor",
    "diode",
    "fuse",
    "inductor",
    "integrated_circuit",
    "led",
    "potentiometer",
    "push_button",
    "resistor",
    "transistor",
]

NUM_CLASSES = len(CLASS_NAMES)   # 10
INPUT_SIZE  = 64                 # H = W = 64


class ComponentCNN(nn.Module):
    """Lightweight 3-block CNN for electronic component classification."""

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
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

        # ── Pooling → classifier ──────────────────────────────────────────
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
        x = self.classifier(x)     # (B, 10)
        return x


# ══════════════════════════════════════════════════════════════════════════
# Quick self-test (not training)
# ══════════════════════════════════════════════════════════════════════════

def _count_parameters(model: nn.Module) -> int:
    """Total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    model = ComponentCNN().to(device)
    print(f"Model  : {model.__class__.__name__}")
    print(f"Params : {_count_parameters(model):,}")
    print()
    print(model)
    print()

    # Smoke test with a random batch
    batch_size = 2
    dummy = torch.randn(batch_size, 3, INPUT_SIZE, INPUT_SIZE, device=device)
    with torch.no_grad():
        logits = model(dummy)

    expected_shape = (batch_size, NUM_CLASSES)
    actual_shape   = tuple(logits.shape)

    print(f"Input  shape : {tuple(dummy.shape)}")
    print(f"Output shape : {actual_shape}")
    print(f"Expected     : {expected_shape}")

    if actual_shape == expected_shape:
        print("\nPASS  Output shape matches expected shape.")
    else:
        print(f"\nFAIL  Expected {expected_shape}, got {actual_shape}.")

    print(f"\nClass names ({len(CLASS_NAMES)}):")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {i}: {name}")


if __name__ == "__main__":
    main()
