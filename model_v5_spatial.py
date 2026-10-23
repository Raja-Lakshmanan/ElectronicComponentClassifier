"""
model_v5_spatial.py
-------------------
ComponentCNNSpatial: FPGA-friendly CNN with 2x2 spatial pooling and 2-stage MLP classifier.

Architecture:
  Input        : RGB 64x64
  Conv Block 1 : Conv2d(3->16, 3, pad=1) -> ReLU -> MaxPool2d(2)   => 16 x 32 x 32
  Conv Block 2 : Conv2d(16->32, 3, pad=1) -> ReLU -> MaxPool2d(2)  => 32 x 16 x 16
  Conv Block 3 : Conv2d(32->64, 3, pad=1) -> ReLU -> MaxPool2d(2)  => 64 x 8 x 8
  Spatial Pool : AdaptiveAvgPool2d((2, 2))                           => 64 x 2 x 2
  Flatten      : Flatten(1)                                          => 256 features
  Classifier   : Linear(256, 64) -> ReLU -> Linear(64, 10)          => 10 raw logits

Key Architectural Rationale:
  - Keeps all 3 convolutional blocks identical to V4 ComponentCNN.
  - Replaces 1x1 Global Average Pooling with 2x2 Spatial Pooling.
  - Retains 4 spatial quadrants (top-left, top-right, bottom-left, bottom-right),
    enabling the network to disambiguate axial lead symmetry (resistors/inductors)
    from bottom-pinned clusters (push buttons) and 3-terminal packages (potentiometers).
  - Trainable parameters: exactly 40,682 (vs 24,234 in V4 baseline).
  - Hardware footprint: 40.7 KB in INT8, comfortably fitting inside PYNQ-Z2 on-chip BRAM (<7%).
"""

from typing import Dict, List, Tuple
import torch
import torch.nn as nn

# ── Stable class ordering (identical to V4 ImageFolder and model.py) ───────
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
PERMITTED_CLASSES = CLASS_NAMES[:9]  # 9 permitted deployment classes (transistor masked)


class ComponentCNNSpatial(nn.Module):
    """ComponentCNN with 2x2 spatial pooling and 2-stage MLP classifier."""

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()

        # ── Feature extractor (Identical to V4 ComponentCNN) ───────────────
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

        # ── 2x2 Spatial pooling (preserves 4 spatial quadrants) ─────────────
        self.pool = nn.AdaptiveAvgPool2d(output_size=(2, 2))

        # ── 2-stage MLP Classifier (Linear 256->64 -> ReLU -> Linear 64->10) ──
        self.classifier = nn.Sequential(
            nn.Linear(64 * 2 * 2, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
            x : Tensor of shape (B, 3, 64, 64)
        Returns:
            Tensor of shape (B, num_classes) — raw logits.
        """
        x = self.features(x)       # (B, 64, 8, 8)
        x = self.pool(x)           # (B, 64, 2, 2)
        x = torch.flatten(x, 1)    # (B, 256)
        x = self.classifier(x)     # (B, 10)
        return x


# Alias for drop-in compatibility
ComponentCNN = ComponentCNNSpatial


# ══════════════════════════════════════════════════════════════════════════
# Verification & Self-Test Helper
# ══════════════════════════════════════════════════════════════════════════

def get_layer_breakdown(model: nn.Module) -> List[Dict]:
    """Provides a detailed layer-by-layer breakdown of parameters and operations."""
    breakdown = []
    
    # Conv1
    c1 = model.features[0]
    p1 = sum(p.numel() for p in c1.parameters())
    # 16 output channels * 32 * 32 spatial * 27 MACs per window = 442,368 MACs (post-pool equivalent)
    breakdown.append({
        "layer": "Conv Block 1",
        "operation": "Conv2d(3, 16, k=3, p=1) + ReLU + MaxPool2d(2)",
        "output_shape": "(1, 16, 32, 32)",
        "params": p1,
        "macs": 16 * 32 * 32 * (3 * 3 * 3),
    })

    # Conv2
    c2 = model.features[3]
    p2 = sum(p.numel() for p in c2.parameters())
    breakdown.append({
        "layer": "Conv Block 2",
        "operation": "Conv2d(16, 32, k=3, p=1) + ReLU + MaxPool2d(2)",
        "output_shape": "(1, 32, 16, 16)",
        "params": p2,
        "macs": 32 * 16 * 16 * (16 * 3 * 3),
    })

    # Conv3
    c3 = model.features[6]
    p3 = sum(p.numel() for p in c3.parameters())
    breakdown.append({
        "layer": "Conv Block 3",
        "operation": "Conv2d(32, 64, k=3, p=1) + ReLU + MaxPool2d(2)",
        "output_shape": "(1, 64, 8, 8)",
        "params": p3,
        "macs": 64 * 8 * 8 * (32 * 3 * 3),
    })

    # Spatial Pool
    breakdown.append({
        "layer": "Spatial Pooling",
        "operation": "AdaptiveAvgPool2d((2, 2))",
        "output_shape": "(1, 64, 2, 2)",
        "params": 0,
        "macs": 64 * 8 * 8,  # Reductions
    })

    # Flatten
    breakdown.append({
        "layer": "Flatten",
        "operation": "Flatten(start_dim=1)",
        "output_shape": "(1, 256)",
        "params": 0,
        "macs": 0,
    })

    # Linear 1
    fc1 = model.classifier[0]
    p_fc1 = sum(p.numel() for p in fc1.parameters())
    breakdown.append({
        "layer": "Dense Hidden",
        "operation": "Linear(256, 64) + ReLU",
        "output_shape": "(1, 64)",
        "params": p_fc1,
        "macs": 256 * 64,
    })

    # Linear 2
    fc2 = model.classifier[2]
    p_fc2 = sum(p.numel() for p in fc2.parameters())
    breakdown.append({
        "layer": "Classifier Output",
        "operation": "Linear(64, 10)",
        "output_shape": "(1, 10)",
        "params": p_fc2,
        "macs": 64 * 10,
    })

    return breakdown


def print_architecture_summary() -> Tuple[int, int]:
    """Prints architecture specification, parameter counts, and MACs."""
    model = ComponentCNNSpatial()
    breakdown = get_layer_breakdown(model)

    print("=" * 82)
    print("      ComponentCNNSpatial (V5) Architecture & Hardware Profile")
    print("=" * 82)
    print(f"  {'Layer / Block':<20} {'Operation':<40} {'Output Shape':<16} {'Params':>8} {'MACs':>12}")
    print("-" * 82)

    total_params = 0
    total_macs = 0
    for b in breakdown:
        total_params += b["params"]
        total_macs += b["macs"]
        print(f"  {b['layer']:<20} {b['operation']:<40} {b['output_shape']:<16} {b['params']:>8,d} {b['macs']:>12,d}")

    print("-" * 82)
    print(f"  {'TOTAL':<20} {'Pipeline Total':<40} {'(1, 10)':<16} {total_params:>8,d} {total_macs:>12,d}")
    print("=" * 82)

    # Verification checks
    assert total_params == 40682, f"Expected 40,682 params, found {total_params}"
    assert breakdown[-2]["params"] == 16448, f"Expected 16,448 params in Linear1, found {breakdown[-2]['params']}"
    assert breakdown[-1]["params"] == 650, f"Expected 650 params in Linear2, found {breakdown[-1]['params']}"

    return total_params, total_macs


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    model = ComponentCNNSpatial().to(device)

    # Dummy forward pass
    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE, device=device)
    with torch.no_grad():
        out = model(dummy)

    print(f"Input shape  : {tuple(dummy.shape)}")
    print(f"Output shape : {tuple(out.shape)}")
    assert out.shape == (1, 10), f"Expected (1, 10), got {out.shape}"
    print("PASS: Forward pass successful.\n")

    print_architecture_summary()
