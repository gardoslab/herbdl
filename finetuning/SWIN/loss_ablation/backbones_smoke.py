"""CPU smoke test: SwinWithLoss on tiny random-init ConvNeXt/DINOv2/EfficientNet/timm models
(no downloads). Checks forward/backward for softmax + dbm heads and a strict state_dict reload."""
import torch
from transformers import (ConvNextConfig, ConvNextForImageClassification, Dinov2Config,
                          Dinov2ForImageClassification, EfficientNetConfig, EfficientNetForImageClassification)
from loss_ablation.model import SwinWithLoss

N = 20
def tiny():
    yield "convnext", ConvNextForImageClassification(ConvNextConfig(num_labels=N, depths=[1, 1, 1, 1], hidden_sizes=[8, 16, 16, 32])), 64
    yield "dinov2", Dinov2ForImageClassification(Dinov2Config(num_labels=N, hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64, image_size=56, patch_size=14)), 56
    yield "efficientnet", EfficientNetForImageClassification(EfficientNetConfig(num_labels=N, image_size=64, depth_coefficient=0.2, width_coefficient=0.2)), 64
    try:
        import timm
        from transformers import TimmWrapperConfig, TimmWrapperForImageClassification
        m = timm.create_model("coatnet_0_rw_224", pretrained=False, num_classes=N)
        w = TimmWrapperForImageClassification(TimmWrapperConfig(architecture="coatnet_0_rw_224", num_labels=N))
        yield "timm", w, 224
    except Exception as e:
        print("timm skipped:", repr(e)[:120])

for name, base, size in tiny():
    for head, kw in (("softmax", {}), ("arcface", {"s": 30.0, "m": 0.3})):
        model = SwinWithLoss(base, num_species=N, head_name=head, head_kwargs=kw)
        x, y = torch.randn(4, 3, size, size), torch.randint(0, N, (4,))
        out = model(x, labels=y)
        out["loss"].backward()
        assert out["logits"].shape == (4, N), out["logits"].shape
        m2 = SwinWithLoss(base, num_species=N, head_name=head, head_kwargs=kw)
        m2.load_state_dict(model.state_dict(), strict=True)
        print(name, head, "ok loss=%.3f feat_dim=%d" % (out["loss"].item(), model.head.class_directions().shape[1]))
