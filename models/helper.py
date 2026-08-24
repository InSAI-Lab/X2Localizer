from models import backbones


def get_backbone(backbone_arch, backbone_config):
    if "deit" not in backbone_arch.lower():
        raise NotImplementedError(f"Only DeiT is included in this release, got {backbone_arch!r}.")
    config = dict(backbone_config)
    config.setdefault("adapter", False)
    return backbones.DeiTWithX2Agg(model_name="deit_small", **config)


def get_aggregator(agg_arch, backbone_arch, backbone_config):
    if agg_arch != "x2" or "deit" not in backbone_arch.lower():
        raise NotImplementedError(
            f"Only agg_arch='x2' with DeiT-S is included, got {agg_arch!r}/{backbone_arch!r}."
        )
    return backbones.DeiTWithX2Agg(model_name="deit_small", **backbone_config)

