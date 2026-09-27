import yaml
from pathlib import Path

def load_policies_config(path="configs/policies.yaml"):
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    if "policies" not in cfg:
        raise ValueError(f"'policies' section missing in {path}")
    if "active_policy" not in cfg:
        raise ValueError(f"'active_policy' missing in {path}")
    if cfg["active_policy"] not in cfg["policies"]:
        raise ValueError(f"active_policy '{cfg['active_policy']}' not found")
    
    for name, body in cfg["policies"].items():
        if "type" not in body:
            raise ValueError(f"policy '{name}' has no 'type' field")
    
    return cfg
