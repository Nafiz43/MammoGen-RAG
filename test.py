import json
from transformers import AutoConfig

model_id = "liuhaotian/llava-v1.5-7b"
cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

# Dump *all* the config fields so you can see what’s there
print(json.dumps(cfg.to_dict(), indent=2))
