import os
import json
import random
import importlib
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoFeatureExtractor,
    AutoTokenizer,
    AutoConfig,
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ─── CONFIGURATION ───────────────────────────────────────────────────────────────
MODEL_ID     = "liuhaotian/llava-v1.5-7b"
IMAGE_DIR    = "/mnt/data1/raiyan/breast_cancer/datasets/vindr/images_png"
REPORT_DIR   = "/mnt/data1/Nafiz/MammoGen-RAG/vindr/ground_truth_reports"
OUTPUT_DIR   = f"{MODEL_ID.replace('/', '_')}_finetune"
EVAL_OUT_DIR = f"evaluated-vindr/{MODEL_ID.replace('/', '_')}_finetune"
BATCH_SIZE   = 8
ACCUM_STEPS  = 4
LR           = 2e-4
EPOCHS       = 3
MAX_LEN      = 512
SEED         = 42

# ─── UTILITIES ─────────────────────────────────────────────────────────────────
def get_file_pairs(image_dir, report_dir):
    pairs = []
    for root, _, files in os.walk(image_dir):
        for fname in files:
            if not fname.lower().endswith('.png'):
                continue
            img = os.path.join(root, fname)
            rpt = os.path.join(report_dir, fname.replace('.png', '.json'))
            if os.path.exists(rpt):
                pairs.append((img, rpt))
    return pairs

class VLMDataset(Dataset):
    def __init__(self, pairs, vision_processor, tokenizer, max_length):
        self.pairs = pairs
        self.vision_processor = vision_processor
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, rpt_path = self.pairs[idx]
        image = Image.open(img_path).convert('RGB')
        pixel_values = self.vision_processor(images=image, return_tensors='pt').pixel_values[0]
        text = json.dumps(json.load(open(rpt_path, 'r', encoding='utf-8')))
        toks = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
            use_fast=False
        )
        return {
            'input_ids': toks.input_ids[0],
            'attention_mask': toks.attention_mask[0],
            'pixel_values': pixel_values,
        }

# ─── MAIN ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # Prepare data splits
    random.seed(SEED)
    pairs = get_file_pairs(IMAGE_DIR, REPORT_DIR)
    random.shuffle(pairs)
    n = len(pairs)
    t0, t1 = int(0.6 * n), int(0.8 * n)
    train_pairs, val_pairs, test_pairs = pairs[:t0], pairs[t0:t1], pairs[t1:]

    # Load feature extractor and tokenizer
    vision_processor = AutoFeatureExtractor.from_pretrained(
        'openai/clip-vit-large-patch14-336',
        torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        use_fast=False
    )

    # Load config and quantization settings
    config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.float16
    )

    # Load LLaVA model via remote code with QLoRA
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.float16,
        device_map='auto',
        trust_remote_code=True
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
            lora_dropout=0.05,
            bias='none',
            task_type='CAUSAL_LM'
        )
    )

    # Create datasets
    train_ds = VLMDataset(train_pairs, vision_processor, tokenizer, MAX_LEN)
    val_ds = VLMDataset(val_pairs, vision_processor, tokenizer, MAX_LEN)

        # Collate function
    def collate_fn(batch):
        ids = torch.stack([b['input_ids'] for b in batch])
        am = torch.stack([b['attention_mask'] for b in batch])
        pv = torch.stack([b['pixel_values'] for b in batch])
        lbl = ids.clone()
        lbl[lbl == tokenizer.pad_token_id] = -100
        return {'input_ids': ids, 'attention_mask': am, 'pixel_values': pv, 'labels': lbl}


    # Setup Trainer with early stopping
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=OUTPUT_DIR,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=ACCUM_STEPS,
            learning_rate=LR,
            fp16=True,
            logging_steps=50,
            evaluation_strategy='epoch',
            save_strategy='epoch',
            load_best_model_at_end=True,
            metric_for_best_model='eval_loss',
            greater_is_better=False,
            save_total_limit=2,
            num_train_epochs=EPOCHS
        ),
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
    )

    # Fine-tune
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    vision_processor.save_pretrained(OUTPUT_DIR)

    # Evaluation: generate for all images
    os.makedirs(EVAL_OUT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    for img_path, _ in pairs:
        image_id = os.path.basename(img_path).replace('.png', '')
        img = Image.open(img_path).convert('RGB')
        inputs = vision_processor(images=img, return_tensors='pt').to(device)
        with torch.no_grad():
            gen = model.generate(pixel_values=inputs.pixel_values, max_new_tokens=MAX_LEN)
        text = tokenizer.decode(gen[0], skip_special_tokens=True)
        try:
            output = json.loads(text[text.find('{'):])
        except:
            output = {'generated': text}
        with open(os.path.join(EVAL_OUT_DIR, image_id + '.json'), 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=4)
    print('✅ Evaluation complete.')
