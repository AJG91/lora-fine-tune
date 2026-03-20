from peft import LoraConfig, get_peft_model

def apply_lora(model):
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["c_attn"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, config)