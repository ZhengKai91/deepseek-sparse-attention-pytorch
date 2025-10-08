from transformers import AutoTokenizer

from models.modeling_qwen3_dsa import load_qwen3_dsa_from_pretrained

if __name__ == "__main__":
    base_model_name_or_path = "Qwen/Qwen3-4B-Instruct-2507"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
    custom_model = load_qwen3_dsa_from_pretrained(base_model_name_or_path)

    # trainner.train(custom_model, tokenizer)

    save_path = "/root/data/Qwen3-4B-Instruct-2507-DSA"
    custom_model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
