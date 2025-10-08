from transformers import AutoTokenizer

from models.modeling_qwen3_dsa import Qwen3DSAForCausalLM

if __name__ == "__main__":
    model_name = "kaizheng9105/Qwen3-4B-Instruct-2507-DSA"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = Qwen3DSAForCausalLM.from_pretrained(
        model_name, dtype="auto", device_map="auto"
    )
    print("✅ Model loaded successfully!")

    prompt = "Give me a short introduction to sparse attention."
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        use_cache=True,
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()

    content = tokenizer.decode(output_ids, skip_special_tokens=True)
    print("content:", content)
