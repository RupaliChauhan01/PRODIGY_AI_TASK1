from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate(prompt, model_dir="fine_tuned_gpt2"):
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
    model = GPT2LMHeadModel.from_pretrained(model_dir)

    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    output = model.generate(
        input_ids,
        max_length=100,
        num_return_sequences=1,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.9
    )

    text = tokenizer.decode(output[0], skip_special_tokens=True)
    print("\nGenerated Text:\n")
    print(text)

if __name__ == "__main__":
    user_input = input("Enter a prompt: ")
    generate(user_input)