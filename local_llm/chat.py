from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def main():
    print(f"Loading model {MODEL_NAME} ... (first time may take a while)")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    model.eval()
    print("Model loaded. Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower().strip() == "exit":
            break

        inputs = tokenizer(user_input, return_tensors="pt")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=120,
                do_sample=True,
                temperature=0.7,
            )

        reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\nAssistant: {reply}\n")


if __name__ == "__main__":
    main()
