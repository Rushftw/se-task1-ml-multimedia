from transformers import pipeline

def main():
    classifier = pipeline(
        "sentiment-analysis",
        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    )

    print("Sentiment analysis demo. Type 'exit' to quit.")
    while True:
        text = input("Enter text: ")
        if text.lower().strip() == "exit":
            break
        result = classifier(text)[0]
        print(f"Label: {result['label']}, score: {result['score']:.4f}")

if __name__ == "__main__":
    main()
