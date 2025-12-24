from transformers import AutoTokenizer

# Load your tokenizer
tokenizer = AutoTokenizer.from_pretrained("yash3056/Llama-3.2-1B-imdb")

# Test with a sample text to see the tokenization pattern
test_text = "I just finished watching this movie and am disappointed"

# Method 1: See the actual tokens
tokens = tokenizer.tokenize(test_text)
print("Tokens:", tokens)

# Method 2: See token IDs and convert back
encoded = tokenizer.encode(test_text, add_special_tokens=False)
decoded_tokens = [tokenizer.decode([token_id]) for token_id in encoded]
print("\nDecoded tokens:")
for i, token in enumerate(decoded_tokens):
    print(f"{i}: '{token}'")

# Method 3: Check what happens with subwords
test_subword = "uninteresting"
tokens_subword = tokenizer.tokenize(test_subword)
print(f"\nTokenization of '{test_subword}':", tokens_subword)

# Method 4: Check the tokenizer type
print(f"\nTokenizer class: {type(tokenizer)}")
print(f"Tokenizer model: {tokenizer.__class__.__name__}")