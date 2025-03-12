import torch
from slidingwindow import create_dataloader_v1

with open("../the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

vocab_size = 50257
output_dim = 256

token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

max_length = 4
dataloader = create_dataloader_v1(raw_text, batch_size=8, max_lenght=max_length, stride=max_length,
                                  shuffle=False)

data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Token IDs:\n", inputs)
print("\nInputs shape:\n", inputs.shape)

token_embeddings = token_embedding_layer(inputs)

context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))

print("Tokens embedding shape: ", token_embeddings.shape)
print("Pos embedding shape: ", pos_embeddings.shape)

input_embbedings = token_embeddings + pos_embeddings
print("Input embedding shape: ", input_embbedings.shape)
