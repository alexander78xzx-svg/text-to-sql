import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import GPT2Config, GPT2LMHeadModel
from torch.optim import AdamW

data = torch.load("train_data.pt")

dataset = TensorDataset(
    data["input_ids"],
    data["attention_mask"],
    data["labels"]
)

batch_size = 8
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model = GPT2LMHeadModel.from_pretrained("gpt2")
model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)
epochs = 3
print("\n--- Starting Training ---")
for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    for step, batch in enumerate(loader):
        b_input_ids = batch[0].to(device)
        b_attention_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=b_input_ids,
            attention_mask=b_attention_mask,
            labels=b_labels
        )
        loss = outputs.loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

        if step % 100 == 0 and step > 0:
            avg_step_loss = total_loss / step
            print(f"Epoch {epoch + 1}/{epochs} | Step {step}/{len(loader)} | Current Loss: {loss.item():.4f} | Running Avg: {avg_step_loss:.4f}")

    epoch_avg = total_loss / len(loader)
    print(f"\n>>> Epoch {epoch + 1} Complete | Average Loss: {epoch_avg:.4f} <<<\n")

torch.save(model.state_dict(), "trained_weights.pt")
print("Training complete.\n")