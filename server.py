import os
import json
import time
import threading
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import uvicorn
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torchvision.models as models
from PIL import Image
import torchvision.transforms as transforms

# =============================================================================
# 1. ORION LMM ARCHITECTURE (Flattened for Cloud Deployment)
# =============================================================================

# NanoGPT Configuration
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.1
block_size = 64

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2,-1) * (C ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class OrionGPTModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

class FastVisionEncoder(nn.Module):
    def __init__(self, embed_dim=64):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.projection = nn.Linear(512, embed_dim)
        
    def forward(self, images):
        features = self.backbone(images)
        features = features.view(features.size(0), -1)
        visual_tokens = self.projection(features)
        return visual_tokens.unsqueeze(1) 

class OrionLMM(nn.Module):
    def __init__(self, vocab_size, embed_dim=64):
        super().__init__()
        self.vision_encoder = FastVisionEncoder(embed_dim)
        self.language_model = OrionGPTModel(vocab_size)
        
    def forward(self, images, text_idx, targets=None):
        B, T = text_idx.shape
        visual_tokens = self.vision_encoder(images)
        text_emb = self.language_model.token_embedding_table(text_idx)
        combined_emb = torch.cat([visual_tokens, text_emb], dim=1)
        pos_emb = self.language_model.position_embedding_table(torch.arange(T+1, device=text_idx.device))
        x = combined_emb + pos_emb
        x = self.language_model.blocks(x)
        x = self.language_model.ln_f(x)
        logits = self.language_model.lm_head(x)
        
        if targets is None:
            loss = None
        else:
            B, T_plus_1, C = logits.shape
            logits_reshaped = logits[:, :-1, :].reshape(B*T, C)
            targets_reshaped = targets.view(B*T)
            loss = torch.nn.functional.cross_entropy(logits_reshaped, targets_reshaped)
            
        return logits, loss


# =============================================================================
# 2. FASTAPI SERVER & CONTINUOUS TRAINING NODE
# =============================================================================

app = FastAPI(title="ORION LMM Training Nexus (Cloud)")

# Paths (Relative to the server root)
MEMORY_FILE = "server_memory_queue.json"
BRAIN_FILE = "orion_lmm_brain_v2.pth"
DATASET_FILE = "dataset_lmm.json"

memory_queue = []

@app.post("/upload_memory")
async def upload_memory(file: UploadFile = File(...)):
    content = await file.read()
    new_logs = json.loads(content.decode('utf-8'))
    global memory_queue
    memory_queue.extend(new_logs)
    return {"status": "success", "added": len(new_logs)}

@app.get("/download_brain")
async def download_brain():
    if os.path.exists(BRAIN_FILE):
        return FileResponse(BRAIN_FILE, media_type='application/octet-stream', filename="orion_lmm_brain_v2.pth")
    return {"error": "Brain not found"}

@app.get("/")
async def health_check():
    return {"status": "Online", "brain_size_kb": os.path.getsize(BRAIN_FILE) / 1024 if os.path.exists(BRAIN_FILE) else 0}

def continuous_training_loop():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[Training Node] Started on {device.upper()}")
    
    if not os.path.exists(DATASET_FILE):
        print("[Training Node] ERROR: Dataset missing.")
        return
        
    with open(DATASET_FILE, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    all_text = "".join([d["description"] for d in dataset])
    chars = sorted(list(set(all_text)))
    vocab_size = len(chars)
    stoi = { ch:i for i,ch in enumerate(chars) }
    
    model = OrionLMM(vocab_size=vocab_size, embed_dim=64)
    if os.path.exists(BRAIN_FILE):
        model.load_state_dict(torch.load(BRAIN_FILE, map_location=device, weights_only=True))
    model.to(device)
    model.train()
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    global memory_queue
    while True:
        if len(memory_queue) > 0:
            print(f"[Training Node] Training on {len(memory_queue)} new memories...")
            batch = memory_queue[:10]
            memory_queue = memory_queue[10:]
            
            total_loss = 0
            for log in batch:
                prompt = log.get("prompt", "")
                
                # Basic Teacher AI Simulation: Enforce proper response format
                target_text = f"คุณ: {prompt} โอไรออน: กำลังเรียนรู้และอัปเดตข้อมูลเกี่ยวกับ {prompt} ครับ"
                
                encoded = [stoi.get(c, 0) for c in target_text]
                if len(encoded) < 2:
                    continue
                    
                idx = torch.tensor([encoded[:-1]], dtype=torch.long).to(device)
                targets = torch.tensor([encoded[1:]], dtype=torch.long).to(device)
                
                img = Image.new('RGB', (224, 224), color='black')
                img_tensor = transform(img).unsqueeze(0).to(device)
                
                optimizer.zero_grad()
                logits, loss = model(img_tensor, idx, targets=targets)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
            print(f"[Training Node] Batch Avg Loss: {total_loss/len(batch):.4f}")
            torch.save(model.state_dict(), BRAIN_FILE)
            print("[Training Node] Brain updated & saved!")
            
        time.sleep(10)

if __name__ == "__main__":
    threading.Thread(target=continuous_training_loop, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
