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
from openai import OpenAI
from github import Github

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
        # Tiny CNN to save RAM on Render Free Tier (Replaces ResNet18)
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )
        # Input: 224x224 -> Conv(112) -> MaxPool(56) -> Conv(28) -> MaxPool(14)
        # Flattened: 32 * 14 * 14 = 6272
        self.projection = nn.Linear(6272, embed_dim)
        
    def forward(self, images):
        features = self.backbone(images)
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

@app.on_event("startup")
async def startup_event():
    threading.Thread(target=continuous_training_loop, daemon=True).start()

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
        try:
            model.load_state_dict(torch.load(BRAIN_FILE, map_location=device, weights_only=True))
            print("[Training Node] Successfully loaded previous brain weights.")
        except Exception as e:
            print(f"[Training Node] Could not load previous weights (Architecture changed? Starting fresh). Error: {e}")
            
    model.to(device)
    model.train()
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    global memory_queue
    dream_counter = 0
    
    # Initialize 9arm Teacher AI once
    try:
        ai_client = OpenAI(
            api_key=os.environ.get("NINEARM_API_KEY", "sk-DvdsqHV_M5uxfQm3wWPWNA"),
            base_url="https://gateway.9arm.co/v1"
        )
    except Exception as e:
        print(f"[Teacher AI] Init error: {e}")
        ai_client = None

    while True:
        target_texts = []
        is_dream = False
        
        if len(memory_queue) > 0:
            print(f"[Training Node] 🧑‍💻 Training on {len(memory_queue)} new memories from User...")
            batch = memory_queue[:10]
            memory_queue = memory_queue[10:]
            
            for log in batch:
                prompt = log.get("prompt", "")
                target_text = f"คุณ: {prompt} โอไรออน: ระบบรับทราบครับ"
                if ai_client:
                    try:
                        response = ai_client.chat.completions.create(
                            model="qwen3.6-35b-a3b",
                            messages=[
                                {"role": "system", "content": "ช่วยแต่งประโยคตอบกลับที่สั้น กระชับ เป็นภาษาไทย เพื่อใช้สอน AI ตอบแค่ตัวข้อความ ไม่ต้องมีคำอธิบาย"},
                                {"role": "user", "content": prompt}
                            ],
                            max_tokens=50
                        )
                        teacher_reply = response.choices[0].message.content.strip()
                        target_text = f"คุณ: {prompt} โอไรออน: {teacher_reply}"
                    except Exception as e:
                        print(f"[Teacher AI] Error: {e}")
                target_texts.append(target_text)
                
        else:
            # --- DREAM MODE (FAST BATCH Self-Play) ---
            is_dream = True
            dream_counter += 1
            if ai_client:
                try:
                    # Request 5 Q&A pairs at once to speed up data collection
                    resp = ai_client.chat.completions.create(
                        model="qwen3.6-35b-a3b",
                        messages=[
                            {"role": "system", "content": "คุณคือ AI สร้างชุดข้อมูลสั้นๆ ให้แต่งคำถามที่คนทั่วไปชอบถาม AI พร้อมคำตอบสั้นๆ กระชับ สร้างมา 5 คู่ โดยให้รูปแบบคือ Q: คำถาม A: คำตอบ"}
                        ],
                        max_tokens=300
                    )
                    content = resp.choices[0].message.content.strip()
                    
                    # Parse the Q&A pairs
                    lines = content.split('\n')
                    current_q = ""
                    for line in lines:
                        line = line.strip()
                        if line.startswith("Q:") or line.startswith("คำถาม:"):
                            current_q = line.split(":", 1)[1].strip()
                        elif (line.startswith("A:") or line.startswith("คำตอบ:")) and current_q:
                            ans = line.split(":", 1)[1].strip()
                            target_text = f"คุณ: {current_q} โอไรออน: {ans}"
                            print(f"[Dream Mode 💭] AI เรียนรู้: {target_text}")
                            target_texts.append(target_text)
                            current_q = ""
                            
                except Exception as e:
                    print(f"[Dream Mode] Error generating batch data: {e}")
                    
        # --- EXECUTE TRAINING ---
        if len(target_texts) > 0:
            total_loss = 0
            for target_text in target_texts:
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
                
            print(f"[Training Node] Batch Avg Loss: {total_loss/len(target_texts):.4f}")
            torch.save(model.state_dict(), BRAIN_FILE)
            
            # --- GITHUB PERSISTENT STORAGE (Non-blocking) ---
            # Push to GitHub immediately if it's user data, or every 10 dream cycles
            if not is_dream or dream_counter % 10 == 0:
                github_token = os.environ.get("GITHUB_TOKEN")
                if github_token:
                    def _upload_to_github():
                        try:
                            g = Github(github_token)
                            repo = g.get_repo("atom2558/orion-cloud")
                            
                            # Read current brain
                            with open(BRAIN_FILE, "rb") as f:
                                content = f.read()
                                
                            try:
                                contents = repo.get_contents("orion_lmm_brain_v2.pth")
                                repo.update_file(contents.path, "Auto-update brain", content, contents.sha)
                                print("[GitHub Storage] ☁️ Successfully updated Brain on GitHub!")
                            except:
                                repo.create_file("orion_lmm_brain_v2.pth", "Auto-create brain", content)
                                print("[GitHub Storage] ☁️ Successfully created Brain on GitHub!")
                        except Exception as e:
                            print(f"[GitHub Storage] ❌ Upload failed: {e}")
                            
                    # Fire and forget thread so it doesn't slow down the main training loop
                    threading.Thread(target=_upload_to_github, daemon=True).start()
                else:
                    print("[GitHub Storage] ⚠️ GITHUB_TOKEN not set. Brain is not saved permanently.")
            
        # Sleep less in Dream Mode to collect data faster (2 seconds instead of 15)
        sleep_time = 2 if is_dream else 5
        time.sleep(sleep_time)

if __name__ == "__main__":
    threading.Thread(target=continuous_training_loop, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
