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
from supabase import create_client, Client

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

# =============================================================================
# 2. FASTAPI SERVER & ON-DEMAND TRAINING NODE
# =============================================================================

import logging
import time
import os
import json
import threading
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image

class EndpointFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.getMessage().find("/api/stats") == -1

logging.getLogger("uvicorn.access").addFilter(EndpointFilter())

app = FastAPI(title="ORION LMM Training Nexus (Cloud)")

# Paths
BRAIN_FILE = "orion_lmm_brain_v2.pth"

# --- TRAINING STATS ---
training_stats = {
    "status": "Idle. Waiting for dataset...",
    "user_train_count": 0,
    "total_qa_learned": 0,
    "last_loss": 0.0,
    "loss_history": [],
    "recent_lessons": [],
    "brain_size_kb": 0,
    "uptime_start": time.time(),
    "progress": 0,
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_dataset_task(dataset):
    global training_stats
    training_stats["status"] = "Preparing model..."
    training_stats["progress"] = 0
    
    # Vocabulary setup (Extended for Thai/English full support)
    all_text = "".join([d.get("q", "") + " " + d.get("a", "") for d in dataset])
    thai_chars = "กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮฤลฦะัาำิีึืุูเแโใไๅๆ็่้๊๋์ํ"
    english_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    numbers = "0123456789"
    symbols = " !\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~\n"
    all_text += thai_chars + english_chars + numbers + symbols
    
    chars = sorted(list(set(all_text)))
    vocab_size = len(chars)
    stoi = { ch:i for i,ch in enumerate(chars) }
    
    model = OrionLMM(vocab_size=vocab_size, embed_dim=64)
    if os.path.exists(BRAIN_FILE):
        try:
            model.load_state_dict(torch.load(BRAIN_FILE, map_location=device, weights_only=True))
        except Exception as e:
            print(f"Could not load previous weights: {e}")
            
    model.to(device)
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    total_items = len(dataset)
    
    for i, item in enumerate(dataset):
        q = item.get("q", "")
        a = item.get("a", "")
        target_text = f"คุณ: {q} โอไรออน: {a}"
        
        training_stats["status"] = f"Training item {i+1}/{total_items}"
        training_stats["progress"] = int(((i + 1) / total_items) * 100)
        
        filtered = [c for c in target_text if c in stoi]
        encoded = [stoi[c] for c in filtered]
        if len(encoded) < 2:
            continue
            
        encoded = [min(e, vocab_size - 1) for e in encoded]
            
        idx = torch.tensor([encoded[:-1]], dtype=torch.long).to(device)
        targets = torch.tensor([encoded[1:]], dtype=torch.long).to(device)
        
        img = Image.new('RGB', (224, 224), color='black')
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        optimizer.zero_grad()
        logits, loss = model(img_tensor, idx, targets=targets)
        loss.backward()
        optimizer.step()
        
        training_stats["last_loss"] = round(loss.item(), 4)
        training_stats["total_qa_learned"] += 1
        
        if len(training_stats["loss_history"]) >= 20:
            training_stats["loss_history"].pop(0)
        training_stats["loss_history"].append(training_stats["last_loss"])
        
        training_stats["recent_lessons"] = [target_text] + training_stats["recent_lessons"][:4]
        
    training_stats["progress"] = 100
    training_stats["status"] = "Saving Brain..."
    
    torch.save(model.state_dict(), BRAIN_FILE)
    
    # Upload to Supabase if configured
    try:
        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_KEY")
        if supabase_url and supabase_key:
            from supabase import create_client, Client
            supabase: Client = create_client(supabase_url, supabase_key)
            bucket_name = "orion_brain"
            file_path = "orion_lmm_brain_v2.pth"
            try: supabase.storage.create_bucket(bucket_name)
            except: pass
            with open(BRAIN_FILE, "rb") as f:
                supabase.storage.from_(bucket_name).upload(
                    file=os.path.abspath(BRAIN_FILE),
                    path=file_path,
                    file_options={"cacheControl": "3600", "upsert": "true"}
                )
    except Exception as e:
        print(f"Supabase upload failed: {e}")
        
    training_stats["status"] = "Training Complete! 🚀"

@app.post("/api/train")
async def train_dataset(files: list[UploadFile] = File(...)):
    if training_stats["progress"] > 0 and training_stats["progress"] < 100:
        return {"error": "Training already in progress."}
        
    all_dataset = []
    for file in files:
        content = await file.read()
        try:
            dataset = json.loads(content.decode('utf-8'))
            all_dataset.extend(dataset)
        except:
            return {"error": f"Invalid JSON file: {file.filename}"}
    
    # Start training in background
    threading.Thread(target=train_dataset_task, args=(all_dataset,), daemon=True).start()
    return {"status": "success", "message": "Training started."}

@app.get("/download_brain")
async def download_brain():
    if os.path.exists(BRAIN_FILE):
        return FileResponse(BRAIN_FILE, media_type='application/octet-stream', filename="orion_lmm_brain_v2.pth")
    return {"error": "Brain not found"}

@app.get("/api/stats")
async def get_stats():
    training_stats["brain_size_kb"] = round(os.path.getsize(BRAIN_FILE) / 1024, 1) if os.path.exists(BRAIN_FILE) else 0
    training_stats["uptime_seconds"] = int(time.time() - training_stats["uptime_start"])
    return training_stats

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return """
<!DOCTYPE html>
<html lang="th">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ORION AI - Training Studio</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: 'Inter', sans-serif;
    background: #0a0a1a;
    color: #e0e0ff;
    min-height: 100vh;
  }
  .bg-glow {
    position: fixed; top: -50%; left: -50%; width: 200%; height: 200%;
    background: radial-gradient(circle at 30% 40%, rgba(100,60,255,0.08) 0%, transparent 50%),
                radial-gradient(circle at 70% 60%, rgba(0,200,255,0.06) 0%, transparent 50%);
    z-index: 0; animation: pulse 8s ease-in-out infinite;
  }
  @keyframes pulse { 0%,100% { opacity: 0.6; } 50% { opacity: 1; } }
  .container { position: relative; z-index: 1; max-width: 1100px; margin: 0 auto; padding: 30px 20px; }
  .header { text-align: center; margin-bottom: 35px; }
  .header h1 {
    font-size: 2.2em; font-weight: 700;
    background: linear-gradient(135deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 5px;
  }
  .header .sub { color: #8888aa; font-size: 0.9em; }
  
  .upload-box {
    background: rgba(255,255,255,0.03); border: 2px dashed rgba(167,139,250,0.5);
    border-radius: 16px; padding: 40px; text-align: center; margin-bottom: 25px;
    transition: all 0.3s;
  }
  .upload-box:hover { background: rgba(167,139,250,0.05); border-color: #a78bfa; }
  input[type=file] { display: none; }
  .btn {
    background: linear-gradient(135deg, #8b5cf6, #3b82f6);
    color: white; border: none; padding: 12px 24px; border-radius: 8px;
    font-size: 1em; font-weight: 600; cursor: pointer; transition: transform 0.2s;
  }
  .btn:hover { transform: translateY(-2px); box-shadow: 0 4px 15px rgba(139,92,246,0.4); }
  
  .progress-container {
    width: 100%; background: rgba(255,255,255,0.1); border-radius: 10px;
    margin-top: 20px; overflow: hidden; display: none;
  }
  .progress-bar {
    height: 8px; background: linear-gradient(90deg, #a78bfa, #34d399);
    width: 0%; transition: width 0.3s ease;
  }
  
  .status-badge {
    display: inline-block; padding: 4px 14px; border-radius: 20px;
    font-size: 0.8em; font-weight: 600; margin-top: 8px;
    background: rgba(52,211,153,0.15); color: #34d399; border: 1px solid rgba(52,211,153,0.3);
  }
  
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-bottom: 25px; }
  .card {
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px; padding: 22px; backdrop-filter: blur(10px);
  }
  .card .label { font-size: 0.75em; color: #8888aa; text-transform: uppercase; letter-spacing: 1px; }
  .card .value { font-size: 2em; font-weight: 700; margin-top: 6px; }
  .card .value.purple { color: #a78bfa; }
  .card .value.blue { color: #60a5fa; }
  .card .value.green { color: #34d399; }
  .card .value.amber { color: #fbbf24; }
  
  .lessons-box {
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px; padding: 22px;
  }
  .lessons-box h3 { font-size: 1em; color: #34d399; margin-bottom: 15px; }
  .lesson-item {
    padding: 10px 14px; margin-bottom: 8px; border-radius: 10px;
    background: rgba(255,255,255,0.02); border-left: 3px solid #a78bfa;
    font-size: 0.85em; line-height: 1.5; word-break: break-all;
  }
</style>
</head>
<body>
<div class="bg-glow"></div>
<div class="container">
  <div class="header">
    <h1>🧠 ORION AI Training Studio</h1>
    <div class="sub">อัปโหลด Dataset เพื่อสอน AI ด้วยตัวเอง</div>
    <div id="statusBadge" class="status-badge">⏳ กำลังโหลด...</div>
  </div>
  
  <div class="upload-box" id="uploadBox">
    <h3 style="margin-bottom: 10px; color: #fff;">อัปโหลดไฟล์ Dataset (.json) ทีละหลายไฟล์ได้</h3>
    <p style="color: #888; font-size: 0.85em; margin-bottom: 20px;">รูปแบบ: [{"q": "คำถาม", "a": "คำตอบ"}]</p>
    <label class="btn" style="display:inline-block;">
      เลือกไฟล์และเริ่มเทรน
      <input type="file" id="fileInput" accept=".json" multiple onchange="uploadAndTrain(this)">
    </label>
    
    <div id="progressContainer" class="progress-container">
      <div id="progressBar" class="progress-bar"></div>
    </div>
    <div id="progressText" style="margin-top: 10px; font-size: 0.85em; color: #a78bfa;"></div>
  </div>

  <div class="grid">
    <div class="card"><div class="label">📊 Q&A Learned</div><div class="value blue" id="learnedCount">0</div></div>
    <div class="card"><div class="label">📉 Last Loss</div><div class="value amber" id="lastLoss">0.0000</div></div>
    <div class="card"><div class="label">🧠 Brain Size</div><div class="value green" id="brainSize">0 KB</div></div>
  </div>

  <div class="lessons-box">
    <h3>📚 กำลังเรียนรู้...</h3>
    <div id="lessonList"><div class="lesson-item" style="color:#555;">กำลังรอข้อมูล...</div></div>
  </div>
</div>

<script>
async function uploadAndTrain(input) {
  if (!input.files || input.files.length === 0) return;
  const formData = new FormData();
  for (let i = 0; i < input.files.length; i++) {
    formData.append("files", input.files[i]);
  }
  
  document.getElementById("progressContainer").style.display = "block";
  document.getElementById("progressText").innerText = "กำลังอัปโหลดและรวมไฟล์...";
  
  try {
    const res = await fetch('/api/train', { method: 'POST', body: formData });
    const data = await res.json();
    if (data.error) alert(data.error);
    else document.getElementById("progressText").innerText = "เริ่มการฝึกสอนแล้ว!";
  } catch (e) { alert("Upload failed: " + e); }
  
  input.value = "";
}

async function refresh() {
  try {
    const res = await fetch('/api/stats');
    const data = await res.json();
    
    document.getElementById('statusBadge').innerText = data.status;
    document.getElementById('learnedCount').innerText = data.total_qa_learned;
    document.getElementById('lastLoss').innerText = data.last_loss.toFixed(4);
    document.getElementById('brainSize').innerText = data.brain_size_kb + ' KB';
    
    if (data.progress > 0) {
      document.getElementById("progressContainer").style.display = "block";
      document.getElementById("progressBar").style.width = data.progress + "%";
      document.getElementById("progressText").innerText = "Training: " + data.progress + "%";
    }
    
    if (data.progress === 100) {
      setTimeout(() => { 
          document.getElementById("progressContainer").style.display = "none"; 
          document.getElementById("progressText").innerText = "เทรนเสร็จสมบูรณ์! พร้อมใช้งาน"; 
      }, 3000);
    }
    
    const lessonsDiv = document.getElementById('lessonList');
    if (data.recent_lessons.length > 0) {
      lessonsDiv.innerHTML = data.recent_lessons.map(l => `<div class="lesson-item">${l}</div>`).join('');
    }
  } catch (e) {}
}
setInterval(refresh, 2000);
refresh();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
