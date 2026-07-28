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
# 2. FASTAPI SERVER & CONTINUOUS TRAINING NODE
# =============================================================================

app = FastAPI(title="ORION LMM Training Nexus (Cloud)")

# Paths (Relative to the server root)
MEMORY_FILE = "server_memory_queue.json"
BRAIN_FILE = "orion_lmm_brain_v2.pth"
DATASET_FILE = "dataset_lmm.json"

memory_queue = []

# --- TRAINING STATS (for Dashboard) ---
training_stats = {
    "status": "Initializing...",
    "dream_count": 0,
    "user_train_count": 0,
    "total_qa_learned": 0,
    "last_loss": 0.0,
    "loss_history": [],
    "recent_lessons": [],
    "brain_size_kb": 0,
    "uptime_start": time.time(),
}

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

@app.get("/api/stats")
async def get_stats():
    training_stats["brain_size_kb"] = round(os.path.getsize(BRAIN_FILE) / 1024, 1) if os.path.exists(BRAIN_FILE) else 0
    training_stats["uptime_seconds"] = int(time.time() - training_stats["uptime_start"])
    return training_stats

from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return """
<!DOCTYPE html>
<html lang="th">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ORION AI - Training Dashboard</title>
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
  .header {
    text-align: center; margin-bottom: 35px;
  }
  .header h1 {
    font-size: 2.2em; font-weight: 700;
    background: linear-gradient(135deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 5px;
  }
  .header .sub { color: #8888aa; font-size: 0.9em; }
  .status-badge {
    display: inline-block; padding: 4px 14px; border-radius: 20px;
    font-size: 0.8em; font-weight: 600; margin-top: 8px;
    animation: glow 2s ease-in-out infinite;
  }
  .status-online { background: rgba(52,211,153,0.15); color: #34d399; border: 1px solid rgba(52,211,153,0.3); }
  .status-training { background: rgba(251,191,36,0.15); color: #fbbf24; border: 1px solid rgba(251,191,36,0.3); }
  @keyframes glow { 0%,100% { box-shadow: 0 0 8px rgba(52,211,153,0.2); } 50% { box-shadow: 0 0 20px rgba(52,211,153,0.4); } }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-bottom: 25px; }
  .card {
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px; padding: 22px; backdrop-filter: blur(10px);
    transition: transform 0.2s, border-color 0.3s;
  }
  .card:hover { transform: translateY(-3px); border-color: rgba(167,139,250,0.3); }
  .card .label { font-size: 0.75em; color: #8888aa; text-transform: uppercase; letter-spacing: 1px; }
  .card .value { font-size: 2em; font-weight: 700; margin-top: 6px; }
  .card .value.purple { color: #a78bfa; }
  .card .value.blue { color: #60a5fa; }
  .card .value.green { color: #34d399; }
  .card .value.amber { color: #fbbf24; }
  .loss-chart {
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px; padding: 22px; margin-bottom: 25px;
  }
  .loss-chart h3 { font-size: 1em; color: #a78bfa; margin-bottom: 15px; }
  .chart-area { height: 120px; display: flex; align-items: flex-end; gap: 3px; }
  .chart-bar {
    flex: 1; background: linear-gradient(to top, #a78bfa, #60a5fa); border-radius: 3px 3px 0 0;
    min-height: 2px; transition: height 0.5s ease;
  }
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
  .lesson-item:nth-child(even) { border-left-color: #60a5fa; }
  .footer { text-align: center; margin-top: 30px; color: #555; font-size: 0.75em; }
</style>
</head>
<body>
<div class="bg-glow"></div>
<div class="container">
  <div class="header">
    <h1>🧠 ORION AI Training Dashboard</h1>
    <div class="sub">ระบบติดตามการฝึกสอน AI แบบเรียลไทม์</div>
    <div id="statusBadge" class="status-badge status-online">⏳ กำลังโหลด...</div>
  </div>
  <div class="grid">
    <div class="card"><div class="label">💭 Dream Cycles</div><div class="value purple" id="dreamCount">-</div></div>
    <div class="card"><div class="label">📚 Q&A Learned</div><div class="value blue" id="qaCount">-</div></div>
    <div class="card"><div class="label">📉 Last Loss</div><div class="value amber" id="lastLoss">-</div></div>
    <div class="card"><div class="label">🧠 Brain Size</div><div class="value green" id="brainSize">-</div></div>
  </div>
  <div class="loss-chart">
    <h3>📈 Loss History (ยิ่งต่ำยิ่งฉลาด)</h3>
    <div class="chart-area" id="chart"></div>
  </div>
  <div class="lessons-box">
    <h3>📝 บทเรียนล่าสุด</h3>
    <div id="lessons"><div class="lesson-item">กำลังรอข้อมูล...</div></div>
  </div>
  <div class="footer">ORION LMM &mdash; Auto-refresh ทุก 5 วินาที</div>
</div>
<script>
async function refresh() {
  try {
    const r = await fetch('/api/stats');
    const d = await r.json();
    document.getElementById('dreamCount').textContent = d.dream_count;
    document.getElementById('qaCount').textContent = d.total_qa_learned;
    document.getElementById('lastLoss').textContent = d.last_loss.toFixed(4);
    document.getElementById('brainSize').textContent = d.brain_size_kb + ' KB';
    const badge = document.getElementById('statusBadge');
    badge.textContent = '🟢 ' + d.status;
    badge.className = 'status-badge ' + (d.status.includes('Dream') ? 'status-training' : 'status-online');
    // Chart
    const chart = document.getElementById('chart');
    const hist = d.loss_history || [];
    if (hist.length > 0) {
      const max = Math.max(...hist, 0.01);
      chart.innerHTML = hist.slice(-60).map(v =>
        '<div class="chart-bar" style="height:' + Math.max((v/max)*100, 2) + '%"></div>'
      ).join('');
    }
    // Lessons
    const box = document.getElementById('lessons');
    const lessons = d.recent_lessons || [];
    if (lessons.length > 0) {
      box.innerHTML = lessons.slice(-8).reverse().map(l => '<div class="lesson-item">' + l + '</div>').join('');
    }
  } catch(e) { console.error(e); }
}
refresh();
setInterval(refresh, 5000);
</script>
</body></html>
"""

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
        api_key = os.environ.get("NINEARM_API_KEY", "sk-DvdsqHV_M5uxfQm3wWPWNA")
        ai_client = OpenAI(
            api_key=api_key,
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
                                {"role": "user", "content": f"ตอบคำถามนี้เป็นภาษาไทยสั้นๆ กระชับ: {prompt} /no_think"}
                            ],
                            max_tokens=80
                        )
                        msg = response.choices[0].message
                        teacher_reply = msg.content
                        # Fallback: try reasoning_content if content is empty
                        if not teacher_reply and hasattr(msg, 'reasoning_content') and msg.reasoning_content:
                            teacher_reply = msg.reasoning_content
                        if teacher_reply:
                            # Strip <think> tags if present
                            import re
                            teacher_reply = re.sub(r'<think>.*?</think>', '', teacher_reply, flags=re.DOTALL).strip()
                            teacher_reply = teacher_reply.replace("/no_think", "").strip()
                            if teacher_reply:
                                target_text = f"คุณ: {prompt} โอไรออน: {teacher_reply}"
                    except Exception as e:
                        print(f"[Teacher AI] Error: {e}")
                target_texts.append(target_text)
                
        else:
            # --- DREAM MODE: ถามแล้วเรียนรู้ ---
            is_dream = True
            dream_counter += 1
            if ai_client:
                try:
                    resp = ai_client.chat.completions.create(
                        model="qwen3.6-35b-a3b",
                        messages=[
                            {"role": "user", "content": "สร้างชุดฝึกสอน AI ภาษาไทย 3 คู่ถาม-ตอบ ได้เลย รูปแบบแต่ละคู่:\nQ: คำถาม\nA: คำตอบ /no_think"}
                        ],
                        max_tokens=400
                    )
                    msg = resp.choices[0].message
                    content = msg.content
                    
                    # Debug: print raw response structure
                    print(f"[Dream Mode DEBUG] content type={type(content)}, content={repr(content)[:200]}")
                    if hasattr(msg, 'reasoning_content'):
                        print(f"[Dream Mode DEBUG] reasoning_content={repr(msg.reasoning_content)[:200]}")
                    
                    # Fallback: try reasoning_content if content is empty
                    if not content and hasattr(msg, 'reasoning_content') and msg.reasoning_content:
                        content = msg.reasoning_content
                    
                    if not content or not content.strip():
                        print("[Dream Mode] API returned empty content. Sleeping 10s...")
                        time.sleep(10)
                        continue
                    
                    # Strip <think> tags if present
                    import re
                    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
                    
                    if not content:
                        print("[Dream Mode] Content was only thinking tags. Sleeping 10s...")
                        time.sleep(10)
                        continue
                    
                    lines = content.split('\n')
                    current_q = ""
                    for line in lines:
                        line = line.strip()
                        if line.lower().startswith("q:"):
                            current_q = line.split(":", 1)[1].strip()
                        elif line.lower().startswith("a:") and current_q:
                            ans = line.split(":", 1)[1].strip()
                            ans = ans.replace("/no_think", "").strip()
                            target_text = f"คุณ: {current_q} โอไรออน: {ans}"
                            print(f"[Dream Mode 💭] AI เรียนรู้: {target_text}")
                            target_texts.append(target_text)
                            current_q = ""
                            
                except Exception as e:
                    print(f"[Dream Mode] Error: {e}. Sleeping 10s...")
                    time.sleep(10)

        # --- EXECUTE TRAINING ---
        if len(target_texts) > 0:
            total_loss = 0
            trained_count = 0
            for target_text in target_texts:
                # Filter: only use characters that exist in vocabulary
                filtered = [c for c in target_text if c in stoi]
                encoded = [stoi[c] for c in filtered]
                if len(encoded) < 2:
                    continue
                
                # Double-check: clamp indices to valid range
                encoded = [min(e, vocab_size - 1) for e in encoded]
                    
                idx = torch.tensor([encoded[:-1]], dtype=torch.long).to(device)
                targets = torch.tensor([encoded[1:]], dtype=torch.long).to(device)
                
                img = Image.new('RGB', (224, 224), color='black')
                img_tensor = transform(img).unsqueeze(0).to(device)
                
                optimizer.zero_grad()
                logits, loss = model(img_tensor, idx, targets=targets)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                trained_count += 1
            
            if trained_count > 0:
                avg_loss = total_loss / trained_count
                print(f"[Training Node] Batch Avg Loss: {avg_loss:.4f}")
                training_stats["last_loss"] = round(avg_loss, 4)
                training_stats["loss_history"].append(round(avg_loss, 4))
                if len(training_stats["loss_history"]) > 200:
                    training_stats["loss_history"] = training_stats["loss_history"][-200:]
                training_stats["total_qa_learned"] += trained_count
                if is_dream:
                    training_stats["dream_count"] = dream_counter
                    training_stats["status"] = f"Dream Mode 💭 (Cycle #{dream_counter})"
                else:
                    training_stats["user_train_count"] += trained_count
                    training_stats["status"] = f"Training from User 🧑‍💻"
                for t in target_texts[-8:]:
                    training_stats["recent_lessons"].append(t)
                training_stats["recent_lessons"] = training_stats["recent_lessons"][-20:]
                
            torch.save(model.state_dict(), BRAIN_FILE)
            
            # --- SUPABASE PERSISTENT STORAGE (Non-blocking) ---
            # Push to Supabase immediately if it's user data, or every 10 dream cycles
            if not is_dream or dream_counter % 10 == 0:
                supabase_url = os.environ.get("SUPABASE_URL")
                supabase_key = os.environ.get("SUPABASE_KEY")
                if supabase_url and supabase_key:
                    def _upload_to_supabase():
                        try:
                            supabase: Client = create_client(supabase_url, supabase_key)
                            bucket_name = "brains"
                            file_path = "orion_lmm_brain_v2.pth"
                            
                            # Ensure bucket exists (ignores if already exists)
                            try:
                                supabase.storage.create_bucket(bucket_name)
                            except:
                                pass
                                
                            with open(BRAIN_FILE, "rb") as f:
                                # Overwrite the file in the bucket
                                supabase.storage.from_(bucket_name).upload(
                                    file=os.path.abspath(BRAIN_FILE),
                                    path=file_path,
                                    file_options={"cacheControl": "3600", "upsert": "true"}
                                )
                            print("[Supabase Storage] ☁️ Successfully updated Brain on Supabase!")
                        except Exception as e:
                            print(f"[Supabase Storage] ❌ Upload failed: {e}")
                            
                    # Fire and forget thread so it doesn't slow down the main training loop
                    threading.Thread(target=_upload_to_supabase, daemon=True).start()
                else:
                    print("[Supabase Storage] ⚠️ SUPABASE_URL or SUPABASE_KEY not set. Brain is not saved permanently.")
            
        # Sleep less in Dream Mode to collect data faster (2 seconds instead of 15)
        sleep_time = 2 if is_dream else 5
        time.sleep(sleep_time)

if __name__ == "__main__":
    threading.Thread(target=continuous_training_loop, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
