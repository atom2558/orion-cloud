import requests, time

# 1. Upload and start training
print("=== Uploading dataset and starting training ===")
files = {'files': ('sample_dataset.json', open('sample_dataset.json', 'rb'), 'application/json')}
r = requests.post('https://orion-server-lm9e.onrender.com/api/train', files=files)
print(f"Upload response: {r.json()}")

# 2. Poll progress every 2 seconds
print("\n=== Monitoring training progress ===")
start_time = time.time()
for i in range(15):
    r = requests.get('https://orion-server-lm9e.onrender.com/api/stats')
    d = r.json()
    elapsed = round(time.time() - start_time, 1)
    print(f"[{elapsed}s] Progress: {d['progress']}% | Status: {d['status']} | QA: {d['total_qa_learned']} | Loss: {d['last_loss']}")
    if d['progress'] == 100:
        print(f"\n=== TRAINING COMPLETE in {elapsed} seconds! ===")
        break
    time.sleep(2)
