import requests, time
for i in range(5):
    r = requests.get("https://orion-server-lm9e.onrender.com/api/stats")
    d = r.json()
    print(f"Progress: {d['progress']}% | Status: {d['status']} | QA: {d['total_qa_learned']} | Loss: {d['last_loss']}")
    time.sleep(2)
