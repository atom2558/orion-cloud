import os
import sys
from supabase import create_client, Client

# =========================================================================
# ใส่ตั้งค่าตรงนี้ให้เหมือนในเว็บ Render ครับ
SUPABASE_URL = "https://xxxxxxxxx.supabase.co"
SUPABASE_KEY = "sb_secret_xxxxxxxxxxxxxxxxxxxxxxxxxx"
# =========================================================================

if SUPABASE_URL == "https://xxxxxxxxx.supabase.co":
    print("❌ กรุณาแก้ไฟล์ archiver.py เพื่อใส่ URL และ Key ของ Supabase ก่อนครับ!")
    sys.exit(1)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def download_brain():
    print("🧠 กำลังดาวน์โหลดไฟล์สมองจาก Supabase...")
    try:
        bucket_name = "brains"
        file_path = "orion_lmm_brain_v2.pth"
        
        # ดาวน์โหลดไฟล์จาก Storage
        data = supabase.storage.from_(bucket_name).download(file_path)
        with open("orion_lmm_brain_v2_backup.pth", "wb") as f:
            f.write(data)
            
        print("✅ ดาวน์โหลดสมองสำเร็จ! เซฟไว้ในชื่อ orion_lmm_brain_v2_backup.pth")
    except Exception as e:
        print(f"❌ โหลดสมองไม่สำเร็จ (ไฟล์อาจจะยังไม่มีในระบบ): {e}")

if __name__ == "__main__":
    print("🚀 [Orion Data Archiver] กำลังเชื่อมต่อ Supabase...")
    download_brain()
    print("🎉 ดำเนินการเสร็จสิ้น!")
