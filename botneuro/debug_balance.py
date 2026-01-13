import ccxt
import asyncio
import os
from dotenv import load_dotenv

# Load API Key dari file .env
load_dotenv()
api_key = os.getenv("BINANCE_API_KEY")
secret = os.getenv("BINANCE_SECRET")

async def test_connection():
    print("--- DEBUGGING SALDO: METODE DIRECT-FUTURES ---")
    
    if not api_key or not secret:
        print("[ERROR] API KEY atau SECRET tidak ditemukan di .env!")
        return

    # Inisialisasi menggunakan CCXT Standar (Sync) untuk stabilitas jaringan
    # Metode ini terbukti berhasil menembus blokir ISP/VPN Anda
    exchange = ccxt.binance({
        'apiKey': api_key,
        'secret': secret,
        'enableRateLimit': True,
        'options': {
            'defaultType': 'future', 
        },
        # Mematikan verifikasi SSL untuk menghindari error 'Handshake Failed' pada VPN tertentu
        'verify': False 
    })

    try:
        print("1. Menghubungkan ke server Binance Futures...")
        
        # Menggunakan thread agar fungsi synchronous tidak membekukan program
        # Memanggil endpoint fapiPrivateV2GetAccount (Direct)
        account_data = await asyncio.to_thread(exchange.fapiPrivateV2GetAccount)
        
        # Mengambil data saldo dari response json
        total_balance = float(account_data.get('totalWalletBalance', 0))
        unrealized_pnl = float(account_data.get('totalUnrealizedProfit', 0))
        available_balance = float(account_data.get('availableBalance', 0))

        print(f"\n[HASIL KONEKSI]")
        print(f"Total Wallet Balance : ${total_balance:,.2f}")
        print(f"Unrealized PNL       : ${unrealized_pnl:,.2f}")
        print(f"Available to Trade   : ${available_balance:,.2f}")
        
        if total_balance > 0:
            print("\n✅ STATUS: SIAP TRADING!")
            print("Gunakan pengaturan yang sama pada file loader.py Anda.")
        else:
            print("\n⚠️ STATUS: KONEKSI OK, TAPI SALDO KOSONG.")
            print("Pastikan saldo berada di 'USDT-M Futures', bukan di Spot.")

    except Exception as e:
        print(f"\n[ERROR] Gagal mendapatkan data: {e}")
            
    # CCXT Sync tidak memerlukan await close()

if __name__ == "__main__":
    try:
        asyncio.run(test_connection())
    except KeyboardInterrupt:
        pass