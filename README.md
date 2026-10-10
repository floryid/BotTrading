# 🧠 AI Hedge Fund v2.0 — Bot Trading Bybit Perpetual Futures

Bot trading otomatis untuk **Bybit Linear Perpetual (USDT)** yang menggabungkan **LSTM, Transformer, Reinforcement Learning (Q-learning)** dan **voting multi-strategi**, lengkap dengan dashboard web modern (dark, responsif di HP), scanner pasar 80 koin, dan notifikasi Telegram.

| | |
|---|---|
| Exchange | Bybit — Futures / Linear Perpetual (USDT-margined) |
| Leverage | 5x, Cross Margin, One-Way Position |
| Koin yang diperdagangkan | BTC, ETH, SOL, BNB, HYPE (bisa diubah di `SYMBOLS`) |
| Timeframe | 15 menit (OHLCV) + harga real-time via WebSocket |
| Mode | **PAPER** (saldo virtual, default) dan **LIVE** (uang sungguhan) |

> ⚠️ **Peringatan risiko.** Trading futures dengan leverage berisiko tinggi; posisi bisa dilikuidasi dan seluruh modal bisa hilang. Software ini tidak menjamin profit dan bukan saran investasi. Jalankan **paper trading minimal 1–2 minggu** sebelum mempertimbangkan live, dan hanya gunakan dana yang siap hilang.

---

## Daftar isi

1. [Fitur](#fitur)
2. [Prasyarat](#prasyarat)
3. [Instalasi](#instalasi)
4. [Konfigurasi `.env`](#konfigurasi-env)
5. [Menjalankan aplikasi](#menjalankan-aplikasi)
6. [Dashboard](#dashboard)
7. [Setup Telegram](#setup-telegram)
8. [Setup API Bybit](#setup-api-bybit)
9. [Cara kerja bot](#cara-kerja-bot)
10. [Pindah ke mode LIVE](#pindah-ke-mode-live)
11. [File data](#file-data)
12. [Struktur proyek](#struktur-proyek)
13. [Testing](#testing)
14. [Troubleshooting](#troubleshooting)
15. [Keamanan](#keamanan)
16. [Lisensi pihak ketiga](#lisensi-pihak-ketiga)

---

## Fitur

**Trading**
- **Dual loop:** FAST loop (1 detik, harga WebSocket, cek SL/TP & entry cepat) dan SLOW loop (15 menit, analisis penuh LSTM + Transformer + RL + Smart Money + Market Regime).
- **Voting multi-strategi:** Trend Following, Mean Reversion, Whale Strategy, Scalper, plus bias Transformer. Whale Trap ≥ 2 membalik semua sinyal.
- **7 lapis filter** (confidence, regime, whale trap, smart money, WAIT, korelasi, total risk) dan **dynamic risk sizing**.
- **Circuit breaker:** bot berhenti total jika drawdown ≥ 20%.
- **Paper trading** dengan fee 0,06% dan slippage 0,1% per sisi, sehingga hasilnya realistis.
- **Backtest** EMA+RSI saat startup; mode LIVE menolak jalan jika strategi tidak viable.
- Recovery posisi saat bot di-restart, shutdown rapi (Ctrl+C / SIGTERM).

**Dashboard** (tab *Ringkasan*, *Market*, *Analisis AI*, *Riwayat & Performa*)
- Grafik **candlestick** (EMA 20/50, volume, garis entry/SL/TP, penanda trade, update real-time, timeframe 1m–4h) untuk koin apa pun.
- **Equity curve** modern dengan peak line dan Max Drawdown.
- **Scanner pasar** 80 pair teratas: filter, preset, favorit ⭐, heatmap, top gainers/losers, funding ekstrem, Fear & Greed.
- Statistik performa, riwayat trade dengan filter dan ekspor CSV.

**Notifikasi Telegram:** bot start, trade buka/tutup, filter skip, circuit breaker, retrain model, WebSocket putus.

---

## Prasyarat

- **Python 3.10+** (diuji di Python 3.13)
- Koneksi internet yang bisa mengakses Bybit (lihat [Troubleshooting](#troubleshooting) jika `bybit.com` diblokir)
- RAM ± 1 GB dan CPU biasa cukup (training model memakai CPU)
- *(Opsional)* akun Bybit + API key, bot Telegram, API key CryptoQuant / Coinglass

---

## Instalasi

```powershell
# 1. Masuk ke folder proyek
cd "d:\wamserver64\www\Bot Crypto"

# 2. (Disarankan) buat virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1          # Linux/macOS: source .venv/bin/activate

# 3. Install dependensi
pip install -r requirements.txt

# 4. Buat file konfigurasi
copy .env.example .env              # Linux/macOS: cp .env.example .env
```

Dependensi utama: `ccxt`, `pandas`, `numpy`, `torch`, `flask`, `websocket-client`, `requests`, `python-dotenv` (+ `pytest` untuk test). Instalasi `torch` cukup besar (± 200 MB untuk versi CPU).

---

## Konfigurasi `.env`

Semua pengaturan ada di file `.env` (di folder yang sama dengan `main.py`). **Jangan pernah membagikan atau meng-commit file ini.** Semua variabel bersifat opsional; tanpa `.env` bot berjalan di mode paper tanpa Telegram.

| Variabel | Default | Keterangan |
|---|---|---|
| `PAPER_TRADING` | `True` | `True` = saldo virtual. `False` = **LIVE, uang sungguhan** |
| `API_KEY`, `SECRET` | kosong | API key Bybit (hanya wajib untuk LIVE) |
| `TELEGRAM_TOKEN`, `CHAT_ID` | kosong | Notifikasi Telegram, lihat [Setup Telegram](#setup-telegram) |
| `CRYPTOQUANT_KEY` | kosong | Data whale inflow BTC. Tanpa key, komponen ini netral (0) |
| `COINGLASS_KEY` | kosong | Data likuidasi. Tanpa key, komponen ini netral (0) |
| `BYBIT_DOMAIN` | `bybit.com` | Ganti ke `bybit.nl` jika `bybit.com` diblokir jaringan Anda |
| `SYMBOLS` | `BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT,HYPE/USDT` | Koin yang **diperdagangkan** bot |
| `LEVERAGE` | `5` | Leverage (hanya dipakai saat setup futures LIVE) |
| `PAPER_BALANCE` | `10000` | Saldo awal paper trading (USDT) |
| `DASHBOARD_HOST` | `0.0.0.0` | `127.0.0.1` = hanya komputer ini; `0.0.0.0` = bisa diakses dari jaringan |
| `DASHBOARD_PORT` | `5000` | Port dashboard |
| `DASHBOARD_TOKEN` | kosong | Token akses dashboard (sangat disarankan jika diakses dari luar) |
| `SCANNER_TOP_N` | `80` | Jumlah pair teratas (menurut volume) yang dipantau di tab Market |
| `SCANNER_ANALYZE_N` | `50` | Dari itu, berapa yang dianalisis teknikal 1H |
| `SCANNER_TICKER_SECONDS` | `15` | Interval refresh harga scanner |
| `SCANNER_ANALYSIS_SECONDS` | `300` | Interval refresh analisis teknikal scanner |

Parameter strategi dan risiko (risk per trade, SL/TP ATR, batas drawdown, bobot voting, dll.) ada di [`bot/config.py`](bot/config.py).

| Parameter risiko | Nilai |
|---|---|
| Risk per trade | 1% equity (maks 2% setelah dynamic sizing) |
| Total risk semua posisi | maks 5% equity |
| Stop Loss / Take Profit | 1,5× ATR / 3,0× ATR (Risk:Reward 1:2) |
| Fee + slippage | 0,06% taker + 0,1% slippage per sisi |
| Circuit breaker | Drawdown ≥ 20% |
| Minimum confidence | 60 |

---

## Menjalankan aplikasi

### Langkah cepat (paper trading)

```powershell
cd "d:\wamserver64\www\Bot Crypto"
python main.py
```

Lalu buka **http://localhost:5000** di browser.

Saat pertama kali jalan, bot melakukan 9 tahap startup (± 1–2 menit karena training model). Contoh log normal:

```
[1/9] Paper trading: skip futures setup
[2/9] Backtesting EMA+RSI strategy
[3/9] RL agent ready
[4/9] Loading market data and training models
[5/9] Restoring positions
[6/9] Paper engine balance: $10000.00
[7/9] Correlation matrix
[8/9] Dashboard on http://0.0.0.0:5000
[9/9] WebSocket connected=True
--- slow loop ---
```

Hentikan dengan **Ctrl+C** (bot menyimpan posisi dan Q-table, lalu mengirim notifikasi Telegram).

### Opsi baris perintah

```powershell
python main.py --reset-paper   # hapus state paper (posisi, saldo, riwayat) lalu mulai dari awal
```
Gunakan setelah circuit breaker menghentikan bot, atau untuk mengulang paper trading dari $10.000. Q-table RL dan model LSTM tidak ikut terhapus.

### Menjalankan terus-menerus di Windows

Bot berhenti jika terminal ditutup. Pilihan:

- **Biarkan terminal terbuka** (paling sederhana), atau
- **Task Scheduler:** buat task *At startup* dengan program `python.exe`, argumen `main.py`, dan *Start in* diisi folder proyek.

### Menjalankan terus-menerus di Linux/VPS

```bash
# screen
screen -S bot
python main.py          # lepas dengan Ctrl+A lalu D; kembali dengan: screen -r bot

# atau nohup
nohup python main.py > data/stdout.log 2>&1 &
```

Contoh service `systemd` (`/etc/systemd/system/aibot.service`):

```ini
[Unit]
Description=AI Hedge Fund bot
After=network-online.target

[Service]
WorkingDirectory=/opt/bot-crypto
ExecStart=/opt/bot-crypto/.venv/bin/python main.py
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```
`Restart=on-failure` aman karena circuit breaker keluar dengan kode 3 dan membutuhkan `--reset-paper` / pengecekan manual; pertimbangkan `RestartPreventExitStatus=3` agar bot **tidak** otomatis hidup kembali setelah circuit breaker.

---

## Dashboard

Buka `http://localhost:5000`. Data diperbarui otomatis (ringkasan tiap 10 detik, harga tiap 2 detik, scanner tiap 15 detik).

| Tab | Isi |
|---|---|
| **📊 Ringkasan** | KPI (equity, PnL, drawdown, total risk, win rate), harga real-time, grafik candlestick, equity curve, bobot strategi & RL, posisi aktif |
| **🌐 Market** | Scanner 80 koin, filter & preset, heatmap, Fear & Greed, top gainers/losers, volume terbesar, funding ekstrem |
| **🧠 Analisis AI** | Regime, trend, RSI, Whale Trap, Smart Money, prediksi LSTM/Transformer, keputusan terakhir per simbol; hasil backtest |
| **📜 Riwayat & Performa** | PnL per simbol, statistik trade, log aktivitas, riwayat trade (filter + ekspor CSV) |

**Grafik candlestick:** pilih koin lewat tombol atau kotak 🔍 *Cari koin*, ganti timeframe (1m/5m/15m/1h/4h), nyalakan/matikan EMA dan Volume. Garis entry/SL/TP muncul untuk posisi yang sedang terbuka. Klik baris di scanner (atau tombol *Grafik*) untuk membuka koin tersebut.

**Scanner:** filter berdasarkan pencarian, sinyal (Bullish/Bearish/Overbought/Oversold/Neutral), trend 1H, perubahan 24h, volume minimum, zona RSI, funding, dan lonjakan volume. Preset cepat: *Breakout*, *Oversold*, *Overbought*, *Trend naik kuat*, *Likuid*. Favorit (⭐) dan filter tersimpan di browser. Scanner **hanya memantau**; bot hanya trading koin di `SYMBOLS`.

### Akses dari HP

1. Pastikan HP dan komputer/VPS berada di jaringan yang sama (atau gunakan IP publik VPS).
2. Buka `http://<IP-komputer>:5000` di browser HP.
3. Izinkan port 5000 di firewall (Windows: *Windows Defender Firewall → Inbound Rule → Port 5000*; VPS: `ufw allow 5000`).
4. **Isi `DASHBOARD_TOKEN`** di `.env`, lalu buka `http://<IP>:5000/?token=TOKEN_ANDA`. Token disimpan sebagai cookie, jadi hanya perlu sekali. Dashboard bersifat *read-only* (tidak ada tombol yang bisa mengubah posisi), tetapi tetap menampilkan data akun Anda.

### Endpoint API

| URL | Isi |
|---|---|
| `/` | Dashboard HTML |
| `/api` | JSON semua data: saldo, equity, DD, posisi, analisis, RL, statistik, riwayat |
| `/scanner` | Data scanner pasar |
| `/candles?symbol=BTC/USDT&tf=15m` | Candle + EMA untuk grafik |
| `/price-stream` | Harga real-time WebSocket |
| `/smart-money` | Skor whale, likuidasi, order book |
| `/strategies` | Bobot strategi saat ini |
| `/backtest` | Hasil backtest per simbol |

---

## Setup Telegram

1. Di Telegram cari **@BotFather** → `/newbot` → isi nama dan username (berakhiran `bot`). Salin **token** (`123456789:AAH…`).
2. Buka bot Anda (`t.me/username_bot`), tekan **Start**, lalu kirim pesan apa saja. *Langkah ini wajib*, kalau tidak bot tidak boleh mengirim pesan ke Anda.
3. Buka `https://api.telegram.org/bot<TOKEN>/getUpdates` dan cari `"chat":{"id":123456789`. Angka itu adalah **chat ID** (untuk grup biasanya negatif).
4. Isi di `.env`:
   ```
   TELEGRAM_TOKEN=123456789:AAH...
   CHAT_ID=123456789
   ```
5. Restart bot. Pesan *"AI Hedge Fund v2.0 aktif"* akan masuk.

Jika token bocor, kirim `/revoke` ke @BotFather dan perbarui `.env`. Bot ini hanya **mengirim** notifikasi, tidak membaca atau membalas perintah.

---

## Setup API Bybit

Hanya diperlukan untuk mode LIVE.

1. Bybit → *Profile → API → Create New Key* → **System-generated API Keys**.
2. Permission: **Read-Write**, centang hanya **Contract → Orders & Positions**. **Jangan** centang Withdrawal (dan sebaiknya tidak Spot/Options/Wallet Transfer).
3. Isi **IP whitelist** dengan IP komputer/VPS bot.
4. Transfer USDT ke **Derivatives (Futures) wallet** — minimal sekitar $200 untuk leverage 5x.
5. Pastikan mode posisi **One-Way** (bot mencoba menyetel One-Way + Cross + leverage otomatis saat start LIVE).
6. Isi `.env`:
   ```
   API_KEY=...
   SECRET=...
   ```

Jika `bybit.com` tidak bisa diakses dari jaringan Anda, isi `BYBIT_DOMAIN=bybit.nl`.

---

## Cara kerja bot

### 9 tahap startup

| # | Tahap | Detail |
|---|---|---|
| 1 | Setup futures | Leverage 5x + Cross margin + One-Way di Bybit (dilewati di PAPER) |
| 2 | Backtest | EMA+RSI pada 500 candle 1H. LIVE berhenti jika win rate < 50% atau profit factor < 1,2 |
| 3 | RL agent | Load `rl_qtable.json` (atau mulai kosong, ε = 0,20) |
| 4 | Model AI | Load LSTM dari `.pt` (atau train 30 epoch); Transformer selalu di-train ulang 20 epoch |
| 5 | Posisi | Pulihkan `positions.json`; di LIVE diverifikasi ke Bybit |
| 6 | Paper engine | Saldo virtual |
| 7 | Korelasi | Matriks korelasi antar simbol dari data 1H |
| 8 | Dashboard | Flask di port 5000 |
| 9 | WebSocket | `wss://stream.bybit.com/v5/public/linear` (ticker semua simbol) |

### FAST loop (tiap 1 detik, harga WebSocket)
1. Cek SL/TP posisi paper → tutup jika kena.
2. Keputusan cepat: Whale Trap + voting. Entry cepat hanya jika **momentum Scalper (0,1%) searah** dan semua filter lolos.
3. Lewati simbol yang sudah punya posisi dan simbol yang baru ditutup (cooldown 15 menit).
4. Hitung ukuran posisi (dynamic risk, leverage 5x, pembulatan presisi Bybit) → eksekusi → notifikasi.

### SLOW loop (tiap 15 menit)
Refresh korelasi → cek posisi live → refresh OHLCV → Smart Money → Market Regime → prediksi LSTM + Transformer → RL + voting → filter → eksekusi (SL/TP dipasang di Bybit) → update drawdown (circuit breaker) → simpan state → sinkron dashboard.

### Voting multi-strategi

| Strategi | Bobot | LONG | SHORT |
|---|---|---|---|
| Trend Following | 30% | EMA20 > EMA50 | EMA20 < EMA50 |
| Mean Reversion | 20% | RSI < 30 | RSI > 70 |
| Whale Strategy | 30% | Trap score ≥ 2 → membalik semua sinyal | |
| Scalper RT | 20% | harga naik > 0,1% | harga turun > 0,1% |
| Transformer | 20% | prediksi > harga | prediksi < harga |

Hasil voting dipadukan dengan aksi RL: jika sama → confidence +10; jika berbeda → voting menang; jika voting WAIT → aksi RL dipakai.

### Filter sebelum order
1. Confidence < 60 → skip
2. Regime VOLATILE → skip
3. Whale Trap ≥ 4 → skip
4. Smart Money berlawanan kuat (< −0,6 untuk LONG, > +0,6 untuk SHORT) → skip
5. Aksi WAIT → skip
6. Korelasi > 0,80 dengan posisi searah yang aktif → skip
7. Total risk + risk baru > 5% → skip

### Dynamic risk (dari BASE_RISK 1%)

| Kondisi | Multiplier |
|---|---|
| Normal | 1,0× |
| Volatilitas tinggi (ATR/harga > 5%) | 0,5× |
| Altcoin kecil (HYPE) | 0,5× |
| Smart Money negatif (< −0,5) | 0,5× |
| Regime SIDEWAYS | 0,3× |
| Regime VOLATILE | 0× (tidak trade) |
| Batas atas | 2% per trade |

### Smart Money Score
`Total = whale × 0,40 + likuidasi × 0,35 + orderbook × 0,25` (cache 5 menit, fallback ke nilai terakhir jika API gagal; tanpa API key komponen itu netral).

### Reinforcement Learning
Q-learning tabular, 5184 state (8 dimensi) × 3 aksi (LONG/SHORT/WAIT), α = 0,1, γ = 0,9, ε 0,20 → 0,05 (decay 0,995). Reward: profit `min(pnl×0,01, 2)`, loss `max(pnl×0,02, −3)`, WAIT saat confidence < 60 = +0,1, WAIT saat ≥ 60 = −0,1. Eksplorasi ε hanya aktif di PAPER; di LIVE agent selalu greedy.

### Keamanan sistem
- **Circuit breaker:** drawdown ≥ 20% → semua posisi ditutup, state disimpan, bot berhenti (kode keluar 3).
- **SL wajib:** di LIVE, jika SL tidak terverifikasi di Bybit setelah entry, posisi langsung ditutup.
- **WebSocket putus:** reconnect dengan backoff 5s × n (maks 10 kali); setelahnya bot memakai harga candle terakhir dan mengirim notifikasi.
- **Error per simbol** tidak menghentikan bot.
- **Thread safety:** semua data bersama dilindungi lock.

---

## Pindah ke mode LIVE

Selesaikan **semua** item ini terlebih dahulu:

**Teknis**
- [ ] Dependensi terpasang, `.env` terisi (API key, Telegram)
- [ ] Bot berjalan stabil (VPS/screen/systemd) dan dashboard bisa diakses

**Akun Bybit**
- [ ] API key Read-Write **tanpa Withdrawal**, IP whitelist aktif
- [ ] Saldo USDT ≥ $200 di Derivatives wallet
- [ ] One-Way Position dan Cross Margin

**Validasi paper trading (minimal 7–14 hari)**
- [ ] Tanpa error, WebSocket tidak putus > 3×/hari
- [ ] Win rate ≥ 50%, profit factor ≥ 1,2, max drawdown < 15%
- [ ] Notifikasi Telegram masuk dengan benar
- [ ] Backtest startup berstatus **VIABLE** (bot menolak LIVE jika tidak)

**Langkah**
1. Ubah `PAPER_TRADING=False` di `.env`.
2. Restart bot. `setup_futures()` berjalan otomatis.
3. Pantau 3 trade pertama langsung di Bybit; pastikan SL/TP terpasang pada setiap posisi.
4. Pantau dashboard tiap 1–2 jam di hari pertama.

Untuk kembali ke paper: `PAPER_TRADING=True`. Posisi yang tersimpan dari mode lain diabaikan saat start.

---

## File data

Semua tersimpan di folder `data/` (dibuat otomatis).

| File | Isi |
|---|---|
| `positions.json` | Posisi aktif; dibaca ulang saat restart |
| `state.json` | Saldo paper, peak equity, riwayat trade, equity curve |
| `rl_qtable.json` | Q-table RL + epsilon (disimpan tiap 15 menit dan saat shutdown) |
| `backtest_result.json` | Hasil backtest terakhir |
| `lstm_<SIMBOL>.pt` | Model LSTM per simbol (retrain otomatis tiap 24 jam) |
| `bot.log` | Log (rotasi otomatis, maks ± 2 MB × 3 file) |

---

## Struktur proyek

```
Bot Crypto/
├── main.py                 # Entry point & signal handler
├── requirements.txt
├── .env.example            # Template konfigurasi
├── bot/
│   ├── config.py           # Semua parameter + pembacaan .env
│   ├── engine.py           # Startup 9 tahap, FAST/SLOW loop, eksekusi, circuit breaker
│   ├── strategies.py       # Voting strategi + matriks RL × Voting
│   ├── risk.py             # Confidence, dynamic risk, sizing, filter, korelasi
│   ├── indicators.py       # EMA, RSI, ATR, ADX, market regime, Whale Trap
│   ├── models.py           # LSTM & Transformer (PyTorch)
│   ├── rl.py               # Q-learning agent
│   ├── smart_money.py      # CryptoQuant + Coinglass + order book
│   ├── ws_feed.py          # WebSocket Bybit + reconnect
│   ├── exchange.py         # Wrapper ccxt Bybit
│   ├── paper.py            # Engine paper trading
│   ├── backtest.py         # Backtest EMA+RSI 1H
│   ├── scanner.py          # Scanner pasar (hanya memantau)
│   ├── dashboard.py        # Flask API + server dashboard
│   ├── notifier.py         # Telegram (non-blocking)
│   ├── storage.py          # Simpan/baca JSON (atomic write)
│   ├── logger.py
│   └── web/
│       ├── index.html      # Dashboard (single page)
│       └── vendor/         # TradingView Lightweight Charts
├── tests/test_core.py
└── data/                   # State, model, log (dibuat otomatis)
```

---

## Testing

```powershell
python -m pytest tests -q
```
Test mencakup voting, confidence, dynamic risk, SL/TP & sizing, filter, paper engine, RL, indikator & Whale Trap, backtest, dan scanner. Test **tidak** menyentuh jaringan atau order sungguhan.

---

## Troubleshooting

| Gejala | Penyebab & solusi |
|---|---|
| `NetworkError` / `No market data` saat start | Jaringan/ISP menolak `bybit.com`. Tambahkan `BYBIT_DOMAIN=bybit.nl` di `.env` |
| `Dashboard failed to start` | Port 5000 dipakai program lain. Ubah `DASHBOARD_PORT` |
| Dashboard tidak bisa dibuka dari HP | Cek firewall (port 5000), pastikan satu jaringan, dan `DASHBOARD_HOST=0.0.0.0` |
| `401 unauthorized` di dashboard | `DASHBOARD_TOKEN` terisi. Buka `http://host:5000/?token=TOKEN` |
| `Telegram disabled` di log | `TELEGRAM_TOKEN` atau `CHAT_ID` kosong. Isi keduanya lalu restart |
| `chat_id is empty` / pesan tidak masuk | Anda belum menekan **Start** dan mengirim pesan ke bot. Lihat [Setup Telegram](#setup-telegram) |
| Bot menolak start LIVE: *Backtest tidak viable* | Strategi tidak memenuhi syarat (win rate ≥ 50%, PF ≥ 1,2, ≥ 10 trade). Itu perlindungan; tetap di PAPER |
| `LIVE mode requires API_KEY and SECRET` / equity tidak terbaca | Key belum diisi, izin kurang, atau saldo Derivatives 0 |
| `quantity below exchange minimum` | Ukuran order hasil hitungan di bawah minimum Bybit (modal kecil / koin mahal). Naikkan modal atau hapus koin dari `SYMBOLS` |
| Bot langsung berhenti setelah restart (kode 3) | Circuit breaker pernah aktif. Jalankan `python main.py --reset-paper` |
| Startup lambat | Normal: training model (± 17 detik per koin). Tiap koin tambahan di `SYMBOLS` menambah waktu |
| Grafik kosong / "Data chart belum tersedia" | Koneksi ke Bybit sedang bermasalah; coba lagi beberapa detik |

Log lengkap ada di `data/bot.log`.

---

## Keamanan

- `.env` berisi kunci rahasia: jangan di-commit, jangan dibagikan, dan jangan ditempel di chat/forum. File `.gitignore` sudah menyertakan `.env` dan `data/`.
- Gunakan API key **tanpa Withdrawal**, dengan **IP whitelist**, dan izin sesempit mungkin.
- Jika key/token pernah terlihat orang lain, **buat ulang** (Bybit API Management; `/revoke` di @BotFather).
- Isi `DASHBOARD_TOKEN` bila dashboard dibuka di jaringan publik, dan pertimbangkan akses lewat VPN/reverse proxy dengan HTTPS.
- Software ini disediakan apa adanya, tanpa jaminan. Anda bertanggung jawab penuh atas keputusan dan kerugian trading.

---

## Lisensi pihak ketiga

Grafik memakai [TradingView Lightweight Charts](https://github.com/tradingview/lightweight-charts) (Apache License 2.0). Salinan lisensinya ada di `bot/web/vendor/LICENSE-lightweight-charts.txt`. Atribusi "Charting by TradingView" pada grafik harus tetap ditampilkan.
