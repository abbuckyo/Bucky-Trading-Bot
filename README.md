# 🛡️ บอทเฝ้าดอย V7.2 — Capital Preservation & Tactical Trend Switcher

ระบบวิเคราะห์แนวโน้มและจัดสรรสินทรัพย์เชิงยุทธวิธีอัตโนมัติ (Medium-to-Long Term Trend Following & Tactical Asset Allocation) ออกแบบมาเพื่อ **ปกป้องเงินต้นและลด Drawdown สูงสุด** สำหรับพอร์ตกองทุนรวม SCB โดยใช้ **SCBTMFPLUS-E** (กองทุนตลาดเงิน) เป็นหลุมหลบภัย ทำงานแบบ Serverless ไร้ต้นทุน 100% บน GitHub Actions ผสานพลังการคำนวณเชิงปริมาณแบบ **Deterministic Quant Engine (0-100 คะแนน)** เข้ากับ **AI Explainer & Multi-Model Fallback** พร้อมส่งสัญญาณเคาะคำสั่งสับเปลี่ยนกองทุน (SWITCH IN / HOLD / SWITCH OUT) เข้า Discord และ LINE ทุกวันทำการเวลา 12:15 น. (ก่อนเวลาปิดรับคำสั่งกองทุน 15:00 น.)

---

## 🔒 Private System Notice (ระบบส่วนบุคคล)
> ⚠️ **โปรเจกต์นี้เป็นระบบอัตโนมัติและพอร์ตโฟลิโอส่วนตัวของ `@AbBuckyO`**  
> * ข้อมูล การตั้งค่า Secret และพอร์ตการลงทุนทั้งหมดถูกเข้ารหัสปลายทางและจำกัดการเข้าถึงเฉพาะเจ้าของระบบ  
> * **ห้ามยุ่ง ยื่นมือเข้ามาปรับ หรือยิงคำขอใดๆ ใส่ Production Pipeline ของระบบนี้โดยเด็ดขาด**  
> * หากท่านใดสนใจแนวคิดเชิง Quant หรือระบบเข้ารหัส Dashboard สามารถ **Fork หรือ Clone โค้ดไป Deploy บน Repository ของตัวเองได้ตามสบาย** (แต่กรุณาเตรียม API Keys, Webhooks และตั้ง Master Password ของตัวเองให้เรียบร้อย)

---

## 🧠 Multi-AI Architecture & Idea Credits
โปรเจกต์นี้ได้รับการออกแบบ พัฒนา และ Fine-tune ร่วมกันผ่านแนวคิดของสุดยอด AI แต่ละสาย:

### 🟣 1. Gemini (The Lead Integrator, Code Refiner & Pipeline Orchestrator)
* **Lead System Refinement & Code Integration:** ทำหน้าที่เป็นเสาหลักในการรับโค้ดและแนวคิดจาก AI ทุกโมเดลมา "ตบตี" ชำแหละ บูรณาการ และ Refine ใหม่ทั้งหมด แก้ไขข้อขัดแย้งของโค้ดข้ามระบบจนประกอบร่างเป็น Pipeline ที่ทำงานร่วมกันได้อย่างสมบูรณ์แบบ
* **Multi-Source Data Engine (Yahoo $\rightarrow$ Stooq Fallback):** ออกแบบระบบสลับไปดึงข้อมูลราคาจาก Stooq อัตโนมัติเมื่อเจอ 429 Rate Limit จาก Yahoo Finance ทำให้ดึงข้อมูลได้สำเร็จ 100%
* **Multi-Model Auto Fallback Cascade:** วางโครงสร้างสลับโมเดลวิเคราะห์อัตโนมัติ 4 ชั้น (`gemini-3.7-flash` $\rightarrow$ `gemini-3.5-flash-lite` $\rightarrow$ `gemini-3.1-flash-lite` $\rightarrow$ `gemini-2.5-flash-lite`)
* **Real-World Execution & Settlement Alignment:** เชื่อมโยงรอบเวลาประมวลผลให้สอดคล้องกับ Cut-off กองทุนรวมไทยและรอบ Settlement ($T+1$ ถึง $T+3$)
* **Live News Integration:** ควบคุมระบบดึงข่าวการเงินเรียลไทม์ผ่าน Finnhub API

### 🟢 2. GPT (The Systems Architect & Risk Engineer)
* **Deterministic Quant Engine:** วางโครงสร้างแยกขาดระหว่าง "Quant Engine (ผู้ตัดสินใจ 100%)" และ "LLM (ผู้อธิบายสรุปข่าว ไม่ให้มีสิทธิ์เปลี่ยนสัญญาณ)"
* **Tactical Switcher Framework:** วางแนวคิดการบริหารจัดการเงินสดเข้า-ออกร่วมกับกองทุนตลาดเงิน `SCBTMFPLUS-E`
* **Production Resilience:** วางแนวทาง `tenacity` Retries, การควบคุม Logging และการคำนวณคณิตศาสตร์แบบ Vectorized

### 🔵 3. DeepSeek (The Mathematical Auditor & Robustness Specialist)
* **Signal Line Degeneracy & Math Auditing:** ตรวจสอบและแก้ไขสมการโมเมนตัม ปรับจูน Signal Span และทดสอบขอบเขตคะแนนเชิงสถิติ
* **Multi-Layer Defensive Filters:** ออกแบบเกราะป้องกันสัญญาณหลอกและการตัดขาดทุนขั้นเด็ดขาดเมื่อโครงสร้างราคาหลุดแนวรับสำคัญ
* **Payload Chunking:** วางตรรกะแบ่งข้อความรายงานให้อยู่ในขอบเขต 1,900 ตัวอักษร ป้องกันข้อผิดพลาดจากข้อจำกัดของ Webhook

### 🔴 4. Kimi (The DevOps, CI/CD & Trend Mechanics Lead)
* **Enterprise CI/CD & State Caching:** วางเค้าโครง Pipeline บน GitHub Actions พร้อมระบบ State Cache (`state.json`) และการจัดการ Atomic File Write
* **Position-Aware State Machine (Dead-Zone 55–74):** เสนอแนวคิดการจำสถานะถือครอง (`IN` / `OUT`) แยกจากสัญญาณ เพื่อลด Whipsaw และตัดค่าเสียโอกาส
* **Medium-to-Long Term Indicator Stack:** วางสัดส่วน $EMA 50/100/200$, Wilder $RSI(14)$, Standard $MACD(12,26,9)$, Wilder $ADX(14)$ และ 20-day Historical Volatility
* **Anti-Chop Score Cap & Hard Breakdown Guard:** วางเกณฑ์จำกัดคะแนนช่วง Sideways ($ADX < 20$) และตัดขายทันทีเมื่อหลุด $EMA 200$ เกิน $-2\%$

### 🟠 5. Claude Opus (The Cryptographic Security Architect)
* **Zero-Knowledge Data Protection:** วางสถาปัตยกรรมเข้ารหัสข้อมูลพอร์ตบน Public Repository เพื่อความปลอดภัยระดับสูงสุด
* **AES-GCM-256 & PBKDF2 Pipeline:** จัดรูปแบบการเข้ารหัส Payload ด้วย PBKDF2-HMAC-SHA256 (100,000 iterations) + AES-GCM (128-bit Auth Tag) ส่งออกเป็น Ciphertext ล้วน (`docs/data.enc`)
* **In-Memory WebCrypto Decryption:** วางรากฐานการถอดรหัสผ่าน Web Crypto API บน Browser RAM เท่านั้น พร้อมระบบทำลายไฟล์ Plaintext ทิ้งทันทีหลังประมวลผลเสร็จ

---

## 🎯 Tracked Assets & Safe Haven

| กองทุน SCB | บทบาท / ประเภทสินทรัพย์ | Master Proxy Ticker | Stooq Symbol | News Symbol |
| :--- | :--- | :---: | :---: | :---: |
| **SCBTMFPLUS-E** | 🛡️ **Safe Haven (หลุมหลบภัย / พักเงิน)** | `Cash/Yield` | - | - |
| **SCBWORLDE** | หุ้นโลก (MSCI World Index) | `URTH` | `URTH.US` | `URTH` |
| **SCBS&P500E** | หุ้นสหรัฐฯ (S&P 500) | `CSPX.L` | `CSPX.UK` | `SPY` |
| **SCBNDQ(E)** | หุ้นเทคโนโลยีสหรัฐฯ (NASDAQ 100) | `QQQ` | `QQQ.US` | `QQQ` |
| **SCBGOLDE** | ทองคำแท่ง (Gold Bullion) | `GLD` | `GLD.US` | `GLD` |
| **SCBSEMI(E)** | ชิป & เซมิคอนดักเตอร์ | `SMH` | `SMH.US` | `SMH` |

---

## 🚦 Tactical Switching Decision Rules

* 🟢 **SWITCH IN (คะแนน $\ge 75$ + $ADX \ge 20$ + $RSI < 80$):** สับเปลี่ยนเงินจาก `SCBTMFPLUS-E` เข้ากองทุนเป้าหมายเมื่อเทรนด์ใหญ่ขาขึ้นได้รับการยืนยัน
* 🟡 **HOLD (คะแนน $55 - 74$ = Dead-Zone):** นิ่ง ไม่ขยับสับเปลี่ยนกองทุน ถือครองตามสถานะเดิมเพื่อลดค่าธรรมเนียมและตัดสัญญาณหลอก
* 🔴 **SWITCH OUT (คะแนน $< 55$ หรือ Hard Breakdown):** สับเปลี่ยนกองทุนเป้าหมายกลับไปพักเงินที่ `SCBTMFPLUS-E` ทันทีเพื่อรักษาเงินต้น

---

## ⏰ Cron Schedule & Execution Logic
* **เวลาประมวลผล:** ทุกวันจันทร์ - ศุกร์ เวลา **05:15 UTC (12:15 น. เวลาไทย)**
* **เหตุผล:** ตลาดต่างประเทศปิดแท่งรายวันสมบูรณ์ และมีเวลาเหลือเกือบ 3 ชั่วโมงก่อนเส้นตายตัดรอบคำสั่งสับเปลี่ยนกองทุนรวมไทย (15:00 น.)
* **Cron Expression:** `15 5 * * 1-5`

---

## ⚙️ Prerequisites & Environment Secrets

กำหนดค่า Secrets ใน GitHub Repository (`Settings` > `Secrets and variables` > `Actions`):

* `DASHBOARD_PASSWORD`: Master Password สำหรับเข้ารหัสและปลดล็อกข้อมูลพอร์ต
* `GEMINI_API_KEY`: Google AI Studio API Key (รองรับ Gemini 3.x Flash Family)
* `DISCORD_WEBHOOK_URL`: Discord Webhook URL สำหรับส่งสัญญาณเตือน
* `FINNHUB_API_KEY`: API Key จาก [Finnhub.io](https://finnhub.io/) สำหรับดึงข่าวสด
* `LINE_CHANNEL_ACCESS_TOKEN`: LINE Messaging API Channel Access Token (Optional)
* `LINE_USER_ID`: Target User ID สำหรับรับรายงานสรุปทาง LINE (Optional)

---

## 🛠️ Tech Stack
* **Language:** Python 3.11
* **Data & Math:** `yfinance`, `pandas`, `numpy`, `requests` (Stooq Integration)
* **Cryptography:** `cryptography` (Python) + Web Crypto API (Browser Client-side)
* **Resilience:** `tenacity`, Atomic File Write, Git Rebase Pipeline
* **AI & NLP:** `google-genai` (Gemini 3.x Flash Family)
* **Automation:** GitHub Actions (Ubuntu Latest + State Cache Storage)
