# 🛡️ บอทเฝ้าดอย V7.1 — Capital Preservation & Tactical Trend Switcher

ระบบวิเคราะห์แนวโน้มและจัดสรรสินทรัพย์เชิงยุทธวิธีอัตโนมัติ (Medium-to-Long Term Trend Following & Tactical Asset Allocation) ออกแบบมาเพื่อ **ปกป้องเงินต้นและลด Drawdown สูงสุด** สำหรับพอร์ตกองทุนรวม SCB โดยใช้ **SCBTMFPLUS-E** (กองทุนตลาดเงิน) เป็นหลุมหลบภัย ทำงานแบบ Serverless ไร้ต้นทุน 100% บน GitHub Actions ผสานพลังการคำนวณเชิงปริมาณแบบ **Deterministic Quant Engine (0-100 คะแนน)** เข้ากับ **AI Explainer & Multi-Model Fallback** พร้อมส่งสัญญาณเคาะคำสั่งสับเปลี่ยนกองทุน (SWITCH IN / HOLD / SWITCH OUT) เข้า Discord ทุกวันทำการเวลา 12:15 น. (ก่อนเวลาปิดรับคำสั่งกองทุน 15:00 น.)

---

## 🧠 Multi-AI Architecture & Idea Credits
โปรเจกต์นี้ได้รับการออกแบบและ Fine-tune ร่วมกันผ่านแนวคิดของสุดยอด AI แต่ละสาย:

### 🟢 1. GPT (The Systems Architect & Risk Engineer)
* **Deterministic Quant Engine:** วางโครงสร้างแยกขาดระหว่าง "Quant Engine (ผู้ตัดสินใจ 100%)" และ "LLM (ผู้อธิบายสรุปข่าว ไม่ให้มีสิทธิ์เปลี่ยนสัญญาณ)"
* **Tactical Switcher Framework:** ปรับเปลี่ยนแนวคิดจาก Swing เทรดเร็ว มาเป็นการบริหารจัดการเงินสดเข้า-ออกร่วมกับกองทุนตลาดเงิน `SCBTMFPLUS-E`
* **Production Resilience & Robust Error Handling:** วางระบบ `tenacity` Retries, การควบคุม Logging และการจัดการคำนวณคณิตศาสตร์แบบ Vectorized

### 🔵 2. DeepSeek (The Mathematical Auditor & Robustness Specialist)
* **Signal Line Degeneracy & Math Auditing:** ตรวจจับและแก้ไขข้อผิดพลาดทางคณิตศาสตร์ในโมเมนตัม ปรับจูน Signal Span และทดสอบขอบเขตคะแนน
* **Multi-Layer Defensive Filters:** เสริมเกราะป้องกันสัญญาณหลอกด้วยการตรวจสอบความผิดปกติของข้อมูล และวางเกณฑ์ตัดขาดทุนขั้นเด็ดขาดเมื่อโครงสร้างเทรนด์พังทลาย
* **Discord Payload Chunking:** วางระบบตัดแบ่งข้อความให้อยู่ในขอบเขต 1,900 ตัวอักษร เพื่อป้องกันข้อผิดพลาด 400 Bad Request เมื่อรายงานมีความยาวเกินกำหนด

### 🟣 3. Gemini (The Data Reliability & Serverless Pipeline Lead)
* **Multi-Source Data Engine (Yahoo $\rightarrow$ Stooq Fallback):** แก้ปัญหา GitHub Runner โดนบล็อก 429 Rate Limit จาก Yahoo Finance โดยวางระบบสลับไปดึงข้อมูลราคาจาก Stooq อัตโนมัติ ทำให้บอทดึงข้อมูลได้ 100%
* **Multi-Model Auto Fallback Cascade:** ระบบสลับโมเดลอัตโนมัติ 4 ชั้นเมื่อติด Rate Limit หรือเซิร์ฟเวอร์หนาแน่น (`gemini-3.7-flash` $\rightarrow$ `gemini-3.5-flash-lite` $\rightarrow$ `gemini-3.1-flash-lite` $\rightarrow$ `gemini-2.5-flash-lite`)
* **Real-World Execution & Settlement Alignment:** ออกแบบจังหวะแจ้งเตือนให้สอดคล้องกับรอบเวลา $T+1$ ถึง $T+3$ และเวลา Cut-off ของ บลจ. ไทย
* **Live News Integration:** เชื่อมต่อระบบดึงข่าวการเงินเรียลไทม์จาก Finnhub API

### 🔴 4. Kimi (The DevOps, CI/CD & Trend Mechanics Lead)
* **Enterprise CI/CD & State Caching:** วาง Pipeline บน GitHub Actions พร้อมระบบ State Cache (`state.json`), การทำ Atomic File Write และระบบแจ้งเตือนฉุกเฉินเข้า Discord เมื่อ Workflow ล้มเหลว
* **Position-Aware State Machine (Dead-Zone 55–74):** สร้างระบบจำสถานะการถือครอง (`IN` / `OUT`) แยกจากสัญญาณ เพื่อตัดปัญหา Whipsaw และลดค่าเสียโอกาสจากการสับเปลี่ยนกองทุนบ่อยเกินไป
* **Medium-to-Long Term Indicator Stack:** ปรับจูนระบบตัวชี้วัดเป็น $EMA 50/100/200$, Wilder $RSI(14)$, Standard $MACD(12,26,9)$, Wilder $ADX(14)$ (พร้อมแยก $+DI/-DI$) และ 20-day Historical Volatility Percentile (252 วัน)
* **Anti-Chop Score Cap & Hard Breakdown Guard:** ล็อกคะแนนไม่ให้เกิน 65 ในช่วงตลาด Sideways ($ADX < 20$) เพื่อปิดทาง SWITCH IN และสั่ง SWITCH OUT ทันทีเมื่อราคาหลุด $EMA 200$ เกิน $-2\%$

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

* `GEMINI_API_KEY`: Google AI Studio API Key (รองรับ Gemini 3.x Flash Family)
* `DISCORD_WEBHOOK_URL`: Discord Webhook URL สำหรับส่งสัญญาณเตือน
* `FINNHUB_API_KEY`: API Key จาก [Finnhub.io](https://finnhub.io/) สำหรับดึงข่าวสด

---

## 🛠️ Tech Stack
* **Language:** Python 3.12
* **Data & Math:** `yfinance`, `pandas`, `numpy`, `requests` (Stooq Integration)
* **Resilience:** `tenacity`, Atomic I/O
* **AI & NLP:** `google-genai` (Gemini 3.x Flash Family)
* **Automation:** GitHub Actions (Ubuntu Latest + State Cache Storage)
