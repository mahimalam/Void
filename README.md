# ⚡ Void

> *"Sometimes you gotta run before you can walk."* — Tony Stark

<div align="center">

```
  ██    ██  ██████  ██ ██████  
  ██    ██ ██    ██ ██ ██   ██ 
  ██    ██ ██    ██ ██ ██   ██ 
   ██  ██  ██    ██ ██ ██   ██ 
    ████    ██████  ██ ██████  
```

**An always-listening, multi-turn AI voice assistant built for the real-life Iron Man experience.**  
*Runs on Linux & Windows • Zero Paid Subscriptions • Deep Persistent Second Brain Memory*

---

</div>

## 🌌 What is Void?

**Void** is a dream project brought to life: a high-octane, low-latency AI voice assistant that lives on your machine, listens exclusively for your **"Void"** command, controls your desktop, and talks back like a razor-sharp British AI butler.

No robotic essays. No explaining every single button it clicks. You speak, it executes, and confirms with cold, effortless precision.

---

## ⚡ The Key Superpowers

### 🧠 1. The "Second Brain" Cognitive Memory System
Most voice bots have severe gold-fish amnesia: close the app, and they forget who you are. **Void has a persistent Second Brain:**
- **Vector Memory (LanceDB)**: Automatically embeds and indexes past conversations, notes, and facts with lightning-fast semantic retrieval.
- **Knowledge Graph Activation**: Connects related concepts, people, projects, and ideas so Void understands context intuitively.
- **Write-Ahead Log (WAL)**: Resilient crash-proof memory that preserves state even through sudden reboots.
- **Contextual Awareness**: Remembers your work style, projects, preferences, and details across days and weeks.

### 💸 2. 100% Free AI Brain Stack (Zero Subscriptions Required!)
You don't need a $20/month ChatGPT or Claude subscription to run a world-class voice assistant:
- **Free Cloud AI Routing**: Seamlessly routes through free-tier gateways (Kilo Gateway, Free OpenAI-compatible endpoints) for high-speed reasoning and accurate tool calling.
- **Dual Engine Architecture**: Automatically routes quick commands to ultra-fast flash engines (~1s response) and complex queries to powerhouse models (Nemotron / DeepSeek / Cohere).
- **Offline Modular Fallback**: Built-in adapter for local **Ollama** (Qwen 2.5 / SmolLM) for users who want 100% offline, air-gapped autonomy.

### 🎙️ 3. Custom Bi-Directional GRU Wake Word (`Hey Void` & `Void`)
- **Trained Neural Detector**: Custom-trained Bidirectional Recurrent GRU running locally via OpenWakeWord ONNX.
- **Zero Cloud Eavesdropping**: 100% on-device wake detection on CPU/ONNX Runtime.
- **Phoneme Discrimination**: Triggers vigorously on **"Hey Void"** or just **"Void"** (>99.5% confidence) while rejecting confusers (*"avoid"*, *"voice"*, *"point"*, room chatter) with 0.0000 false triggers.

### 🗣️ 4. Continuous Multi-Turn Conversations
- **No Wake-Word Fatigue**: After waking Void and receiving an answer, the follow-up window stays open for 7 seconds.
- **Natural Flow**: Keep conversing naturally without repeating the wake word on every turn.
- **Effortless Standby**: Stays quiet when you're done, or goes to sleep immediately when you say *"That's all"*, *"Go to sleep"*, or *"Sleep"*.

### 🖥️ 5. Real PC Desktop Execution & Tools
- **Media & YouTube**: *"Play Bohemian Rhapsody"* → instantly opens and autoplays the top track; pause, skip, and volume control.
- **App Launcher**: Launches Chrome, VS Code, Spotify, Telegram, terminals, and custom desktop apps.
- **Web & Live Knowledge**: Real-time web search (Tavily/Serper) for live scores, weather, flights, and news.
- **Voice-Confirmed Actions**: High-privilege tasks (running shell scripts, managing files) require your voice confirmation (*"I confirm"*).
- **Cyberpunk Electron HUD**: Sleek, frameless, neon-reactive audio waveform HUD that floats on your desktop.

---

## 🛠️ Architecture Stack

```
   [ Mic Audio ] ──> [ DC-Blocking HPF ] ──> [ OpenWakeWord (void.onnx) ]
                                                       │ (Wake!)
                                                       ▼
   [ Deepgram Aura-2 TTS ] <── [ Brain Router ] <── [ Silero VAD + Deepgram STT ]
             │              (Free Kilo / Gemini / Qwen)
             ▼
   [ Speakers / HUD ]
```

| Component | Technology | Role |
|---|---|---|
| **Wake Word** | OpenWakeWord + Recurrent GRU | Local on-device *"Hey Void"* / *"Void"* trigger |
| **VAD** | Silero VAD | Millisecond-accurate voice activity & silence detection |
| **STT** | Deepgram (Nova-3) | Rapid, accent-robust cloud speech-to-text |
| **AI Brains** | Free Kilo Gateway / Ollama | Dual-tier reasoning (Fast Flash + Deep Reasoning) |
| **TTS** | Deepgram Aura-2 | Ultra-natural, low-latency conversational voices |
| **Memory** | LanceDB + Knowledge Graph | Persistent semantic vector retrieval & memory graph |
| **HUD** | Electron + WebSockets | Cyberpunk Stark Industries visualizer overlay |

---

## 🚀 Quick Start

### 1. Clone & Setup
```bash
git clone https://github.com/mahimalam/Void.git
cd Void
```

### 2. Run the Linux / Windows Setup
On Linux:
```bash
chmod +x setup.sh run.sh
./setup.sh
```

### 3. Add Free API Keys
Copy the example environment file:
```bash
cp data/.env.example data/.env
```
Edit `data/.env` with your free Deepgram key (for speech) and search/gateway keys:
```env
DEEPGRAM_API_KEY=your_deepgram_key_here
TAVILY_API_KEY_1=your_free_tavily_key_here
```

### 4. Wake Him Up!
```bash
./run.sh
```
Say: **"Hey Void"** or press **Enter** in the terminal.

---

## 🎩 Voice Persona & Etiquette

Void is calibrated like Tony Stark's personal butler:
- **Direct & Punchy**: One short sentence for most confirmations (*"Opened YouTube, sir."*, *"Volume at 50%."*).
- **Silent Execution**: Never narrates internal thoughts or announces what tool it's calling.
- **Standby Commands**: Say *"sleep"*, *"standby"*, or *"that's all"* to put Void to rest.

---

## 📜 License & Acknowledgments

Built with passion for Iron Man fans, tinkerers, and builders who believe that personal computing should feel like magic. 

*License: MIT. Have fun, and don't blow up the workshop.* 🦾
