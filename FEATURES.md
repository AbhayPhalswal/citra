# Citra — what it actually does

A voice assistant that runs **your flat and your PC**. Speech in, real
switches flipped, real windows opened. No subscription. Wake word,
speech-to-text and text-to-speech all run **on your machine**.

**44 tools.** Every one below is in the code, not on a roadmap.

---

## 🏠 It runs your actual flat

Not smart bulbs — **your existing switchboards**, rewired with relay
boards and a ₹200 NodeMCU behind each one.

- `"Citra, turn on the lights"` → a physical relay clicks
- Per-room control across 11 boards: bedrooms, hall, kitchen, bathroom
- Air conditioner over IR — power, temperature, the lot
- `"turn everything off"` on your way out
- **Works with the internet down.** Light and AC commands never leave
  your WiFi: a local pattern match, then a direct call to the board.
  Alexa cannot do this.

## 🖥️ It runs your PC

Fifteen tools, by voice:

- Open apps, files, folders — or a whole **layout** at once
  (`"Citra, open my study setup"`)
- **Install and uninstall software** via winget, silently
- **Block a website** when you should be studying. Unblock it later.
- Lock the screen, screenshot, force-quit something frozen
- Dark mode, brightness, search your files
- Run any shell command *(see the safety note in the README)*

## 👁️ It can see your screen — locally

A vision model running **on your own machine** (no screenshots leave it):

- `"what's on my screen?"` — it reads and answers
- It can **click** where it needs to, **type**, and **press keys**

Ask it about an error message and it looks, instead of guessing.

## 💻 It writes code and opens it for you

- `"write me a Python script that renames all my photos by date"`
- Code appears in Notepad, ready to run
- Fast model by default; a much larger one when you ask for something
  big like a working website — you say which

## ⏰ It remembers and schedules

- `"remind me in 20 minutes to submit the assignment"` — spoken aloud
- **Schedule the room itself**: `"turn the geyser on at 6am"`
- List what's pending, cancel any of it

## 📱 Phone dashboard

- Installs to your home screen like an app (PWA)
- Every switch, the AC, live state
- Type to it instead of talking, and have it read replies aloud
- A console button that opens the real logs

## 🗣️ Speech that isn't annoying

- Wake word **"Hey Citra"** — trained on the developer's own voice
- Whisper transcription **on-device**, ~250ms warm
- Neural TTS (Piper), local
- **She tells you why she is slow** instead of leaving dead air — and
  says nothing at all when the answer is quick
- **Mute her.** Muted means muted, expires by itself, and no feature can
  talk over it.

## 🛠️ Built to be debugged

- `python citra_doctor.py` — one command, checks every subsystem in ~10s
- Honest partial failures: *"3 of 4 lights are on, the fourth board isn't
  answering"* — never a silent lie
- Every module explains **why** it is written the way it is, including
  the bugs that shaped it

---

## Why this is interesting

It is not "another ChatGPT wrapper". Three quarters of the hard parts
are physical: mains-powered relay boards behind real switches, IR
timing, an ESP8266 that handles one request at a time, DHCP conflicts,
audio device routing on Windows.

It costs about **₹1,200 per switchboard** in parts, runs offline for the
things that matter, and pays nobody a monthly fee.

Built by a first-year Data Science student to automate his own flat,
then his neighbours'.
