# Hermes Agent with Signal and llama.cpp

This is the step-by-step setup we used to run Hermes Agent with Signal as the chat interface and a local Qwen model served by `llama.cpp`.

Sensitive values are shown as placeholders:

- `<SIGNAL_ACCOUNT_NUMBER>`: your Signal number in E.164 format, for example `+1234567890`
- `<LLAMA_API_KEY>`: the local API key used between Hermes and `llama-server`

## 1. Base Assumptions

- OS: Ubuntu Linux on `aarch64`
- Project workspace: `~/Documents/github/llm-spark-exp`
- Hermes home: `~/.hermes`
- Local bin directory: `~/.local/bin`
- Local model:

```bash
~/.local/share/llama.cpp/models/qwen3.6-aeon-q4_k_m.gguf
```

The model is served through an OpenAI-compatible `llama.cpp` server on localhost.

## 2. Check Toolchain

We verified the compiler and protobuf tools:

```bash
clang --version
protoc --version
```

The machine had:

```text
Ubuntu clang version 18.1.3
libprotoc 3.21.12
```

If these are missing, install them:

```bash
sudo DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=a apt-get update
sudo DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=a apt-get install -y clang libclang-dev protobuf-compiler libprotobuf-dev
```

## 3. Install Hermes Agent

We downloaded the Hermes Agent installer and inspected it before running it.

Installer source shown in the script:

```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh -o /tmp/hermes-install.sh
less /tmp/hermes-install.sh
```

Then we installed Hermes without interactive setup or browser launch:

```bash
bash /tmp/hermes-install.sh --skip-setup --skip-browser
```

Important installed paths:

```text
~/.local/bin/hermes
~/.hermes/config.yaml
~/.hermes/.env
~/.hermes/hermes-agent
```

## 4. Configure Hermes for llama.cpp

We configured Hermes to use a custom OpenAI-compatible provider:

```bash
~/.local/bin/hermes config set model.provider custom
~/.local/bin/hermes config set model.default qwen-3.6-local
~/.local/bin/hermes config set model.base_url http://127.0.0.1:18080/v1
~/.local/bin/hermes config set model.context_length 65536
```

We chose not to store the API key in `config.yaml`. Instead, we stored it in `~/.hermes/.env`.

Generate a local API key:

```bash
openssl rand -hex 32
```

Add it to `~/.hermes/.env`:

```bash
OPENAI_API_KEY=<LLAMA_API_KEY>
LLAMA_API_KEY=<LLAMA_API_KEY>
```

Then restrict the file permissions:

```bash
chmod 600 ~/.hermes/.env
```

`OPENAI_API_KEY` is used by Hermes. `LLAMA_API_KEY` is used by `llama-server`, so the key does not need to appear in process arguments.

## 5. Start llama.cpp Server

The first working command used this shape:

```bash
~/.local/bin/llama-server \
  -m ~/.local/share/llama.cpp/models/qwen3.6-aeon-q4_k_m.gguf \
  -c 65536 \
  -t 16 \
  --host 127.0.0.1 \
  --port 18080 \
  --api-key <LLAMA_API_KEY> \
  -a qwen-3.6-local \
  --chat-template chatml \
  --reasoning off \
  --reasoning-budget 0 \
  --reasoning-format deepseek \
  --no-webui
```

Later, for systemd, we moved the API key into `~/.hermes/.env` and removed `--api-key` from the service command.

Why these flags matter:

- `-c 65536`: Hermes rejected smaller context windows for this local model.
- `--chat-template chatml`: the model's default template produced unwanted thinking text.
- `--reasoning off`: keeps the visible assistant replies cleaner.
- `--host 127.0.0.1`: keeps the model server private to this machine.
- `--no-webui`: reduces exposed surface area.

## 6. Smoke Test Hermes

After the server was running, we tested Hermes directly:

```bash
~/.local/bin/hermes chat -q 'Reply with exactly: ready' -Q --ignore-rules --max-turns 1
```

Expected result:

```text
ready
```

## 7. Install signal-cli

The native `signal-cli` Linux build we tried was `x86_64`, so it could not run on this `aarch64` machine.

We used the JVM build instead:

```bash
mkdir -p ~/.local/opt
tar -xzf /tmp/signal-cli-0.14.3.tar.gz -C ~/.local/opt
```

Installed path:

```text
~/.local/opt/signal-cli-0.14.3
```

We also installed a user-local Temurin JRE:

```text
~/.local/opt/temurin-25-jre
```

Then we created this wrapper:

```bash
mkdir -p ~/.local/bin
```

```bash
cat > ~/.local/bin/signal-cli <<'EOF'
#!/usr/bin/env bash
export JAVA_HOME="$HOME/.local/opt/temurin-25-jre"
export PATH="$JAVA_HOME/bin:$PATH"
exec "$HOME/.local/opt/signal-cli-0.14.3/bin/signal-cli" "$@"
EOF
chmod +x ~/.local/bin/signal-cli
```

Verify:

```bash
~/.local/bin/signal-cli --version
```

Expected result:

```text
signal-cli 0.14.3
```

## 8. Build libsignal JNI for aarch64

The JVM `signal-cli` package did not include the required `libsignal-client` native library for Linux `aarch64`.

We built it from the official Signal source instead of downloading a third-party binary.

Install Rust:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o /tmp/rustup-init.sh
less /tmp/rustup-init.sh
sh /tmp/rustup-init.sh -y --profile minimal
```

Clone the matching `libsignal` tag:

```bash
git clone --depth 1 --branch v0.92.1 https://github.com/signalapp/libsignal.git /tmp/libsignal-v0.92.1
```

Build the desktop JNI library:

```bash
cd /tmp/libsignal-v0.92.1
JAVA_HOME=$HOME/.local/opt/temurin-25-jre \
PATH=$HOME/.cargo/bin:$HOME/.local/opt/temurin-25-jre/bin:$PATH \
./java/build_jni.sh desktop
```

The build produced:

```text
/tmp/libsignal-v0.92.1/java/client/src/main/resources/libsignal_jni_aarch64.so
```

Back up the original jar and add the `aarch64` JNI library:

```bash
cp ~/.local/opt/signal-cli-0.14.3/lib/libsignal-client-0.92.1.jar \
   ~/.local/opt/signal-cli-0.14.3/lib/libsignal-client-0.92.1.jar.orig

zip -j ~/.local/opt/signal-cli-0.14.3/lib/libsignal-client-0.92.1.jar \
  /tmp/libsignal-v0.92.1/java/client/src/main/resources/libsignal_jni_aarch64.so
```

## 9. Link Signal as a Secondary Device

Run:

```bash
~/.local/bin/signal-cli link -n HermesAgent
```

This prints a one-time `sgnl://linkdevice?...` URL.

On the phone:

1. Open Signal.
2. Go to linked devices.
3. Add a new linked device.
4. Scan the QR code or open the `sgnl://` link.

After linking, `signal-cli` prints:

```text
Associated with: <SIGNAL_ACCOUNT_NUMBER>
```

## 10. Start signal-cli HTTP Daemon

Start the local Signal HTTP daemon:

```bash
~/.local/bin/signal-cli --account <SIGNAL_ACCOUNT_NUMBER> daemon --http 127.0.0.1:18081
```

Health check:

```bash
curl --max-time 5 http://127.0.0.1:18081/api/v1/check
```

A `200` response with an empty body is OK.

## 11. Configure Hermes Signal Gateway

We backed up `~/.hermes/.env` and added Signal settings:

```bash
cp ~/.hermes/.env ~/.hermes/.env.bak-signal-$(date +%Y%m%d%H%M%S)
```

Add:

```bash
SIGNAL_HTTP_URL=http://127.0.0.1:18081
SIGNAL_ACCOUNT=<SIGNAL_ACCOUNT_NUMBER>
SIGNAL_ALLOWED_USERS=<SIGNAL_ACCOUNT_NUMBER>
SIGNAL_HOME_CHANNEL=<SIGNAL_ACCOUNT_NUMBER>
SIGNAL_ALLOW_ALL_USERS=false
```

Then:

```bash
chmod 600 ~/.hermes/.env
```

`SIGNAL_ALLOWED_USERS` and `SIGNAL_ALLOW_ALL_USERS=false` are important: they keep Hermes from replying to arbitrary Signal senders.

## 12. Start Hermes Gateway

Run:

```bash
~/.local/bin/hermes gateway run --accept-hooks
```

Then check status:

```bash
~/.local/bin/hermes status
```

Useful log file:

```bash
tail -n 80 ~/.hermes/logs/gateway.log
```

## 13. Send a Test Message from Signal

From the phone:

1. Open Signal.
2. Tap compose.
3. Search for `Note to Self` or your own name/number.
4. Send:

```text
Hello Hermes
```

Hermes received the message through Signal and replied successfully.

We also tested:

```text
How can you help me
```

Hermes processed it and sent a longer reply back through Signal.

## 14. Enable Signal Voice Notes

Hermes can auto-transcribe inbound voice/audio messages from Signal. We use local `faster-whisper` so voice notes stay on this machine.

Install local STT into the Hermes virtualenv:

```bash
~/.local/bin/uv pip install \
  --python ~/.hermes/hermes-agent/venv/bin/python \
  faster-whisper
```

Force Hermes to use local STT, instead of falling back to any cloud provider:

```bash
~/.local/bin/hermes config set stt.provider local
~/.local/bin/hermes config set stt.local.language en
```

The existing config already had:

```yaml
stt:
  enabled: true
  local:
    model: base
    language: en
```

Smoke-test the local STT path:

```bash
cd ~/.hermes/hermes-agent
~/.hermes/hermes-agent/venv/bin/python - <<'PY'
import math, struct, wave
from pathlib import Path

path = Path("/tmp/hermes-stt-smoke.wav")
rate = 16000

with wave.open(str(path), "wb") as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(rate)
    for i in range(rate):
        sample = int(0.05 * 32767 * math.sin(2 * math.pi * 440 * i / rate))
        w.writeframes(struct.pack("<h", sample))

from tools.transcription_tools import transcribe_audio
print(transcribe_audio(str(path), model="base"))
PY
```

Expected shape:

```text
{'success': True, 'transcript': '', 'provider': 'local'}
```

The transcript is empty because the smoke test is only a tone. It still verifies that the local Whisper model can load and run.

Restart the Hermes gateway after installing STT:

```bash
systemctl --user restart hermes-gateway-signal.service
```

From Signal, send:

```text
/reset
```

Then send a voice note. Hermes should transcribe it and answer normally.

## 15. Response Length Tuning

Very short Signal replies were not caused by the Signal transport itself. The raw `llama.cpp` endpoint and direct Hermes chat both produced normal-length replies.

We made two conservative config changes:

```bash
~/.local/bin/hermes config set model.max_tokens 2048
~/.local/bin/hermes config set display.personality helpful
systemctl --user restart hermes-gateway-signal.service
```

If a Signal conversation starts giving strange one-word replies, send:

```text
/reset
```

That starts a fresh Hermes session for the Signal chat.

If `/reset` does not clear it, rotate the active Signal session from the machine and restart the gateway:

```bash
systemctl --user stop hermes-gateway-signal.service

cd ~/.hermes/hermes-agent
~/.hermes/hermes-agent/venv/bin/python - <<'PY'
from pathlib import Path
from gateway.config import load_gateway_config
from gateway.session import SessionStore

store = SessionStore(Path.home() / ".hermes" / "sessions", load_gateway_config())
store._ensure_loaded()

signal_keys = [
    key for key, entry in store._entries.items()
    if getattr(entry.platform, "value", None) == "signal" and entry.chat_type == "dm"
]

if not signal_keys:
    raise SystemExit("No Signal DM session mapping found")

for key in signal_keys:
    old = store._entries[key].session_id
    new_entry = store.reset_session(key)
    print(f"reset Signal session: {old} -> {new_entry.session_id}")
PY

systemctl --user start hermes-gateway-signal.service
```

## 16. User systemd Services

We created three user services:

```text
~/.config/systemd/user/hermes-llama-qwen.service
~/.config/systemd/user/hermes-signal-cli.service
~/.config/systemd/user/hermes-gateway-signal.service
```

### llama.cpp Service

```ini
[Unit]
Description=Hermes local llama.cpp Qwen server
After=network-online.target

[Service]
Type=simple
WorkingDirectory=/home/cukaric
EnvironmentFile=/home/cukaric/.hermes/.env
Environment=PATH=/home/cukaric/.local/bin:/home/cukaric/.cargo/bin:/usr/local/bin:/usr/bin:/bin
ExecStart=/home/cukaric/.local/bin/llama-server -m /home/cukaric/.local/share/llama.cpp/models/qwen3.6-aeon-q4_k_m.gguf -c 65536 -t 16 --host 127.0.0.1 --port 18080 -a qwen-3.6-local --chat-template chatml --reasoning off --reasoning-budget 0 --reasoning-format deepseek --no-webui
Restart=on-failure
RestartSec=5
TimeoutStopSec=60

[Install]
WantedBy=default.target
```

### Signal Service

```ini
[Unit]
Description=Signal CLI HTTP daemon for Hermes
After=network-online.target

[Service]
Type=simple
Environment=PATH=/home/cukaric/.local/bin:/home/cukaric/.cargo/bin:/usr/local/bin:/usr/bin:/bin
ExecStart=/home/cukaric/.local/bin/signal-cli --account <SIGNAL_ACCOUNT_NUMBER> daemon --http 127.0.0.1:18081
Restart=on-failure
RestartSec=5
TimeoutStopSec=60

[Install]
WantedBy=default.target
```

### Gateway Service

```ini
[Unit]
Description=Hermes Gateway for Signal
After=network-online.target hermes-llama-qwen.service hermes-signal-cli.service
Wants=hermes-llama-qwen.service hermes-signal-cli.service

[Service]
Type=simple
WorkingDirectory=/home/cukaric
Environment=PATH=/home/cukaric/.local/bin:/home/cukaric/.cargo/bin:/usr/local/bin:/usr/bin:/bin
ExecStart=/home/cukaric/.local/bin/hermes gateway run --accept-hooks
Restart=on-failure
RestartSec=5
TimeoutStopSec=240

[Install]
WantedBy=default.target
```

Reload systemd after creating or editing services:

```bash
systemctl --user daemon-reload
```

We do not enable these services by default. That keeps Hermes from starting automatically when the computer turns on.

Make sure they are disabled:

```bash
systemctl --user disable hermes-llama-qwen.service hermes-signal-cli.service hermes-gateway-signal.service
```

Manual start, stop, status, and logs are handled by:

```bash
scripts/hermes-signal-stack.sh
```

The script starts `llama.cpp` and `signal-cli`, waits until both local health checks pass, then restarts the Hermes gateway so it connects after its dependencies are ready.

Start the full stack after reboot:

```bash
scripts/hermes-signal-stack.sh start
```

Stop it:

```bash
scripts/hermes-signal-stack.sh stop
```

Restart it:

```bash
scripts/hermes-signal-stack.sh restart
```

Check status:

```bash
scripts/hermes-signal-stack.sh status
```

Show recent logs:

```bash
scripts/hermes-signal-stack.sh logs
```

If you later decide services should start automatically, use `systemctl --user enable --now ...` and optionally `sudo loginctl enable-linger "$USER"`. For the current manual setup, do not run those.

## 17. Security Notes

Signal is generally a safer default than Telegram for this kind of personal assistant because Signal uses end-to-end encryption by default. However, the assistant still sees plaintext once messages reach this machine.

Main risks and mitigations:

- Linked device trust: Signal treats this computer as a trusted linked device. Keep the machine locked and remove the linked device from the phone if compromised.
- Local HTTP daemons: both `llama-server` and `signal-cli` are bound to `127.0.0.1`, not the LAN.
- Model API key exposure: keep the key in `~/.hermes/.env` with `chmod 600`; avoid putting it in process arguments.
- Unauthorized senders: keep `SIGNAL_ALLOW_ALL_USERS=false` and restrict `SIGNAL_ALLOWED_USERS`.
- Prompt injection: messages can tell the agent to do unsafe things. Keep Hermes hooks/tool permissions conservative and review any command that touches files, credentials, network, or system state.
- Secrets in logs: do not send passwords, private keys, recovery codes, or tokens through Signal messages to the agent.
- Supply chain risk: use official release sources, verify checksums where available, and avoid third-party native crypto binaries.
- Public exposure: do not bind `llama-server`, Signal HTTP, or Hermes gateway to `0.0.0.0` unless there is a separate authentication and firewall plan.

## 18. Useful Commands

Manual stack script:

```bash
scripts/hermes-signal-stack.sh start
scripts/hermes-signal-stack.sh stop
scripts/hermes-signal-stack.sh restart
scripts/hermes-signal-stack.sh status
scripts/hermes-signal-stack.sh logs
scripts/hermes-signal-stack.sh disable-autostart
```

Hermes:

```bash
~/.local/bin/hermes status
~/.local/bin/hermes chat -q 'Reply with exactly: ready' -Q --ignore-rules --max-turns 1
tail -n 80 ~/.hermes/logs/gateway.log
```

Signal:

```bash
~/.local/bin/signal-cli --version
curl --max-time 5 http://127.0.0.1:18081/api/v1/check
```

llama.cpp:

```bash
curl --max-time 5 http://127.0.0.1:18080/health
```

systemd:

```bash
systemctl --user daemon-reload
systemctl --user restart hermes-llama-qwen.service
systemctl --user restart hermes-signal-cli.service
systemctl --user restart hermes-gateway-signal.service
systemctl --user --no-pager --full status hermes-llama-qwen.service hermes-signal-cli.service hermes-gateway-signal.service
journalctl --user -u hermes-gateway-signal.service -n 100 --no-pager
```
