# ASUS ProArt X870E + RTX PRO 6000 GPU Server Setup Guide
> **Target Environment:** Ubuntu 24.04.4 LTS Server  
> **Hardware:** ASUS ProArt X870E-CREATOR WIFI + NVIDIA RTX PRO 6000 Ada (48GB VRAM)  
> **Deployment Scope:** NVIDIA Driver (v580 Open Kernel Module) → Docker & Compose v2 → NVIDIA Container Toolkit → vLLM (v0.28.0) Qwen3.8-27B Deployment & API Verification

---

## 📋 Table of Contents
1. [Prerequisites and System Inspection](#1-prerequisites-and-system-inspection)
2. [NVIDIA Driver Installation (v580 Open Module)](#2-nvidia-driver-installation)
3. [Docker and Docker Compose Installation](#3-docker-and-docker-compose-installation)
4. [NVIDIA Container Toolkit Installation & Docker Integration](#4-nvidia-container-toolkit-installation--docker-integration)
5. [vLLM Container Deployment (Qwen3.8-27B)](#5-vllm-container-deployment-qwen38-27b)
6. [API Functionality Testing & Verification](#6-api-functionality-testing--verification)
7. [Key Troubleshooting & Operational Tips](#7-key-troubleshooting--operational-tips)

---

## 1. Prerequisites and System Inspection

### 1.1 Essential Build Tools & Packages
Immediately after a fresh Ubuntu Server installation, install build tools required for kernel module builds (DKMS) and network utilities:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential dkms curl wget git jq net-tools pciutils
```

### 1.2 PCIe GPU Detection
Verify that the RTX PRO 6000 GPU installed in the PCIe slot on the ASUS ProArt X870E motherboard is properly detected:

```bash
lspci -nnk | grep -iE 'vga|3d|nvidia'
```
* Example output:
  ```text
  01:00.0 VGA compatible controller [0300]: NVIDIA Corporation AD102GL [RTX 6000 Ada Generation] ...
  ```

---

## 2. NVIDIA Driver Installation

Install the stable `580-open` kernel module driver from the official Ubuntu 24.04 LTS repository (consistent with the driver version used on the dual-GPU cluster).

### 2.1 Driver Installation
```bash
sudo apt install -y nvidia-driver-580-open
```

### 2.2 Enable Persistence Daemon
Enable GPU Persistence Mode so that the driver stays loaded in memory even when no GPU processes are active (essential for low inference startup latency):

```bash
sudo systemctl enable --now nvidia-persistenced
```

### 2.3 System Reboot & Verification
Reboot the system once to load the newly installed open kernel modules:

```bash
sudo reboot
```

After rebooting, reconnect via SSH and inspect the GPU status:
```bash
nvidia-smi
```

* Verification checklist:
  - **Driver Version**: `580.173.xx`
  - **CUDA Version**: `13.0` (or driver-supported release)
  - **GPU Name**: `NVIDIA RTX 6000 Ada Generation` (or RTX PRO 6000)
  - **Memory**: `49140MiB` (~48GB VRAM)
  - **Persistence-M**: `On`

---

## 3. Docker and Docker Compose Installation

Install the official Docker Engine and Compose v2 packages to isolate and deploy vLLM in containers.

### 3.1 Package Installation
```bash
sudo apt install -y docker.io docker-compose-v2
```

### 3.2 User Privileges and Service Activation
Add the current user (`$USER`) to the `docker` group to execute Docker commands without `sudo`:

```bash
# Add current user to docker group
sudo usermod -aG docker $USER

# Enable and start Docker service
sudo systemctl enable --now docker

# Apply group changes to current shell session
newgrp docker
```

### 3.3 Verify Installed Versions
```bash
docker --version
docker compose version
```

---

## 4. NVIDIA Container Toolkit Installation & Docker Integration

Configure the NVIDIA Container Toolkit so that Docker containers can directly access the host GPU via the container runtime.

### 4.1 Register NVIDIA Official APT Repository & GPG Key
```bash
# 1. Register GPG key
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

# 2. Register APT repository list
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# 3. Update index and install toolkit
sudo apt update
sudo apt install -y nvidia-container-toolkit
```

### 4.2 Configure Docker Runtime & Restart Daemon
```bash
# Configure nvidia as a Docker runtime
sudo nvidia-ctk runtime configure --runtime=docker

# Restart Docker daemon
sudo systemctl restart docker
```

### 4.3 GPU Passthrough Smoke Test
Confirm that the host GPU is correctly passed through into a Docker container:

```bash
docker run --rm --gpus all ubuntu nvidia-smi
```
* If the `nvidia-smi` output inside the container matches the host output, GPU passthrough configuration is complete.

---

## 5. vLLM Container Deployment (Qwen3.8-27B)

Deploy the production-tested **`vllm/vllm-openai:v0.28.0`** image with the **`cyankiwi/Qwen3.8-27B-AWQ-INT4`** model.

> 💡 **Architectural Advantage of RTX PRO 6000 (48GB VRAM):**  
> While dual RTX 5070 Ti setups (16GB × 2 = 32GB) require strict Tensor Parallelism (TP=2) and tight VRAM allocation limits, the **RTX PRO 6000 features a massive single 48GB VRAM pool**. This allows running on a single GPU (`TP=1`) without inter-GPU PCIe communication overhead, comfortably accommodating 16K–32K context windows and large batch sizes.

### 5.1 Create Working Directory
```bash
mkdir -p ~/docker-runtime/vllm
cd ~/docker-runtime/vllm
```

### 5.2 Create Environment Configuration (.env)
Define Hugging Face cache and PyTorch memory allocator settings:

```bash
cat <<'EOF' > ~/docker-runtime/vllm/.env
# Hugging Face Token (optional for public models)
HF_TOKEN=

# Prevent PyTorch VRAM memory fragmentation (essential for CUDA Graph capture)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
EOF

chmod 600 ~/docker-runtime/vllm/.env
```

### 5.3 Create `docker-compose.yml`

```bash
cat <<'EOF' > ~/docker-runtime/vllm/docker-compose.yml
services:
  vllm-qwen:
    image: vllm/vllm-openai:v0.28.0
    container_name: vllm-qwen3.8-27b
    runtime: nvidia
    restart: unless-stopped
    ipc: host
    ports:
      - "8000:8000"
    env_file:
      - .env
    volumes:
      - ~/.cache/huggingface:/root/.cache/huggingface
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 30
      start_period: 600s
    command: >
      --model cyankiwi/Qwen3.8-27B-AWQ-INT4
      --served-model-name qwen3.8:27b
      --tensor-parallel-size 1
      --gpu-memory-utilization 0.92
      --max-model-len 16384
      --max-num-seqs 16
      --max-num-batched-tokens 8192
      --kv-cache-dtype fp8
      --enable-chunked-prefill
      --enable-prefix-caching
      --reasoning-parser qwen3
      --port 8000
      --host 0.0.0.0
EOF
```

> **Parameter Breakdown:**
> - `--tensor-parallel-size 1`: Leverages the single RTX PRO 6000 48GB GPU (change to `2` if two cards are installed).
> - `--gpu-memory-utilization 0.92`: Allocates ~44GB of 48GB VRAM to vLLM (~19.5GB for weights + ~24.5GB for KV cache).
> - `--max-model-len 16384`: Supports 16K context window (can be extended to 32768 if needed).
> - `--max-num-batched-tokens 8192`: Prevents computation shrinkage from chunking and maximizes throughput.
> - `--enable-prefix-caching`: Drastically cuts TTFT by caching system prompts and repetitive instructions.
> - `--reasoning-parser qwen3`: Enables thought token tag parsing for Qwen 3.8 / DeepSeek reasoning models.

### 5.4 Start vLLM Service & Monitor Logs
```bash
# Start container in detached background mode
docker compose up -d

# Inspect startup logs and model downloading progress
docker compose logs -f
```

* **Startup Complete Message:**  
  When the Uvicorn server startup log appears at the end, the server is ready:
  ```text
  INFO:     Started server process [1]
  INFO:     Waiting for application startup.
  INFO:     Application startup complete.
  INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
  ```
  *(Press `Ctrl + C` to exit the log stream)*

---

## 6. API Functionality Testing & Verification

### 6.1 Health Check
```bash
curl -i http://localhost:8000/health
```
* **Success Response:** `HTTP/1.1 200 OK`

### 6.2 Inspect Model List Endpoint
```bash
curl -s http://localhost:8000/v1/models | jq .
```
* **Expected Response:**
  ```json
  {
    "object": "list",
    "data": [
      {
        "id": "qwen3.8:27b",
        "object": "model",
        "created": 1789635000,
        "owned_by": "vllm"
      }
    ]
  }
  ```

### 6.3 Chat Completion Inference Test (cURL)
```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.8:27b",
    "messages": [
      {"role": "system", "content": "You are an AI assistant running on RTX PRO 6000."},
      {"role": "user", "content": "Testing server environment. Please reply with a brief welcome and a 3-bullet summary in English."}
    ],
    "max_tokens": 200,
    "temperature": 0.7
  }' | jq .
```
* Verify that generated text appears in the `choices[0].message.content` field.

### 6.4 Real-time Streaming (SSE) Test
```bash
curl -N http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.8:27b",
    "messages": [{"role": "user", "content": "Count from 1 to 10, one number per line."}],
    "stream": true
  }'
```

---

## 7. Key Troubleshooting & Operational Tips

### 7.1 Motherboard (ASUS ProArt X870E) BIOS Settings
* **Above 4G Decoding:** Must be set to **Enabled** so that the large 48GB VRAM address space can be correctly mapped.
* **Re-Size BAR Support:** Recommended **Auto** or **Enabled** for direct GPU VRAM access efficiency.
* **PCIe Link Speed:** Verify that the card links at Gen4 or Gen5 (`lspci -vv -s 01:00.0 | grep -i lnksta`).

### 7.2 Disk Space Management
* The Qwen3.8-27B AWQ model requires approximately 21GB of storage space.
* Default Ubuntu Server LVM installations may limit the root partition to ~100GB. Verify via `df -h /`.
* Extend LVM volume if needed:
  ```bash
  sudo lvextend -l +100%FREE /dev/ubuntu-vg/ubuntu-lv
  sudo resize2fs /dev/ubuntu-vg/ubuntu-lv
  ```

### 7.3 Firewall Configuration for External Access
If external client machines need to connect to port `8000`:
```bash
sudo ufw allow 8000/tcp
```
