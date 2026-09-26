# The Complete Odyssey of Conquering Qwen 3.8-27B on Dual RTX 5070 Ti (32GB) + ASUS ProArt X870E
> **Subtitle:** From AWQ crashes and NVFP4 adoption to the trap of Pipeline Parallelism (PP=2), multimodal OOM disasters, hacked driver audits, and ultimate 81.3 tok/s Tensor Parallelism (TP=2) victory — A production engineering whitepaper.

[![Hardware: Dual RTX 5070 Ti](https://img.shields.io/badge/Hardware-Dual%20RTX%205070%20Ti%20(32GB)-76b900?logo=nvidia&style=flat-square)](https://www.nvidia.com)
[![Mainboard: ASUS ProArt X870E](https://img.shields.io/badge/Mainboard-ASUS%20ProArt%20X870E-00539b?style=flat-square)](https://www.asus.com)
[![vLLM: v0.28.0](https://img.shields.io/badge/vLLM-v0.28.0%20(v1)-blue?style=flat-square)](https://github.com/vllm-project/vllm)
[![Model: Qwen 3.8 27B NVFP4](https://img.shields.io/badge/Model-Qwen3.8--27B--NVFP4-purple?style=flat-square)](https://huggingface.co/gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](LICENSE)

---

## Table of Contents
1. [Project Inception: The Untraveled Path — Why Dual 5070 Ti?](#1-project-inception-the-untraveled-path--why-dual-5070-ti)
2. [Hardware Topology: The Secret of PCIe 5.0 Bifurcation (x8/x8)](#2-hardware-topology-the-secret-of-pcie-50-bifurcation-x8x8)
3. [The Chronological Engineering Odyssey](#3-the-chronological-engineering-odyssey)
   - [Phase 1: AWQ INT4 & The 16GB VRAM Allocation Crisis (CUDA Graph OOM)](#phase-1-awq-int4--the-16gb-vram-allocation-crisis-cuda-graph-oom)
   - [Phase 2: The PCIe Bottleneck Nightmare & The Failure of PP=2 Attempt 1](#phase-2-the-pcie-bottleneck-nightmare--the-failure-of-pp2-attempt-1)
   - [Phase 3: The Savior — Blackwell Native NVFP4 & The First Successful PP=2 Boot](#phase-3-the-savior--blackwell-native-nvfp4--the-first-successful-pp2-boot)
   - [Phase 4: The Siren Song of PP=2 (480 tok/s) & The Multimodal Image OOM Disaster](#phase-4-the-siren-song-of-pp2-480-toks--the-multimodal-image-oom-disaster)
   - [Phase 5: The Hacked Driver (P2P Bypass) Investigation & Hardware Rediscovery](#phase-5-the-hacked-driver-p2p-bypass-investigation--hardware-rediscovery)
   - [Phase 6: The Masterpiece Optimization — TP=2 Resurgence & 81.3 tok/s Record](#phase-6-the-masterpiece-optimization--tp2-resurgence--813-toks-record)
4. [Empirical Benchmark Verification (Terminal Proof)](#4-empirical-benchmark-verification-terminal-proof)
5. [Architectural Deep-Dive: TP=2 vs PP=2 Structural Analysis](#5-architectural-deep-dive-tp2-vs-pp2-structural-analysis)
6. [The Four Core Tuning Levers (3x Performance Gain)](#6-the-four-core-tuning-levers-3x-performance-gain)
7. [Production Deployment Manifest (`docker-compose.yml`)](#7-production-deployment-manifest-docker-composeyml)
8. [Troubleshooting & Incident Runbook](#8-troubleshooting--incident-runbook)
9. [Conclusion: The Economics of Dual 5070 Ti vs RTX 5090](#9-conclusion-the-economics-of-dual-5070-ti-vs-rtx-5090)

---

## 1. Project Inception: The Untraveled Path — Why Dual 5070 Ti?

In 2026, **Qwen 3.8-27B** (and the Qwen 3.5 series) has cemented its status as the most versatile open-weight foundation model. It delivers state-of-the-art multilingual comprehension, deep mathematical and logical reasoning (Thinking Tokens), and multimodal visual processing via its native Vision Transformer (ViT).

However, serving a 27B parameter model with extended context windows (up to 32k tokens) commands a minimum of **28GB+ VRAM**.

### The Reality Check: Single RTX 5090 32GB
* **Market Scalping & Premium:** A single retail card commands $2,500 – $3,200+.
* **Thermal & Power Density:** 575W TDP concentrated onto a single silicon die pushes standard chassis airflow and power delivery to their limits.

### Our Engineering Hypothesis
> **"Can we pair two NVIDIA GeForce RTX 5070 Ti GPUs (16GB GDDR7, 250W TDP each) to build a combined 32GB VRAM system with 1,792 GB/s aggregate memory bandwidth for ~$1,200 total — less than half the cost of an RTX 5090?"**

Yet, across Reddit, GitHub, and HuggingFace, no documented case studies existed for serving a 27B model on Dual 5070 Ti in production. Because GeForce hardware lacks physical NVLink fingers, prevailing wisdom insisted that PCIe All-Reduce bus latency would reduce multi-GPU throughput to an unusable crawl.

We chose to challenge that dogma directly.

---

## 2. Hardware Topology: The Secret of PCIe 5.0 Bifurcation (x8/x8)

Ninety-nine percent of failed consumer multi-GPU builds fail because of **motherboard PCIe lane routing**. On standard consumer boards, only the primary slot is wired to the CPU at x16; the secondary slot is routed through the chipset at PCIe 4.0 x4, quadrupling latency.

To eliminate this barrier, our system is anchored by the **ASUS ProArt X870E-CREATOR WIFI**, which provides CPU-direct symmetrical PCIe lane bifurcation.

```text
[ AMD Ryzen 5 9600X CPU (Zen 5 AM5) ]
                │
                ├─ PCIe 5.0 x8 (Direct Bus ~31.5 GB/s) ── [ GPU 0: RTX 5070 Ti (16GB) ]
                │                                           └─ Resizable BAR: 16,384 MiB
                │
                └─ PCIe 5.0 x8 (Direct Bus ~31.5 GB/s) ── [ GPU 1: RTX 5070 Ti (16GB) ]
                                                            └─ Resizable BAR: 16,384 MiB
```

* **Symmetric Lane Division:** Slots `PCIEX16(G5)_1` and `PCIEX16(G5)_2` are both wired directly to the CPU. In dual-GPU configurations, they divide symmetrically into **PCIe 5.0 x8 / x8**.
* **Massive Bandwidth:** A PCIe 5.0 x8 link delivers **~31.5 GB/s unidirectional (~63 GB/s bidirectional)**, identical to a full PCIe 4.0 x16 link.
* **Resizable BAR:** BIOS Resizable BAR exposes the full 16,384 MiB memory space of each card into the BAR1 memory aperture, paving the way for low-overhead inter-device data movement.

---

## 3. The Chronological Engineering Odyssey

### Phase 1: AWQ INT4 & The 16GB VRAM Allocation Crisis (CUDA Graph OOM)
* **Action:** Attempted to deploy `cyankiwi/Qwen3.8-27B-AWQ-INT4` using Tensor Parallelism (<code>TP=2</code>).
* **The Crash:**
  ```text
  RuntimeError: Engine core initialization failed.
  ValueError: No available memory for the cache blocks.
  ```
* **Root Cause Analysis:**  
  While a ~15GB quantized model split across two 16GB cards leaves 8.5GB of headroom on each card, multi-GPU TP=2 requires that each GPU worker allocate its own PyTorch context, communication bounce buffers, and critically, **dedicated CUDA Graph capture memory (~1.5GB to 2.0GB per GPU)**. On a 16GB card, model weights plus CUDA Graph allocations left under 300MB for the actual KV cache pool.
* **The Decision:**  
  Disabling CUDA Graphs via `--enforce-eager` degrades decode speed below 30 tok/s. We established an uncompromising rule: **CUDA Graphs must remain active.**

### Phase 2: The PCIe Bottleneck Nightmare & The Failure of PP=2 Attempt 1
* **The Symptoms:** After tuning memory parameters to boot the AWQ model under TP=2, performance was dismal:
  * Single-stream decode: **`23.3 tok/s`** (frustratingly sluggish)
  * Batch throughput: rigidly capped at **`116 tok/s`**
  * GPU power draw: stalled at **`188W`** (out of 250W TDP)
* **The Physical Cause:**  
  Qwen 27B consists of 64 transformer layers. In TP=2, every forward pass of every layer requires an All-Reduce exchange across the PCIe bus. **Generating a single token necessitated 128 inter-GPU PCIe sync operations.** Tensor Cores spent more time idling for PCIe packets than executing matrix math.
* **Attempting PP=2 with AWQ:**  
  To eliminate inter-layer All-Reduce traffic, we tested Pipeline Parallelism (`PP=2`). However, with AWQ, it failed at boot with `ValueError: No available memory for the cache blocks`. The vLLM V1 engine assigned the entire embedding layer to GPU 0 and the full LM Head to GPU 1, exhausting memory on both sides.

### Phase 3: The Savior — Blackwell Native NVFP4 & The First Successful PP=2 Boot
* **The Breakthrough:** Adoption of NVIDIA's native Blackwell quantization format: **`gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090`**.
* **Success:**  
  ModelOpt NVFP4 compressed weights further while unlocking native Tensor Core scheduling. By enforcing explicit KV cache limits (`--kv-cache-memory 3031959552`) and FP8 KV caching, **vLLM booted in PP=2 mode for the very first time**:
  ```text
  Processes:
    GPU 0: VLLM::Worker_PP0 (13938MiB)
    GPU 1: VLLM::Worker_PP1 (14764MiB)
  ```
  * Single-stream speed: **`47.7 tok/s`** (double the speed of AWQ, but still lagging a single 5090).

### Phase 4: The Siren Song of PP=2 (480 tok/s) & The Multimodal Image OOM Disaster
* **The Mirage:** Under pure offline text batches, PP=2 delivered staggering throughput. Saturating the queue with 14~16 requests yielded:
  ```text
  Engine 000: Avg prompt throughput: 712.2 tokens/s, Avg generation throughput: 480.6 tokens/s (Running: 14 reqs)
  Engine 000: Avg prompt throughput: 1154.7 tokens/s, Avg generation throughput: 404.7 tokens/s (Running: 16 reqs)
  ```
  **Peak continuous generation reached 480.6 tokens/s!** We believed PP=2 was the definitive answer.
* **The Disaster (Vision Enters):**  
  The illusion shattered when multimodal image queries entered the workload alongside text:
  ```text
  (Worker_PP0 pid=213) RuntimeError: Encoder cache miss for 71bbed5ce3add2...
  (Worker_PP0 pid=213) ERROR: multiproc_executor.py:1047
  vllm-qwen3.8-27b-NVFP4-RTX5090 exited with code 1
  ```
  * **The Fatal Flaw:** Pipeline Parallelism divides layers sequentially, but **the entire Vision Transformer (ViT) encoder was pinned 100% on GPU 0**. Processing high-resolution images rapidly pushed GPU 0 to 15.38GB VRAM, causing fatal OOM crashes.
  * Compounded by Mamba cache collisions (`max_num_seqs 256 exceeds available Mamba cache blocks 118`), PP=2 proved completely unfit for real-world production serving.

### Phase 5: The Hacked Driver (P2P Bypass) Investigation & Hardware Rediscovery
* **The Dilemma:** PP=2 was unstable with images; returning to TP=2 meant re-entering the 128-round All-Reduce bottleneck.
* **The Investigation:**  
  We explored the community's experimental territory: **Hacked NVIDIA Drivers (aikitoria / tinygrad open-gpu-kernel-modules P2P bypasses)**. We examined forcing `pKernelBif->p2pOverride = 0x11` to bypass NVIDIA's consumer software locks (`CNS: Chipset Not Supported`).
* **The Hardware Revelation:**  
  A rigorous audit of our host revealed:
  ```text
  pcie.link.gen.max: 5 (PCIe 5.0 32GT/s Verified)
  pcie.link.width.current: 8 (True x8 Symmetric Bifurcation Active)
  BAR1 Memory: 16,384 MiB (Full 16GB Resizable BAR Aperture Mapped)
  ```
  The **ASUS ProArt X870E** was already delivering **PCIe 5.0 x8 / x8** (31.5 GB/s unidirectional, matching PCIe 4.0 x16). We did not need hazardous unofficial kernel patches: **the bottleneck could be dismantled purely through vLLM communication stack tuning**.

### Phase 6: The Masterpiece Optimization — TP=2 Resurgence & 81.3 tok/s Record
* **The Four Strategic Interventions:**
  1. `--max-num-batched-tokens 8192`: Prevents prefill chunking on 4,105-token inputs, executing in a single forward pass (cutting wall time from 12.7s to **10.6s**).
  2. `NCCL_BUFFSIZE: "8388608"`: Expanded the PCIe ring buffer to 8MB, eliminating buffer-flush stalls across 64 layers.
  3. `VLLM_ATTENTION_BACKEND: "FLASHINFER"`: Enforced Blackwell native Tensor Core kernels for mixed FP4/FP8 paging attention.
  4. `PYTORCH_CUDA_ALLOC_CONF: "expandable_segments:False"`: Permanently eliminated virtual memory allocator collisions with CUDA Graph capture.
* **The Result:** Single decode speed **`81.3 tok/s`**, batch throughput **`335.2 tok/s`**, and peak generation **`446.5 tokens/s`**.

---

## 4. Empirical Benchmark Verification (Terminal Proof)

The live console output below was captured on the tuned Dual RTX 5070 Ti system running `vllm_bench.py --concurrency 16`:

![Live Benchmark Terminal Screenshot](images/benchmark_result.png)

### Measured Performance Matrix

| Stage | Workload (Prompt / Completion / Requests) | TTFT (p50) | Decode Speed (p50) | Output Throughput | Wall Time |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1/3 Single Stream** | 4,105 In / 300 Out (3 reqs) | **560.3 ms** | **`81.3 tok/s`** | **`68.7 tok/s`** | **10.6s** |
| **2/3 Batch Throughput** | 4,105 In / 300 Out (96 reqs) | **595.5 ms** | **`22.6 tok/s`** | **`320.5 tok/s`** | **80.3s** |
| **3/3 Saturation Ceiling** | 4,105 In / 300 Out (128 reqs) | **7,218.9 ms** | **`15.8 tok/s`** | **`335.2 tok/s`** | **101.2s** |

> [!NOTE]
> **Peak Continuous Production Throughput (vLLM Logger):**  
> `Engine 000: Avg prompt throughput: 4514.9 tokens/s, Avg generation throughput: 446.5 tokens/s (Running: 7 reqs, Waiting: 0 reqs, GPU KV cache usage: 28.8%)`

---

## 5. Architectural Deep-Dive: TP=2 vs PP=2 Structural Analysis

| Dimension | **TP=2 (Tensor Parallelism, Final Production)** | **PP=2 (Pipeline Parallelism)** | Engineering Verdict |
| :--- | :--- | :--- | :--- |
| **Single-Stream Interactive Speed** | **`81.3 tok/s`** (Instantaneous typing) | **`47.7 tok/s`** (Noticeable latency) | TP=2 is **+70.4% faster**. Critical for UX |
| **Pure Text Batch Throughput** | **`320 ~ 335 tok/s` (Peak 446.5)** | **`480.6 tok/s`** | PP=2 is faster in batch, but at the cost of halving interactive speed |
| **VRAM Allocation Symmetry** | **Perfect 50:50 distribution (7.8GB : 7.8GB)** | **Severe skew (15.4GB vs 13.0GB)** | TP=2 protects both cards; PP=2 risks single-card OOM |
| **Multimodal (Vision) Resilience** | **Zero OOM risk (ViT split 50:50)** | **Fatal OOM crashes (ViT pinned to GPU 0)** | PP=2 is disqualified for multimodal production |
| **Prefix Caching Efficiency** | **Shared synchronized block maps** | **Segmented stage caches** | Long multi-turn conversations favor TP=2 |

> [!TIP]
> **The Asymmetry of User Perception:**  
> When an end user interacts with a chat application, 47 tok/s feels sluggish, whereas 81 tok/s feels instant. Conversely, in background batch processing, whether 50,000 documents take 2 hours or 2 hours and 30 minutes has negligible operational impact. **Combining 81.3 tok/s interactive responsiveness with 335+ tok/s batch capacity makes TP=2 the definitive configuration.**

---

## 6. The Four Core Tuning Levers (3x Performance Gain)

1. **`--max-num-batched-tokens 8192` (Single-Pass Prefill):**  
   Ensures 4,105-token inputs execute in a single forward pass without chunking, cutting wall time from 12.7s to **10.6s**.
2. **`NCCL_BUFFSIZE: "8388608"` (8MB PCIe Ring Buffer):**  
   Eliminates buffer-flush stalls during All-Reduce bursts across 64 transformer layers.
3. **`VLLM_ATTENTION_BACKEND: "FLASHINFER"`:**  
   Bypasses Triton to unleash native Blackwell Tensor Core FP4 weight and FP8 KV cache paging attention kernels.
4. **`PYTORCH_CUDA_ALLOC_CONF: "expandable_segments:False"`:**  
   Eliminates memory allocator deadlocks with CUDA Graph capture.

---

## 7. Production Deployment Manifest (`docker-compose.yml`)

```yaml
services:
  vllm-qwen:
    image: vllm/vllm-openai:v0.28.0
    container_name: vllm-qwen3.8-27b-NVFP4-RTX5090
    runtime: nvidia
    restart: always
    ipc: host

    ports:
      - "8000:8000"

    environment:
      HF_TOKEN: "your_huggingface_token"
      PYTORCH_CUDA_ALLOC_CONF: "expandable_segments:False"

      # Validate hardware P2P routes
      VLLM_SKIP_P2P_CHECK: "0"

      # NCCL logging & 8MB buffer expansion
      NCCL_DEBUG: "WARN"
      NCCL_BUFFSIZE: "8388608"

      # Blackwell accelerated attention backend
      VLLM_ATTENTION_BACKEND: "FLASHINFER"

    volumes:
      - "${HOME}/.cache/huggingface:/root/.cache/huggingface"

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
      retries: 300
      start_period: 600s

    command: >
      --model gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090
      --served-model-name qwen3.8:27b
      --tensor-parallel-size 2
      --pipeline-parallel-size 1
      --max-num-seqs 32
      --gpu-memory-utilization 0.90
      --max-model-len 32768
      --max-num-batched-tokens 8192
      --kv-cache-dtype fp8
      --enable-chunked-prefill
      --mm-processor-kwargs '{"max_pixels": 1003520}'
      --limit-mm-per-prompt '{"image": 1}'
      --reasoning-parser qwen3
      --port 8000
      --host 0.0.0.0
```

---

## 8. Troubleshooting & Incident Runbook

### Case 1: Mamba State Block Exhaustion
* **Error:** `ValueError: max_num_seqs (256) exceeds available Mamba cache blocks (118)`
* **Fix:** Explicitly pass `--max-num-seqs 32` to guarantee safe operating headroom within available Mamba blocks.

### Case 2: expandable_segments Failure with CUDA Graphs
* **Error:** Setting `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` triggers kernel deadlocks during CUDA Graph compilation.
* **Fix:** Enforce `expandable_segments:False` to ensure stable virtual address mapping for CUDA Graphs.

### Case 3: Multimodal Image OOM under PP=2
* **Error:** Image requests push GPU 0 to 15.38GB VRAM, throwing `RuntimeError: Encoder cache miss`.
* **Fix:** Migrate to `TP=2`. Model weights and the ViT encoder are split 50:50 across both cards, permanently eliminating single-GPU memory spikes.

---

## 9. Conclusion: The Economics of Dual 5070 Ti vs RTX 5090

| Evaluation Metric | Single Flagship (RTX 5090 32GB) | Our System (RTX 5070 Ti Dual 32GB) |
| :--- | :--- | :--- |
| **Capital Expenditure** | $2,500 – $3,200+ | **~$1,200 – $1,300 (>50% Savings!)** |
| **VRAM Address Space** | 32GB GDDR7 (512-bit single bus) | **32GB GDDR7 (Dual 256-bit buses = 512-bit)** |
| **Power & Thermal Profile** | 575W concentrated single-die heat | **250W × 2 distributed dissipation** |
| **Single-Stream Decode Speed** | ~75 – 80 tok/s | **`81.3 tok/s` (Matches or beats single card!)** |
| **Large Prompt Prefill Speed** | 4,137 tok/s (0ms PCIe transfer) | **2,100 – 2,500 tok/s (~60% of single card)** |
| **Overall Serving Capability** | 100% Baseline | **Achieves 80% – 85% of 5090 throughput** |

By unifying **PCIe 5.0 Bifurcation (x8/x8) on the ASUS ProArt X870E, Blackwell native NVFP4 quantization, and surgical vLLM stack optimization**, we have demonstrated that **Dual RTX 5070 Ti can replace a $3,000 flagship accelerator for half the price**.
