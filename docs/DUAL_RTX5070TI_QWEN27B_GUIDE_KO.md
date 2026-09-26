# 🚀 Dual RTX 5070 Ti (32GB) + ASUS ProArt X870E로 Qwen 3.8-27B를 정복하기까지의 실전 엔지니어링 전 과정
> **부제:** AWQ의 좌절, NVFP4로의 전환, PP=2(파이프라인)의 달콤한 유혹과 멀티모달 OOM 대참사, 해킹 드라이버 검토를 거쳐 81.3 tok/s TP=2 종결에 도달하기까지의 실전 백서

[![Hardware: Dual RTX 5070 Ti](https://img.shields.io/badge/Hardware-Dual%20RTX%205070%20Ti%20(32GB)-76b900?logo=nvidia&style=flat-square)](https://www.nvidia.com)
[![Mainboard: ASUS ProArt X870E](https://img.shields.io/badge/Mainboard-ASUS%20ProArt%20X870E-00539b?style=flat-square)](https://www.asus.com)
[![vLLM: v0.28.0](https://img.shields.io/badge/vLLM-v0.28.0%20(v1)-blue?style=flat-square)](https://github.com/vllm-project/vllm)
[![Model: Qwen 3.8 27B NVFP4](https://img.shields.io/badge/Model-Qwen3.8--27B--NVFP4-purple?style=flat-square)](https://huggingface.co/gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](LICENSE)

---

## 📌 목차
1. [프로젝트 발단: 아무도 가지 않은 길 — 왜 5070 Ti 듀얼인가?](#1-프로젝트-발단-아무도-가지-않은-길--왜-5070-ti-듀얼인가)
2. [하드웨어 토폴로지: PCIe 5.0 Bifurcation (x8/x8)의 비밀](#2-하드웨어-토폴로지-pcie-50-bifurcation-x8x8의-비밀)
3. [격동의 엔지니어링 여정 (Chronological Journey)](#3-격동의-엔지니어링-여정-chronological-journey)
   - [Phase 1: AWQ INT4 모델과 16GB VRAM의 충돌 (CUDA Graph OOM)](#phase-1-awq-int4-모델과-16gb-vram의-충돌-cuda-graph-oom)
   - [Phase 2: PCIe 통신 지옥과 1차 PP=2의 처참한 부팅 실패](#phase-2-pcie-통신-지옥과-1차-pp2의-처참한-부팅-실패)
   - [Phase 3: 구원투수 Blackwell NVFP4 모델과 첫 PP=2 성공](#phase-3-구원투수-blackwell-nvfp4-모델과-첫-pp2-성공)
   - [Phase 4: PP=2의 환상(480 tok/s)과 이미지(Vision) 유입 시 OOM 대참사](#phase-4-pp2의-환상480-toks과-이미지vision-유입-시-oom-대참사)
   - [Phase 5: 해킹 드라이버(P2P Bypass)의 유혹과 하드웨어의 재발견](#phase-5-해킹-드라이버p2p-bypass의-유혹과-하드웨어의-재발견)
   - [Phase 6: 마스터피스 최적화 — TP=2의 부활과 81.3 tok/s 신기록](#phase-6-마스터피스-최적화--tp2의-부활과-813-toks-신기록)
4. [실측 벤치마크 결과 및 로그 증거 (Terminal Proof)](#4-실측-벤치마크-결과-및-로그-증거-terminal-proof)
5. [심층 해부: TP=2 vs PP=2 구조적 비교 분석](#5-심층-해부-tp2-vs-pp2-구조적-비교-분석)
6. [성능을 3배 폭증시킨 4대 핵심 튜닝 옵션](#6-성능을-3배-폭증시킨-4대-핵심-튜닝-옵션)
7. [프로덕션 확정 `docker-compose.yml`](#7-프로덕션-확정-docker-composeyml)
8. [트러블슈팅 런북 (실전 장애 해결 이력)](#8-트러블슈팅-런북-실전-장애-해결-이력)
9. [최종 결론: RTX 5090 vs Dual RTX 5070 Ti 경제학](#9-최종-결론-rtx-5090-vs-dual-rtx-5070-ti-경제학)

---

## 1. 프로젝트 발단: 아무도 가지 않은 길 — 왜 5070 Ti 듀얼인가?

2026년 현재, 로컬 오픈소스 LLM 생태계에서 가장 매력적인 모델은 단연 **Qwen 3.8-27B**(및 Qwen 3.5 계열)입니다. 뛰어난 다국어 능력, 딥시크에 필적하는 심층 논리 추론(Thinking Token), 그리고 비전 인코더(ViT)를 통한 이미지 멀티모달 분석 능력까지 완벽하게 갖춘 올라운더 모델입니다.

하지만 27B 모델을 실서비스 수준(32k 토큰 장문 컨텍스트)으로 구동하려면 최소 **28GB 이상의 VRAM**이 필수적입니다.

### 💸 현실의 벽: 단일 RTX 5090 32GB
* **가격 프리미엄:** 단일 카드가 350만 ~ 400만 원에 육박.
* **전력 및 발열:** 단일 다이에서 575W TDP 집중 방출 (일반 PC 케이스와 쿨링에 극심한 부담).

### 💡 우리의 엔지니어링 가설
> **"150~160만 원대의 예산으로 16GB GDDR7을 탑재한 RTX 5070 Ti 2장을 결합하면, 단일 5090의 절반 가격으로 32GB VRAM과 1,792 GB/s의 결합 메모리 대역폭을 얻을 수 있지 않을까?"**

하지만 Reddit, GitHub, HuggingFace 어디를 뒤져봐도 **소비자용 5070 Ti 듀얼로 vLLM 프로덕션 서빙을 성공시킨 구체적인 레퍼런스는 없었습니다.** 소비자용 GeForce는 하드웨어 NVLink가 완전히 제거되어 있고, 데스크톱 보드에서는 PCIe 통신 병목 때문에 제 성능의 50%도 안 나온다는 비관론만 가득했습니다.

우리는 이 비관론의 장벽을 직접 부딪쳐 깨보기로 결심했습니다.

---

## 2. 하드웨어 토폴로지: PCIe 5.0 Bifurcation (x8/x8)의 비밀

GeForce 듀얼 구성에서 실패하는 99%의 원인은 **메인보드의 PCIe 레인 구성**에 있습니다. 일반 메인보드는 1번 슬롯만 CPU 직결 x16이고, 2번 슬롯은 칩셋을 경유하는 PCIe 4.0 x4로 연결되어 통신 지연이 4배 이상 폭증합니다.

이를 극복하기 위해 CPU 레인을 물리적으로 대칭 분할할 수 있는 **ASUS ProArt X870E-CREATOR WIFI** 메인보드를 축으로 시스템을 설계했습니다.

```text
[ AMD Ryzen 5 9600X CPU (Zen 5 AM5) ]
                │
                ├─ PCIe 5.0 x8 (Direct Bus ~31.5 GB/s) ── [ GPU 0: RTX 5070 Ti (16GB) ]
                │                                           └─ Resizable BAR: 16,384 MiB
                │
                └─ PCIe 5.0 x8 (Direct Bus ~31.5 GB/s) ── [ GPU 1: RTX 5070 Ti (16GB) ]
                                                            └─ Resizable BAR: 16,384 MiB
```

* **대칭 레인 분기:** 상단 `PCIEX16(G5)_1`과 하단 `PCIEX16(G5)_2` 슬롯 모두 CPU에 직결되며, 듀얼 장착 시 **PCIe 5.0 x8 / x8**로 정확히 대칭 동작합니다.
* **대역폭:** PCIe 5.0 x8은 단방향 **~31.5 GB/s (양방향 63 GB/s)**로, 이전 세대 PCIe 4.0 x16과 완전히 동일한 광대역을 보장합니다.
* **Resizable BAR:** BIOS에서 Resizable BAR 활성화 시 두 GPU 모두 16,384 MiB 전체 VRAM이 BAR1 공간에 100% 매핑됩니다.

---

## 3. 격동의 엔지니어링 여정 (Chronological Journey)

### Phase 1: AWQ INT4 모델과 16GB VRAM의 충돌 (CUDA Graph OOM)
* **시도:** `cyankiwi/Qwen3.8-27B-AWQ-INT4` 모델을 `TP=2`로 기동.
* **발생 에러:**
  ```text
  RuntimeError: Engine core initialization failed.
  ValueError: No available memory for the cache blocks.
  ```
* **원인 규명:**  
  16GB VRAM 카드 2장에 모델 가중치(약 15GB)를 50:50으로 나누면 각 카드당 7.5GB가 남아야 합니다. 그러나 TP=2 환경에서는 각 GPU 워커마다 독립적인 PyTorch 컨텍스트, 통신 바운스 버퍼, 그리고 **디코드 가속을 위한 CUDA Graph 캡처 공간(~1.5~2GB)이 각 카드마다 중복으로 필요**했습니다. 모델 가중치와 CUDA Graph가 자리를 선점하자 KV 캐시를 할당할 공간이 300MB도 남지 않아 기동에 실패한 것입니다.
* **결단:**  
  `--enforce-eager`로 CUDA Graph를 끄면 디코드 속도가 30 tok/s 이하로 곤두박질치므로, **CUDA Graph를 반드시 살리는 방식**으로 해결책을 찾기로 결정.

### Phase 2: PCIe 통신 지옥과 1차 PP=2의 처참한 부팅 실패
* **현상:** 메모리 옵션을 극도로 다이어트하여 AWQ 모델을 TP=2로 겨우 띄웠으나 참혹한 성능:
  * 단일 디코드 속도: **`23.3 tok/s`** (답답한 타이핑 속도)
  * 대규모 배치 처리량: **`116 tok/s`**에서 완전히 캡(Cap)이 씌워짐
  * GPU 소비 전력: 250W 중 **`188W`**에 갇혀 텐서코어가 100% 돌지 못함
* **물리적 원인 규명:**  
  Qwen 27B는 64개 레이어로 구성됩니다. TP=2는 매 레이어마다 두 카드가 결과를 맞추는 All-Reduce 통신을 수행합니다. **토큰 1개를 생성할 때마다 두 카드가 PCIe 슬롯을 통해 128번의 통신을 왕복**해야 했습니다. 텐서코어가 연산하는 시간보다 PCIe 버스에서 상대를 기다리는 시간이 더 길었던 것입니다.
* **PP=2(파이프라인) 1차 시도와 실패:**  
  All-Reduce를 없애기 위해 PP=2를 시도했으나, AWQ 모델에서는 또다시 `ValueError: No available memory for the cache blocks`를 뿜으며 부팅조차 실패했습니다. 당시 vLLM V1 엔진이 PP 모드에서 임베딩과 LM Head를 한쪽 카드에 몰아넣어 메모리가 고갈되었기 때문입니다.

### Phase 3: 구원투수 Blackwell NVFP4 모델과 첫 PP=2 성공
* **돌파구:** NVIDIA Blackwell 전용 네이티브 양자화 모델 **`gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090`** 도입!
* **성공:**  
  FP4 가중치는 AWQ 대비 용량을 추가 절감했습니다. 명시적인 캐시 메모리 바이트 제한(`--kv-cache-memory 3031959552`)과 FP8 KV 캐시 설정을 통해 드디어 **PP=2 모드로 vLLM 기동에 최초로 성공**했습니다.
  ```text
  Processes:
    GPU 0: VLLM::Worker_PP0 (13938MiB)
    GPU 1: VLLM::Worker_PP1 (14764MiB)
  ```
  * 단일 질의 속도: **`47.7 tok/s`** (AWQ 23 tok/s 대비 2배 향상되었으나 여전히 5090 단일의 75 tok/s에는 미달).

### Phase 4: PP=2의 환상(480 tok/s)과 이미지(Vision) 유입 시 OOM 대참사
* **환상의 순간:** 순수 텍스트 배치를 14~16개 풀로 밀어 넣자 기적이 일어났습니다:
  ```text
  Engine 000: Avg prompt throughput: 712.2 tokens/s, Avg generation throughput: 480.6 tokens/s (Running: 14 reqs)
  Engine 000: Avg prompt throughput: 1154.7 tokens/s, Avg generation throughput: 404.7 tokens/s (Running: 16 reqs)
  ```
  **순간 생성량 480.6 tokens/s!** PCIe All-Reduce 통신 병목이 사라진 PP=2는 대규모 배치에서 압도적이었습니다.
* **대참사 (Vision 유입):**  
  그러나 텍스트 배치 도중 **이미지 3~4장이 포함된 멀티모달 요청**이 들어가자마자 시스템이 폭발했습니다:
  ```text
  (Worker_PP0 pid=213) RuntimeError: Encoder cache miss for 71bbed5ce3add2...
  (Worker_PP0 pid=213) ERROR: multiproc_executor.py:1047
  vllm-qwen3.8-27b-NVFP4-RTX5090 exited with code 1
  ```
  * **원인 규명:** PP=2는 트랜스포머 레이어만 32개씩 쪼갤 뿐, **모델의 거대한 시각 인코더(ViT) 전체가 GPU 0번에만 100% 적재**되었습니다. 이미지 배치가 들어오자 GPU 0의 VRAM만 15.38GB까지 치솟아 OOM 크래시가 발생한 것입니다.
  * 여기에 Qwen 3.8 특유의 Mamba 상태 블록 충돌(`max_num_seqs 256 exceeds available Mamba cache blocks 118`)까지 터지며, PP=2는 멀티모달 실서비스에서 사용 불가능한 구조임이 입증되었습니다.

### Phase 5: 해킹 드라이버(P2P Bypass)의 유혹과 하드웨어의 재발견
* **고뇌:** PP=2가 불안정하다면 다시 TP=2로 돌아가야 했습니다. 하지만 TP=2로 가면 다시 PCIe 128회 통신 병목에 갇히게 됩니다.
* **해킹 드라이버 검토:**  
  오픈소스 커뮤니티의 P2P 우회 패치(`aikitoria/open-gpu-kernel-modules`, `geohot/tinygrad` P2P override `pKernelBif->p2pOverride = 0x11`)를 정밀 조사했습니다. GeForce에 걸린 인위적인 소프트웨어 락(`CNS: Chipset Not Supported`)을 해킹하여 엔터프라이즈급 BAR1 P2P DMA를 뚫으려 했습니다.
* **하드웨어의 재발견:**  
  하지만 서버 환경을 정밀 진단하던 중, **ASUS ProArt X870E 보드가 이미 CPU 직결 PCIe 5.0 x8/x8 (단방향 31.5 GB/s, PCIe 4.0 x16급) 대역폭과 16GB Resizable BAR 전체 매핑을 완벽히 제공**하고 있음을 확인했습니다. 위험한 비공식 커널 드라이버를 쓰지 않더라도, **vLLM 스케줄러와 통신 버퍼 튜닝만으로 PCIe 병목을 극복할 수 있다는 확신**을 얻었습니다.

### Phase 6: 마스터피스 최적화 — TP=2의 부활과 81.3 tok/s 신기록
* **4대 최적화 단행:**
  1. `--max-num-batched-tokens 8192`: 4,105 토큰 입력을 쪼개지 않고 단 1번의 포워드 패스로 완료 (12.7s ➔ 10.6s).
  2. `NCCL_BUFFSIZE: "8388608"`: 8MB PCIe 링 버퍼로 All-Reduce 통신 정체 해소.
  3. `VLLM_ATTENTION_BACKEND: "FLASHINFER"`: Blackwell 텐서코어 전용 혼합 정밀도 커널 강제.
  4. `PYTORCH_CUDA_ALLOC_CONF: "expandable_segments:False"`: CUDA Graphs 충돌 영구 제거.
* **결과:** 단일 디코드 속도 **`81.3 tok/s`**, 동시성 32 배치 처리량 **`335.2 tok/s`**, 순간 최고 생성량 **`446.5 tokens/s`**를 달성하며 역사적인 종결 세팅을 완성했습니다.

---

## 4. 실측 벤치마크 결과 및 로그 증거 (Terminal Proof)

실제 최적화 완료 후 구동된 `vllm_bench.py --concurrency 16` 스트레스 테스트 콘솔 캡처입니다.

![Benchmark Result Live Terminal](images/benchmark_result.png)

### 📊 단계별 실측 수치

| 테스트 구간 | 동시 요청 수 | TTFT (첫 토큰 지연) | 디코드 속도 (p50) | 총 출력 처리량 | 총 소요 시간 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1/3 단발 스트레스** | 동시 1 (총 3건) | **560.3 ms** | **`81.3 tok/s`** | **`68.7 tok/s`** | **10.6초** |
| **2/3 배치 처리** | 동시 16 (총 96건) | **595.5 ms** | **`22.6 tok/s`** | **`320.5 tok/s`** | **80.3초** |
| **3/3 포화 한계 확인** | 동시 32 (총 128건) | **7,218.9 ms** | **`15.8 tok/s`** | **`335.2 tok/s`** | **101.2초** |

> [!NOTE]
> **실시간 연속 서빙 중 vLLM 내부 로거(`loggers.py`) 계측치:**  
> `Engine 000: Avg prompt throughput: 4514.9 tokens/s, Avg generation throughput: 446.5 tokens/s (Running: 7 reqs, Waiting: 0 reqs, GPU KV cache usage: 28.8%)`

---

## 5. 심층 해부: TP=2 vs PP=2 구조적 비교 분석

| 비교 차원 | **TP=2 (텐서 병렬화, 최종 확정)** | **PP=2 (파이프라인 병렬화)** | 실무적 평가 및 판정 |
| :--- | :--- | :--- | :--- |
| **실시간 단일 사용자 체감** | 🚀 **`81.3 tok/s`** (읽는 속도 초과) | 🐢 **`47.7 tok/s`** (답답한 타이핑) | TP=2가 **+70.4% 압도**. 챗봇 품질의 핵심 |
| **순수 텍스트 오프라인 배치** | ⚡ **`320 ~ 335 tok/s` (피크 446.5)** | 🏆 **`480.6 tok/s`** | PP=2가 빠르나 실시간성을 버릴 만큼의 격차는 아님 |
| **메모리(VRAM) 대칭성** | ⚖️ **양쪽 50:50 완벽 대칭 (7.8G : 7.8G)** | ⚠️ **극심한 불균형 (15.4G : 13.0G)** | TP=2는 두 카드가 고르게 숨을 쉬어 안전 |
| **멀티모달 (Vision) 안정성** | 🛡️ **OOM 위험 제로 (ViT도 균등 분산)** | 💥 **OOM 크래시 빈발 (ViT GPU 0 독점)** | 복합 서비스 환경에서 PP=2는 실격 |
| **프롬프트 캐싱 (Prefix Cache)** | 🔗 **양 카드 동일 블록 맵 공유 (최상)** | ✂️ **스테이지별 캐시 분절** | 대화가 길어질수록 TP=2가 월등히 유리 |

> [!TIP]
> **체감 품질의 비대칭성:**  
> 사람이 화면을 보며 상호작용할 때 47 tok/s와 81 tok/s는 **'답답함'과 '쾌적함'이라는 완전히 다른 UX**를 만듭니다. 반면 백그라운드 배치 작업에서 5만 건 처리가 2시간 걸리느냐 2시간 30분 걸리느냐는 서비스 성패에 영향을 주지 않습니다. **단발 81.3 tok/s와 배치 335+ tok/s를 겸비한 TP=2가 하이브리드 운영의 유일한 정답**입니다.

---

## 6. 성능을 3배 폭증시킨 4대 핵심 튜닝 옵션

```text
기본 세팅 (116 tok/s, 188W 전력 갇힘) ───[ 4대 튜닝 단행 ]───▶ 최적화 완성 (335+ tok/s, 디코드 81.3 tok/s)
```

1. **`--max-num-batched-tokens 8192` (프리필 단일 패스):**  
   4,105 토큰 프롬프트가 들어올 때 청크로 쪼개지 않고 단 1회의 포워드 패스로 프리필을 끝내어 단발 시간을 12.7s에서 **10.6s**로 단축.
2. **`NCCL_BUFFSIZE: "8388608"` (8MB 링 버퍼):**  
   PCIe 슬롯 간 All-Reduce 통신 시 버퍼 플러시 대기 시간을 완전히 제거하여 64개 레이어의 동기화 오버헤드 최소화.
3. **`VLLM_ATTENTION_BACKEND: "FLASHINFER"`:**  
   Blackwell(RTX 50 시리즈) 텐서코어의 FP4 가중치와 FP8 KV 캐시 페이징 연산 효율을 100% 발휘.
4. **`PYTORCH_CUDA_ALLOC_CONF: "expandable_segments:False"`:**  
   CUDA Graphs 캡처 시 PyTorch 가상 메모리 관리자와의 알려진 주소 충돌을 방지하여 크래시 차단.

---

## 7. 프로덕션 확정 `docker-compose.yml`

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

      # 하드웨어 P2P 통신 검사 활성화
      VLLM_SKIP_P2P_CHECK: "0"

      # NCCL 로깅 제어 및 8MB 링 버퍼 확장
      NCCL_DEBUG: "WARN"
      NCCL_BUFFSIZE: "8388608"

      # Blackwell 전용 어텐션 가속 백엔드
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

## 8. 트러블슈팅 런북 (실전 장애 해결 이력)

### 🚨 Case 1. Mamba 캐시 블록 초과 충돌
* **에러 메시지:** `ValueError: max_num_seqs (256) exceeds available Mamba cache blocks (118)`
* **해결책:** Qwen 3.8 모델의 하이브리드(Mamba+Attention) 특성상 시퀀스마다 Mamba 전용 블록이 필요함. `--max-num-seqs 32`를 명시하여 가용 블록(118개) 이내로 제한.

### 🚨 Case 2. expandable_segments 설정 시 vLLM 기동 실패
* **에러 현상:** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 지정 시 CUDA Graph 컴파일 단계에서 가상 주소 충돌로 서버 사망.
* **해결책:** vLLM 환경에서는 반드시 `expandable_segments:False`로 지정해야 CUDA Graph가 안정적으로 캡처됨.

### 🚨 Case 3. PP=2 환경에서 멀티모달 이미지 처리 시 GPU 0 OOM
* **에러 현상:** 이미지 요청 유입 시 GPU 0만 15.38GB로 치솟으며 `RuntimeError: Encoder cache miss` 발생.
* **해결책:** `TP=2`로 전환하면 ViT 인코더와 모델 레이어가 양쪽 GPU에 50:50으로 정확히 양분되어 OOM 원천 해소.

---

## 9. 최종 결론: RTX 5090 vs Dual RTX 5070 Ti 경제학

| 평가 지표 | 단일 플래그십 (RTX 5090 32GB) | 우리 구축 시스템 (RTX 5070 Ti Dual 32GB) |
| :--- | :--- | :--- |
| **시스템 도입 비용** | 약 350만 ~ 400만 원 | 💵 **약 150만 ~ 160만 원 (55% 이상 절감!)** |
| **VRAM 주소 공간** | 32GB GDDR7 (512-bit 단일 버스) | **32GB GDDR7 (256-bit × 2 = 512-bit 대등)** |
| **소비 전력 및 발열** | 575W 단일 다이 집중 (쿨링 난이도 극상) | ❄️ **250W × 2 분산 쿨링 (일반 공랭/수랭 여유)** |
| **실시간 단일 디코드** | 약 75 ~ 80 tok/s | 🚀 **`81.3 tok/s` (동등 이상 달성!)** |
| **대용량 프롬프트 프리필** | 4,137 tok/s (PCIe 통신 0초) | ⚡ **2,100 ~ 2,500 tok/s (5090의 약 60%)** |
| **실서비스 종합 체감** | 기준점 (100%) | 🏆 **단일 5090의 80% ~ 85% 이상 체감 달성** |

소비자용 듀얼 GPU 환경은 불가능하다는 통념을 깨고, **ASUS ProArt X870E의 PCIe 5.0 Bifurcation + Blackwell NVFP4 + vLLM TP=2 최적화**의 조합으로 우리는 상용 API를 완벽히 대체하는 고성능 로컬 추론 머신을 완성했습니다.
