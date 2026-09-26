# ASUS ProArt X870E + RTX PRO 6000 GPU 서버 설치 가이드
> **대상 환경:** Ubuntu 24.04.4 LTS Server  
> **하드웨어:** ASUS ProArt X870E-CREATOR WIFI + NVIDIA RTX PRO 6000 (48GB)  
> **구축 범위:** NVIDIA 드라이버(v580) → Docker & Compose → NVIDIA Container Toolkit → vLLM(v0.28.0) Qwen3.8-27B 배포 및 API 검증

---

## 목차
1. [사전 준비 및 시스템 점검](#1-사전-준비-및-시스템-점검)
2. [NVIDIA 드라이버 설치 (v580 Open Module)](#2-nvidia-드라이버-설치)
3. [Docker 및 Docker Compose 설치](#3-docker-및-docker-compose-설치)
4. [NVIDIA Container Toolkit 설치 및 Docker 연동](#4-nvidia-container-toolkit-설치-및-docker-연동)
5. [vLLM 컨테이너 구성 (Qwen3.8-27B)](#5-vllm-컨테이너-구성-qwen38-27b)
6. [API 동작 테스트 및 검증](#6-api-동작-테스트-및-검증)
7. [주요 트러블슈팅 및 운영 팁](#7-주요-트러블슈팅-및-운영-팁)

---

## 1. 사전 준비 및 시스템 점검

### 1.1 기본 빌드 도구 및 필수 유틸리티 설치
우분투 서버 초기 설치 직후, 커널 모듈 빌드(DKMS)와 네트워크 통신에 필요한 필수 패키지를 설치합니다.

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential dkms curl wget git jq net-tools pciutils
```

### 1.2 PCIe GPU 인식 확인
메인보드(ASUS ProArt X870E)의 PCIe 슬롯에 장착된 RTX PRO 6000 그래픽카드가 정상 인식되는지 확인합니다.

```bash
lspci -nnk | grep -iE 'vga|3d|nvidia'
```
* 출력 예시:
  ```text
  01:00.0 VGA compatible controller [0300]: NVIDIA Corporation AD102GL [RTX 6000 Ada Generation] ...
  ```

---

## 2. NVIDIA 드라이버 설치

gpu2 머신과 동일하게 Ubuntu 24.04 LTS 공식 리포지토리의 안정화된 `580-open` 커널 모듈 드라이버를 설치합니다.

### 2.1 드라이버 설치
```bash
sudo apt install -y nvidia-driver-580-open
```

### 2.2 Persistence Daemon 활성화
GPU의 지속 모드(Persistence Mode)를 활성화하여 프로세스가 없을 때도 드라이버가 언로드되지 않도록 설정합니다. (추론 서버 구동 시 필수)

```bash
sudo systemctl enable --now nvidia-persistenced
```

### 2.3 시스템 재부팅 및 드라이버 검증
새로 설치된 커널 모듈을 로드하기 위해 시스템을 1회 재부팅합니다.

```bash
sudo reboot
```

재부팅 후 SSH로 다시 접속하여 GPU 상태를 확인합니다:
```bash
nvidia-smi
```

* 정상 출력 확인 항목:
  - **Driver Version**: `580.173.xx`
  - **CUDA Version**: `13.0` (또는 드라이버 지원 버전)
  - **GPU Name**: `NVIDIA RTX 6000 Ada Generation` (또는 RTX PRO 6000)
  - **Memory**: `49140MiB` (약 48GB VRAM)
  - **Persistence-M**: `On`

---

## 3. Docker 및 Docker Compose 설치

vLLM을 컨테이너 기반으로 격리 배포하기 위해 Ubuntu 24.04 공식 Docker 엔진과 Compose v2를 설치합니다.

### 3.1 패키지 설치
```bash
sudo apt install -y docker.io docker-compose-v2
```

### 3.2 사용자 권한 부여 및 서비스 등록
현재 로그인 계정(`$USER`)을 `docker` 그룹에 추가하여 `sudo` 없이 docker 명령어를 실행할 수 있도록 합니다.

```bash
# docker 그룹 등록
sudo usermod -aG docker $USER

# docker 서비스 자동 실행 활성화
sudo systemctl enable --now docker

# 현재 셸 세션에 그룹 즉시 적용
newgrp docker
```

### 3.3 설치 버전 확인
```bash
docker --version
docker compose version
```

---

## 4. NVIDIA Container Toolkit 설치 및 Docker 연동

컨테이너 내부에서 호스트 GPU를 직접 접근할 수 있도록 NVIDIA Container Toolkit을 설치하고 Docker 런타임에 등록합니다.

### 4.1 NVIDIA 공식 APT 저장소 키 및 리스트 등록
```bash
# 1. GPG 키 등록
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

# 2. APT 저장소 리스트 등록
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# 3. 패키지 인덱스 갱신 및 툴킷 설치
sudo apt update
sudo apt install -y nvidia-container-toolkit
```

### 4.2 Docker 런타임 구성 및 재시작
```bash
# Docker 기본 런타임에 nvidia 추가
sudo nvidia-ctk runtime configure --runtime=docker

# Docker 데몬 재시작
sudo systemctl restart docker
```

### 4.3 GPU 컨테이너 동작 테스트
호스트의 GPU가 Docker 컨테이너 내부로 올바르게 패스스루되는지 테스트합니다:

```bash
docker run --rm --gpus all ubuntu nvidia-smi
```
* 컨테이너 안에서 호스트와 동일한 `nvidia-smi` 테이블이 출력되면 GPU 패스스루 설정이 완벽히 완료된 것입니다.

---

## 5. vLLM 컨테이너 구성 (Qwen3.8-27B)

gpu2에서 안정성과 성능이 실측 검증된 **`vllm/vllm-openai:v0.28.0`** 이미지와 **`cyankiwi/Qwen3.8-27B-AWQ-INT4`** 모델을 배포합니다.

> **RTX PRO 6000 (48GB VRAM) 이점:**  
> gpu2(듀얼 5070 Ti, 16GB x 2 = 32GB)에서는 VRAM이 빠듯하여 TP=2 분할과 메모리 한도 설정이 엄격해야 했으나, **RTX PRO 6000은 단일 48GB VRAM**을 탑재하고 있어 단일 GPU(TP=1)로 통신 오버헤드 없이 구동되며, 32k 이상의 긴 컨텍스트와 대량의 동시 배치 처리도 여유롭게 소화합니다.

### 5.1 작업 디렉토리 생성
```bash
mkdir -p ~/docker-runtime/vllm
cd ~/docker-runtime/vllm
```

### 5.2 환경 변수 파일 (.env) 생성
Hugging Face 모델 다운로드 및 PyTorch 메모리 단편화 방지 설정을 정의합니다.

```bash
cat <<'EOF' > ~/docker-runtime/vllm/.env
# Hugging Face 토큰 (필요 시 본인 토큰 입력, 공개 모델은 빈값 유지 가능)
HF_TOKEN=

# PyTorch VRAM 메모리 단편화 방지 (CUDA Graph 안정화 필수)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
EOF

chmod 600 ~/docker-runtime/vllm/.env
```

### 5.3 `docker-compose.yml` 작성

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

> **주요 파라미터 해설:**
> - `--tensor-parallel-size 1`: 단일 RTX PRO 6000 48GB GPU를 활용합니다. (만약 2장이 장착되어 있다면 `2`로 변경)
> - `--gpu-memory-utilization 0.92`: 48GB 중 약 44GB를 vLLM에 할당 (가중치 약 19.5GB + KV Cache 약 24.5GB 확보).
> - `--max-model-len 16384`: 16K 토큰 지원 (필요 시 32768(32K)로 확장 가능).
> - `--max-num-batched-tokens 8192`: 청크 분할로 인한 연산 축소 방지 (성능 극대화).
> - `--enable-prefix-caching`: 시스템 프롬프트 및 반복 지시문 캐싱으로 응답 지연(TTFT) 대폭 단축.
> - `--reasoning-parser qwen3`: Qwen3 계열 씽킹/추론 태그 분리 지원.

### 5.4 vLLM 서비스 시작 및 로그 모니터링
```bash
# 백그라운드로 컨테이너 기동
docker compose up -d

# 기동 및 모델 다운로드/가중치 로딩 로그 확인
docker compose logs -f
```

* **정상 기동 완료 메시지:**  
  로그 마지막에 아래와 같이 Uvicorn 서버 기동 안내가 나타나면 완료입니다:
  ```text
  INFO:     Started server process [1]
  INFO:     Waiting for application startup.
  INFO:     Application startup complete.
  INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
  ```
  *(Ctrl + C를 눌러 로그 화면을 빠져나옵니다)*

---

## 6. API 동작 테스트 및 검증

### 6.1 헬스체크 (Health Check)
```bash
curl -i http://localhost:8000/health
```
* **성공 응답:** `HTTP/1.1 200 OK`

### 6.2 서빙 모델 목록 조회
```bash
curl -s http://localhost:8000/v1/models | jq .
```
* **응답 예시:**
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

### 6.3 Chat Completion 추론 테스트 (Curl)
```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.8:27b",
    "messages": [
      {"role": "system", "content": "You are an AI assistant running on RTX PRO 6000."},
      {"role": "user", "content": "서버 환경 테스트 중입니다. 한국어로 간단하게 환영 메시지와 3줄 요약을 작성해주세요."}
    ],
    "max_tokens": 200,
    "temperature": 0.7
  }' | jq .
```

* **출력 확인:** `choices[0].message.content` 필드에 Qwen3.8 모델의 정답 텍스트가 정상 출력되는지 확인합니다.

### 6.4 스트리밍(SSE) 실시간 테스트
```bash
curl -N http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.8:27b",
    "messages": [{"role": "user", "content": "1부터 10까지 숫자를 한 줄에 하나씩 출력해줘."}],
    "stream": true
  }'
```

---

## 7. 주요 트러블슈팅 및 운영 팁

### 7.1 메인보드(ASUS ProArt X870E) BIOS 설정 점검
* **Above 4G Decoding:** 반드시 **Enabled** 상태여야 대용량 VRAM(48GB) 주소 공간이 정상 매핑됩니다.
* **Re-Size BAR Support:** **Auto** 또는 **Enabled** 권장 (Direct VRAM 액세스 효율화).
* **PCIe Link Speed:** Gen4 또는 Gen5로 정상 링크되었는지 확인 (`lspci -vv -s 01:00.0 | grep -i lnksta`).

### 7.2 디스크 여유 공간 관리
* Qwen3.8-27B AWQ 모델은 약 21GB의 다운로드 용량을 차지합니다.
* Ubuntu Server 설치 시 LVM 볼륨 기본값이 작게(100GB 등) 잡힌 경우가 있으므로 `df -h /`로 확인하십시오.
* 필요 시 LVM 확장:
  ```bash
  sudo lvextend -l +100%FREE /dev/ubuntu-vg/ubuntu-lv
  sudo resize2fs /dev/ubuntu-vg/ubuntu-lv
  ```

### 7.3 외부 통신 및 방화벽 포트 오픈
외부(사내 타 PC 등)에서 해당 vLLM 서버의 `8000` 포트로 접근해야 하는 경우:
```bash
sudo ufw allow 8000/tcp
```
