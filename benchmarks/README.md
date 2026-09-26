# 🚀 vLLM Serving Benchmark Tool (`vllm_bench.py`)

vLLM 추론 서버의 **프리필(Prefill) 처리량, 첫 토큰 응답 지연(TTFT), 스트리밍 디코드(Decode) 속도, 동시성 포화 지점**을 실시간 SSE 스트림 기반으로 분리 계측하는 벤치마크 도구입니다.

---

## 📋 측정 지표 (Metrics)

| 지표 | 정의 | 단위 | 측정 의미 |
| :--- | :--- | :---: | :--- |
| **TTFT (Time to First Token)** | 요청 시작부터 첫 토큰 반환까지 소요된 시간 | `ms` | 프리필(Prefill) 단계 지연 체감 시간 |
| **프리필 속도 (Prefill TPS)** | `input_len / TTFT` | `tok/s` | 프롬프트 입력 컨텍스트 처리 속도 |
| **디코드 속도 (Decode TPS)** | `(output_tokens - 1) / (총시간 - TTFT)` | `tok/s` | 단일 요청당 실시간 생성 타이핑 속도 |
| **총 출력 처리량 (Throughput)** | `전체 생성 토큰 수 / 전체 벽시계 시간` | `tok/s` | 서버 전체 동시 생성 처리량 |
| **요청 처리량 (RPS)** | `완료된 요청 수 / 전체 벽시계 시간` | `req/s` | 초당 처리 가능한 요청 건수 |

---

## ⚙️ 사전 요구사항 및 설치

Python 3.9+ 및 비동기 HTTP 라이브러리 `httpx`가 필요합니다:

```bash
pip install -r benchmarks/requirements.txt
# 또는
pip install httpx
```

---

## 🏃 실행 방법 (Usage)

### 1. 기본 실행 (Localhost vLLM 서버 자동 모델 탐색)
```bash
python benchmarks/vllm_bench.py
```
> 모델명을 별도로 지정하지 않으면 `/v1/models` 엔드포인트를 조회하여 현재 서빙 중인 첫 번째 모델을 자동으로 타겟팅합니다.

### 2. 원격 서버 타겟팅
```bash
python benchmarks/vllm_bench.py --url http://10.0.0.2:8000
# 또는
python benchmarks/vllm_bench.py --url http://gpu2:8000
```

### 3. 단발 지연만 빠른 점검 (`--quick`)
서버가 정상적으로 최소 지연으로 응답하는지 3회 단발 요청으로 빠르게 확인합니다:
```bash
python benchmarks/vllm_bench.py --quick
```

### 4. 8K 장문 컨텍스트 & 고동시성 부하 테스트
```bash
python benchmarks/vllm_bench.py \
  --url http://localhost:8000 \
  --input-len 8192 \
  --output-len 300 \
  --concurrency 16
```

---

## 🔍 3단계 벤치마크 단계 (Execution Phases)

`vllm_bench.py`는 단일 실행 시 다음 3단계로 부하를 점진적으로 가하며 분석합니다:

1. **Phase 1 (1/3 단발, 동시성 1)**:
   - 다른 부하 간섭 없는 순수 하드웨어 단일 스트림 디코드 속도(약 80+ tok/s)와 최저 TTFT 측정.
2. **Phase 2 (2/3 목표 동시성 배치, 예: 동시성 16)**:
   - 프로덕션 운영 형상에서의 실제 배치 처리량 및 지연 시간 측정.
3. **Phase 3 (3/3 포화 지점 탐색, 동시성 2배, 예: 동시성 32)**:
   - GPU 컴퓨팅 및 KV Cache 대역폭 한계 도달 여부 확인.

---

## 📊 결과 요약 및 튜닝 피드백 가이드

실행이 완료되면 아래와 같이 포화 분석 요약이 출력됩니다:

```text
==============================================================
  요약
==============================================================
  구간                          출력 tok/s     req/s    TTFT ms
  1/3  단발 (동시성 1)                81.3      0.27      112
  2/3  배치 (동시성 16)              820.5      2.74      340
  3/3  포화 확인 (동시성 32)         1240.2      4.13      780

  → 동시성 32에서 처리량 1.51배. --max-num-seqs 상향 여지 있음
```

* **처리량이 10% 이상 증가하는 경우:**
  - GPU VRAM 대역폭에 아직 여유가 있으므로 `--max-num-seqs` 값을 추가 상향하여 전체 Throughput을 극대화할 수 있습니다.
* **처리량 증가가 미미하거나 TTFT가 급증하는 경우:**
  - 메모리 대역폭 또는 KV Cache 경합 상태이므로 현재 동시성 수준을 적정 상한으로 유지하십시오.

---

## 🛠️ CLI 옵션 목록

| 옵션 | 기본값 | 설명 |
| :--- | :---: | :--- |
| `--url` | `http://localhost:8000` | vLLM 서버의 기본 URL |
| `--model` | `None` (자동 조회) | 타겟 모델 ID (생략 시 `/v1/models` 자동 감지) |
| `--api-key` | `os.environ[VLLM_API_KEY]` | vLLM API 인증 키 (필요 시) |
| `--input-len` | `8192` | 입력 프롬프트 목표 토큰 길이 (최대 7000으로 자동 클램프) |
| `--output-len` | `300` | 생성 요청 토큰 수 |
| `--concurrency` | `16` | 목표 동시 요청 수 |
| `--num-prompts` | `concurrency * 6` | 총 발행할 프롬프트 수 |
| `--priority` | `None` | vLLM 스케줄링 정책 우선순위 값 |
| `--quick` | `False` | 단발 지연만 3회 측정 후 즉시 종료 |
