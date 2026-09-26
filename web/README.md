# LLM Streaming Web Client (Demo Application)

vLLM 서버(`http://gpu2:8000/v1` 또는 로컬 vLLM 인스턴스)와 연동하여 실시간 스트리밍 답변, 사고 과정(Reasoning) 시각화, 멀티모달 비전(Vision) 이미지 분석을 제공하는 경량 웹 클라이언트입니다.

---

## 주요 기능

1. **자동 모델 탐색 (Model Discovery)**:
   - `/v1/models` 엔드포인트를 실시간 조회하여 배포된 모델(`unsloth/Qwen3.8-27B-NVFP4`, `qwen3.8:27b` 등)을 자동 감지하고 선택합니다.

2. **실시간 토큰 스트리밍 (SSE)**:
   - vLLM의 Server-Sent Events(SSE) 스트리밍을 브라우저로 실시간 중계하여 지연 없이 즉각적인 타이핑 효과를 제공합니다.

3. **비전(Vision) 멀티모달 이미지 분석**:
   - 이미지 첨부 버튼, 클립보드 붙여넣기(`Cmd+V` / `Ctrl+V`), 드래그 앤 드롭으로 이미지(PNG, JPG, WEBP, GIF)를 업로드하여 실시간 시각 분석 및 묘사 결과를 스트리밍합니다.
   - 업로드 이미지 썸네일 미리보기 및 라이트박스 확대 뷰어를 지원합니다.

4. **빠른 답변 (No-Think) vs 심층 사고 (Thinking) 모드 전환**:
   - 상단 모드 스위처를 통해 불필요한 사고 토큰 없이 즉답을 생성하는 빠른 답변(No-Think) 모드와 단계별 논리 추론을 거치는 심층 사고(Thinking) 모드를 전환할 수 있습니다.

5. **사고 과정(Reasoning) 시각화**:
   - DeepSeek 및 Qwen 3.8 Reasoning 모델의 `reasoning` 사고 토큰을 별도의 아코디언 컴포넌트(사고 과정)에 분리하여 실시간 표시합니다.

6. **마크다운 & 코드 하이라이트 & 1클릭 복사**:
   - 코드 블록 구문 강조, 표, 목록, 인라인 코드 렌더링 및 원클릭 복사 기능을 제공합니다.

7. **실시간 추론 속도 및 메트릭 표시**:
   - 초당 생성 토큰 수(Tokens/sec), 총 생성 토큰 수, 소요 시간(초)을 실시간으로 계산하여 표시합니다.

8. **Zero Dependencies (외부 패키지 설치 불필요)**:
   - Node.js 내장 모듈 및 Python 표준 라이브러리만을 사용하여 별도의 `npm install`이나 `pip install` 없이 즉시 실행됩니다.

---

## 실행 방법

### 방법 1: Node.js로 실행 (권장)
```bash
cd web

# 기본 포트 3000으로 실행
npm start
# 또는
node server.js
```

### 방법 2: Python 3으로 실행
```bash
cd web

python3 server.py
# 또는
npm run start:py
```

브라우저에서 **`http://localhost:3000`** 으로 접속합니다.

---

## 환경 변수 설정

다른 호스트 주소나 포트를 사용하려면 환경 변수를 지정할 수 있습니다:

```bash
# 기본값: PORT=3000, LLM_BASE_URL=http://gpu2:8000/v1
PORT=8080 LLM_BASE_URL=http://localhost:8000/v1 node server.js
```

또는 Python 실행 시:
```bash
PORT=8080 LLM_BASE_URL=http://localhost:8000/v1 python3 server.py
```

---

## 디렉토리 구조

```
web/
├── server.js          # Node.js 기반 스트리밍 프록시 및 정적 파일 서버
├── server.py          # Python 표준 라이브러리 기반 프록시 서버
├── package.json       # 실행 스크립트 정의
├── README.md          # 웹 클라이언트 가이드 (본 문서)
└── public/
    ├── index.html     # 반응형 웹 UI 레이아웃
    ├── style.css      # 다크/라이트 테마, 아코디언, 코드 블록 스타일
    └── app.js         # SSE 스트리밍 클라이언트, 마크다운 렌더러, 상태 관리
```
