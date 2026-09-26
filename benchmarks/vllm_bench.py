#!/usr/bin/env python3
"""
vLLM 서빙 성능 측정 - 프리필/디코드 분리 계측

사용법:
    pip install httpx
    python vllm_bench.py                          # 기본 (localhost)
    python vllm_bench.py --url http://10.0.0.2:8000
    python vllm_bench.py --input-len 8192 --concurrency 16

측정 항목:
    TTFT      : 첫 토큰까지 시간 = 프리필 소요
    프리필     : input_len / TTFT  (tok/s)
    디코드     : (output-1) / (총시간 - TTFT)  (tok/s, 요청당)
    총 처리량   : 전체 출력 토큰 / 전체 벽시계 시간
"""

import argparse
import asyncio
import json
import os
import statistics
import sys
import time

try:
    import httpx
except ImportError:
    sys.exit("httpx 필요: pip install httpx")


def pct(xs, p):
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def make_prompt(n_tokens):
    """대략 n_tokens 크기의 한국어 프롬프트. 캐시 방지를 위해 호출마다 다르게."""
    seed = f"[{time.time_ns()}] "
    # 한국어는 대략 토큰당 1.5~2자. 보수적으로 1.6자로 계산
    body = "뉴스 기사 본문 데이터 분석 대상 문서 내용 " * max(1, int(n_tokens * 1.6 / 22))
    return seed + body


async def one_request(client, url, model, prompt, out_len, api_key, priority):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": out_len,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if priority is not None:
        payload["priority"] = priority

    t0 = time.perf_counter()
    ttft = None
    n_chunks = 0
    usage = None

    try:
        async with client.stream("POST", f"{url}/v1/chat/completions",
                                 json=payload, headers=headers) as r:
            if r.status_code != 200:
                body = await r.aread()
                return {"error": f"HTTP {r.status_code}: {body[:200].decode('utf-8', 'replace')}"}
            async for line in r.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data.strip() == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if obj.get("usage"):
                    usage = obj["usage"]
                choices = obj.get("choices") or []
                if choices and choices[0].get("delta", {}).get("content"):
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    n_chunks += 1
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}

    total = time.perf_counter() - t0
    if ttft is None:
        return {"error": "출력 토큰 없음"}

    n_out = usage.get("completion_tokens", n_chunks) if usage else n_chunks
    n_in = usage.get("prompt_tokens") if usage else None

    return {
        "ttft": ttft,
        "total": total,
        "n_out": n_out,
        "n_in": n_in,
        "decode_tps": (n_out - 1) / (total - ttft) if n_out > 1 and total > ttft else 0.0,
    }


async def run_phase(url, model, n_req, conc, in_len, out_len, api_key, priority, label):
    print(f"\n{'=' * 62}")
    print(f"  {label}")
    print(f"  입력 ~{in_len} / 출력 {out_len} / 동시 {conc} / 요청 {n_req}건")
    print(f"{'=' * 62}")

    sem = asyncio.Semaphore(conc)
    limits = httpx.Limits(max_connections=conc + 4, max_keepalive_connections=conc + 4)

    async with httpx.AsyncClient(timeout=httpx.Timeout(900.0), limits=limits) as client:
        async def worker(i):
            async with sem:
                return await one_request(client, url, model, make_prompt(in_len),
                                         out_len, api_key, priority)

        t0 = time.perf_counter()
        results = await asyncio.gather(*[worker(i) for i in range(n_req)])
        wall = time.perf_counter() - t0

    ok = [r for r in results if "error" not in r]
    bad = [r for r in results if "error" in r]

    if bad:
        print(f"  실패 {len(bad)}건 - 예: {bad[0]['error']}")
    if not ok:
        print("  측정 실패")
        return None

    ttfts = [r["ttft"] for r in ok]
    decodes = [r["decode_tps"] for r in ok]
    total_out = sum(r["n_out"] for r in ok)
    real_in = ok[0].get("n_in")

    prefill_tps = (real_in / statistics.median(ttfts)) if real_in else None

    print(f"\n  [프리필]")
    if real_in:
        print(f"    실제 입력 토큰      {real_in}")
    print(f"    TTFT   p50 {pct(ttfts,50)*1000:8.1f} ms   p95 {pct(ttfts,95)*1000:8.1f} ms")
    if prefill_tps:
        print(f"    프리필 속도         {prefill_tps:8.0f} tok/s  (동시 {conc} 합산)")

    print(f"\n  [디코드]")
    print(f"    요청당  p50 {pct(decodes,50):7.1f} tok/s   p95 {pct(decodes,95):7.1f} tok/s")

    print(f"\n  [전체]")
    print(f"    출력 처리량         {total_out/wall:8.1f} tok/s")
    print(f"    요청 처리량         {len(ok)/wall:8.2f} req/s")
    print(f"    소요                {wall:8.1f} s   ({len(ok)}건)")

    return {
        "label": label, "conc": conc,
        "ttft_p50_ms": pct(ttfts, 50) * 1000,
        "decode_p50": pct(decodes, 50),
        "output_tps": total_out / wall,
        "req_per_s": len(ok) / wall,
    }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--model", default=None, help="미지정 시 /v1/models에서 자동 조회")
    ap.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY"))
    ap.add_argument("--input-len", type=int, default=8192)
    ap.add_argument("--output-len", type=int, default=300)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--num-prompts", type=int, default=None)
    ap.add_argument("--priority", type=int, default=None,
                    help="scheduling-policy priority 사용 시")
    ap.add_argument("--quick", action="store_true", help="단발 지연만 측정")
    args = ap.parse_args()

    url = args.url.rstrip("/")
    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}

    model = args.model
    if not model:
        async with httpx.AsyncClient(timeout=30.0) as c:
            r = await c.get(f"{url}/v1/models", headers=headers)
            r.raise_for_status()
            info = r.json()["data"][0]
            model = info["id"]
            print(f"모델          {model}")
            print(f"max_model_len {info.get('max_model_len')}")

    in_len = min(args.input_len, 7000)  # 출력 여유 확보

    rows = []
    if args.quick:
        rows.append(await run_phase(url, model, 3, 1, in_len, args.output_len,
                                    args.api_key, args.priority, "단발 지연"))
    else:
        # 1) 단발 - 지연 상한 확인
        rows.append(await run_phase(url, model, 3, 1, in_len, args.output_len,
                                    args.api_key, args.priority, "1/3  단발 (동시성 1)"))
        # 2) 목표 동시성 - 실제 운영 형상
        n = args.num_prompts or args.concurrency * 6
        rows.append(await run_phase(url, model, n, args.concurrency, in_len,
                                    args.output_len, args.api_key, args.priority,
                                    f"2/3  배치 (동시성 {args.concurrency})"))
        # 3) 포화 지점 탐색
        hi = args.concurrency * 2
        rows.append(await run_phase(url, model, hi * 4, hi, in_len, args.output_len,
                                    args.api_key, args.priority,
                                    f"3/3  포화 확인 (동시성 {hi})"))

    rows = [r for r in rows if r]
    if len(rows) > 1:
        print(f"\n{'=' * 62}")
        print("  요약")
        print(f"{'=' * 62}")
        print(f"  {'구간':<26} {'출력 tok/s':>12} {'req/s':>9} {'TTFT ms':>10}")
        for r in rows:
            print(f"  {r['label']:<26} {r['output_tps']:>12.1f} "
                  f"{r['req_per_s']:>9.2f} {r['ttft_p50_ms']:>10.0f}")

        base, peak = rows[1], rows[-1]
        if peak["output_tps"] > base["output_tps"] * 1.10:
            print(f"\n  → 동시성 {peak['conc']}에서 처리량 "
                  f"{peak['output_tps']/base['output_tps']:.2f}배. "
                  f"--max-num-seqs 상향 여지 있음")
        else:
            print(f"\n  → 동시성 {base['conc']} 부근에서 포화. 현재 설정이 적정")


if __name__ == "__main__":
    asyncio.run(main())
