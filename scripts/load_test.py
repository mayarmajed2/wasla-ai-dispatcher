"""Real Concurrent Load Test & Live Security Verification against Running Wasla Server."""

from __future__ import annotations

import asyncio
import statistics
import time

import httpx

BASE = "http://127.0.0.1:8000"


async def run_load_test(concurrency: int = 20, total_requests: int = 100) -> dict:
    sem = asyncio.Semaphore(concurrency)
    latencies_ms: list[float] = []
    status_counts: dict[int, int] = {}

    async with httpx.AsyncClient(timeout=15.0) as client:
        async def _one(i: int) -> None:
            async with sem:
                endpoint = ["/api/health", "/api/ready", "/api/menu", "/api/metrics"][i % 4]
                t0 = time.perf_counter()
                r = await client.get(f"{BASE}{endpoint}")
                dt = (time.perf_counter() - t0) * 1000.0
                latencies_ms.append(dt)
                status_counts[r.status_code] = status_counts.get(r.status_code, 0) + 1

        started = time.perf_counter()
        await asyncio.gather(*[_one(i) for i in range(total_requests)])
        total_sec = time.perf_counter() - started

    latencies_ms.sort()
    p50 = statistics.median(latencies_ms)
    p95 = latencies_ms[int(len(latencies_ms) * 0.95) - 1]
    p99 = latencies_ms[int(len(latencies_ms) * 0.99) - 1]

    # Live security probes
    async with httpx.AsyncClient(timeout=10.0) as client:
        r_ready = await client.get(f"{BASE}/api/ready")
        sec_headers = {
            "X-Content-Type-Options": r_ready.headers.get("X-Content-Type-Options"),
            "X-Frame-Options": r_ready.headers.get("X-Frame-Options"),
            "Referrer-Policy": r_ready.headers.get("Referrer-Policy"),
        }
        r_bad_wh = await client.get(
            f"{BASE}/api/webhook",
            params={"hub.mode": "subscribe", "hub.verify_token": "WRONG_TOKEN", "hub.challenge": "123"},
        )
        r_sqli = await client.post(
            f"{BASE}/api/ai/chat",
            json={"phone_number": "01000009999", "message": "'; DROP TABLE orders; --"},
        )

    return {
        "total_requests": total_requests,
        "concurrency": concurrency,
        "total_seconds": round(total_sec, 3),
        "throughput_rps": round(total_requests / total_sec, 2),
        "status_counts": status_counts,
        "latency_ms": {
            "min": round(min(latencies_ms), 2),
            "p50": round(p50, 2),
            "p95": round(p95, 2),
            "p99": round(p99, 2),
            "max": round(max(latencies_ms), 2),
        },
        "security_checks": {
            "headers": sec_headers,
            "invalid_webhook_token_status": r_bad_wh.status_code,
            "sqli_probe_handled_safely_status": r_sqli.status_code,
        },
    }


if __name__ == "__main__":
    import json

    print(json.dumps(asyncio.run(run_load_test()), indent=2))
