#!/usr/bin/env python3
"""
Benchmark for fetch_news headline deduplication: List vs Set lookup.
"""

import time
import random
import string

def generate_mock_items(num_items: int = 100, headline_len: int = 50, duplicate_ratio: float = 0.3):
    unique_headlines = [
        ''.join(random.choices(string.ascii_letters + string.digits + ' ', k=headline_len))
        for _ in range(int(num_items * (1 - duplicate_ratio)))
    ]
    items = []
    for _ in range(num_items):
        h = random.choice(unique_headlines)
        items.append({"headline": h})
    return items

def headlines_dedup_list(items, max_headlines=3, max_check=9):
    heads = []
    for it in items[:max_check]:
        h = str(it.get("headline", "")).strip()
        if h and h not in heads:
            heads.append(h[:140])
        if len(heads) >= max_headlines:
            break
    return heads

def headlines_dedup_set(items, max_headlines=3, max_check=9):
    heads = []
    seen = set()
    for it in items[:max_check]:
        h = str(it.get("headline", "")).strip()
        if h and h not in seen:
            seen.add(h)
            heads.append(h[:140])
        if len(heads) >= max_headlines:
            break
    return heads

def headlines_dedup_set_truncated(items, max_headlines=3, max_check=9):
    heads = []
    seen = set()
    for it in items[:max_check]:
        raw_h = str(it.get("headline", "")).strip()
        if raw_h:
            h = raw_h[:140]
            if h not in seen:
                seen.add(h)
                heads.append(h)
                if len(heads) >= max_headlines:
                    break
    return heads

def run_benchmark(iterations: int = 500_000):
    small_items = generate_mock_items(10, headline_len=80, duplicate_ratio=0.3)
    large_items = generate_mock_items(200, headline_len=80, duplicate_ratio=0.3)

    print(f"=== Running Micro-Benchmark ({iterations:,} iterations) ===")

    # Small scenario
    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = headlines_dedup_list(small_items, 3, 9)
    t1 = time.perf_counter()
    list_small_time = t1 - t0

    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = headlines_dedup_set(small_items, 3, 9)
    t1 = time.perf_counter()
    set_small_time = t1 - t0

    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = headlines_dedup_set_truncated(small_items, 3, 9)
    t1 = time.perf_counter()
    set_trunc_small_time = t1 - t0

    print(f"\nSmall Slice (check top 9 items, max 3 headlines):")
    print(f"  List lookup time:     {list_small_time:.4f}s")
    print(f"  Set lookup time:      {set_small_time:.4f}s")
    print(f"  Set (trunc) lookup:   {set_trunc_small_time:.4f}s")

    # Large scenario
    large_iters = iterations // 5
    t0 = time.perf_counter()
    for _ in range(large_iters):
        _ = headlines_dedup_list(large_items, 50, 100)
    t1 = time.perf_counter()
    list_large_time = t1 - t0

    t0 = time.perf_counter()
    for _ in range(large_iters):
        _ = headlines_dedup_set(large_items, 50, 100)
    t1 = time.perf_counter()
    set_large_time = t1 - t0

    t0 = time.perf_counter()
    for _ in range(large_iters):
        _ = headlines_dedup_set_truncated(large_items, 50, 100)
    t1 = time.perf_counter()
    set_trunc_large_time = t1 - t0

    print(f"\nLarger Slice (check top 100 items, max 50 headlines, {large_iters:,} iterations):")
    print(f"  List lookup time:     {list_large_time:.4f}s")
    print(f"  Set lookup time:      {set_large_time:.4f}s")
    print(f"  Set (trunc) lookup:   {set_trunc_large_time:.4f}s")

if __name__ == "__main__":
    run_benchmark()
