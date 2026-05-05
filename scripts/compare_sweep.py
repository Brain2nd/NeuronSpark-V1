"""按 (prompt, config) 对比 nocache vs cached 输出。"""
import re, sys


def parse(path):
    """return list of (prompt, config_label, output)"""
    text = open(path).read()
    blocks = re.split(r"={70,}", text)
    out = []
    cur_prompt = None
    for b in blocks:
        m = re.search(r"\[([^\]]+)\]\s+'([^']+)'", b)
        if m:
            cur_prompt = (m.group(1), m.group(2))
            continue
        # config block
        for cfg_match in re.finditer(
            r"--\s+([^(]+?)\s*\([^)]+\)\s*\[[\d.]+s\]\s*--\s*\n(.*?)(?=\n\s*--|\Z)",
            b, re.DOTALL,
        ):
            cfg = cfg_match.group(1).strip()
            content = cfg_match.group(2).strip()
            if cur_prompt is not None:
                out.append((cur_prompt, cfg, content))
    return out


def main():
    nocache = {(p, c): t for (p, c, t) in parse("/tmp/nocache_sweep_full.log")}
    cached = {(p, c): t for (p, c, t) in parse("/tmp/cached_sweep_full.log")}

    print(f"nocache entries: {len(nocache)}, cached entries: {len(cached)}")
    common = set(nocache) & set(cached)
    print(f"common keys: {len(common)}")

    exact = []
    diff_after = []
    for k in sorted(common, key=lambda x: (x[0][0], x[1])):
        a = nocache[k]
        b = cached[k]
        if a == b:
            exact.append(k)
        else:
            # find first divergent char
            i = 0
            while i < min(len(a), len(b)) and a[i] == b[i]:
                i += 1
            diff_after.append((k, i, len(a), len(b), a, b))

    print(f"\nexact match: {len(exact)}/{len(common)}")
    for (prompt, cfg) in exact:
        print(f"  ✓ [{prompt[0]}] {cfg}")

    print(f"\ndiverging: {len(diff_after)}")
    for k, i, la, lb, a, b in diff_after:
        prompt, cfg = k
        print(f"\n  ✗ [{prompt[0]}] {cfg}  (diverge at char {i}/{min(la,lb)})")
        print(f"    common ({i} chars): {a[:i][-50:]!r}")
        print(f"    nocache: {a[i:i+80]!r}")
        print(f"    cached:  {b[i:i+80]!r}")


if __name__ == "__main__":
    main()
