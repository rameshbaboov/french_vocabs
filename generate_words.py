#!/usr/bin/env python3
import argparse, os, re, time, json, datetime
from typing import List, Set
import requests

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"

# Accept a single French token (letters + accents + apostrophes/dash)
WORD_TOKEN = re.compile(r"^[A-Za-zÀ-ÖØ-öø-ÿœŒçÇ][A-Za-zÀ-ÖØ-öø-ÿœŒçÇ'’-]*$")

def normalize(s: str) -> str:
    return s.strip().replace("’","'")

def extract_words(raw: str) -> List[str]:
    """Keep only single-token French words; allow l'xxx, aujourd'hui, porte-clé; drop everything else."""
    out: List[str] = []
    for ln in (l.strip() for l in raw.splitlines()):
        if not ln: 
            continue
        ln = normalize(ln)
        if ln.lower() == "de la":              # allow this specific article phrase
            out.append(ln); continue
        if WORD_TOKEN.match(ln):
            out.append(ln)
    # de-dup within the call while preserving order
    seen: Set[str] = set(); res: List[str] = []
    for w in out:
        if w not in seen:
            seen.add(w); res.append(w)
    return res

def call_ollama(model: str, prompt: str, system: str, timeout: int = 300) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "options": {
            "temperature": 0.7,
            "top_p": 0.9,
            "repeat_penalty": 1.05,
            "num_predict": 512       # allow many lines per response
        },
        "stream": False
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    if data.get("error"):
        raise RuntimeError(f"Ollama error: {data['error']}")
    resp = data.get("response","")
    if not resp.strip():
        raise RuntimeError(f"Ollama returned empty response. Payload preview: {json.dumps(data)[:400]}")
    return resp

def build_prompts(level: str, want: int, blacklist_tail: List[str]) -> tuple[str,str]:
    system = (
        "Tu renvoies UNIQUEMENT des mots français isolés, un par ligne.\n"
        "- Autorisé: articles (le, la, l', un, une, des, du, de, «de la»).\n"
        "- Interdit: numérotation, traduction, phrases, explications.\n"
        "- Pas de doublons."
    )
    blacklist = ", ".join(blacklist_tail[-200:]) if blacklist_tail else "—"
    prompt = (
        f"Donne exactement {want} mots français de niveau {level}.\n"
        f"Ne répète AUCUN des mots déjà produits: {blacklist}\n"
        "Un mot par ligne. Rien d'autre."
    )
    return system, prompt

def save_batch(words: List[str], level: str, outdir: str) -> str:
    os.makedirs(outdir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    fname = f"french_{level}_{ts}.txt"
    path = os.path.join(outdir, fname)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(words) + "\n")
    return path

def main():
    ap = argparse.ArgumentParser(description="Generate French words forever, saving timestamped batches.")
    ap.add_argument("--model", default="llama3.1:8b", help="Ollama model (e.g., llama3.1, mistral, gemma2:9b)")
    ap.add_argument("--level", choices=["A1","A2","B1","B2"], default="A1", help="CEFR level")
    ap.add_argument("--batch-size", type=int, default=50, help="Words per output file")
    ap.add_argument("--outdir", default="out_french", help="Output folder")
    ap.add_argument("--timeout", type=int, default=300, help="HTTP timeout seconds")
    ap.add_argument("--multiplier", type=float, default=2.0, help="Ask ~multiplier×batch-size per API call")
    ap.add_argument("--sleep-between-batches", type=float, default=1.5, help="Pause after saving each batch (seconds)")
    ap.add_argument("--max-calls-per-batch", type=int, default=200, help="Safety cap to fill one batch")
    args = ap.parse_args()

    batch_size = args.batch_size
    want_per_call = max(int(batch_size * args.multiplier), batch_size + 20)
    produced_global: List[str] = []   # track tail to reduce repeats across calls

    print(f"▶ Infinite mode started | model={args.model} level={args.level} batch={batch_size} ask/call={want_per_call}")
    print("   Press Ctrl+C to stop.\n")

    try:
        batch_counter = 0
        while True:
            batch_counter += 1
            current_batch: List[str] = []
            calls = 0
            t0 = time.time()
            while len(current_batch) < batch_size and calls < args.max_calls_per_batch:
                calls += 1
                system, prompt = build_prompts(args.level, want_per_call, produced_global)
                try:
                    resp = call_ollama(args.model, prompt, system, timeout=args.timeout)
                except Exception as e:
                    print(f"⚠️  Call #{calls} failed: {e}; retrying in 2s…")
                    time.sleep(2.0)
                    continue

                got = extract_words(resp)
                added = 0
                for w in got:
                    if w not in current_batch:     # keep uniqueness within the batch
                        current_batch.append(w)
                        produced_global.append(w)
                        added += 1
                        if len(current_batch) >= batch_size:
                            break

                print(f"✅ Call #{calls}: got={len(got)} added={added} | batch={len(current_batch)}/{batch_size}")

                if added == 0:
                    time.sleep(0.8)   # nudge randomness if it’s stuck

            if current_batch:
                out_path = save_batch(current_batch, args.level, args.outdir)
                elapsed = time.time() - t0
                print(f"💾 Saved {out_path} | +{len(current_batch)} words | elapsed={elapsed:.1f}s | batch #{batch_counter}")
            else:
                print("⚠️ Unable to collect any words for this batch; backing off 5s…")
                time.sleep(5)

            time.sleep(args.sleep_between_batches)

    except KeyboardInterrupt:
        print("\n👋 Stopped. Goodbye!")

if __name__ == "__main__":
    main()
