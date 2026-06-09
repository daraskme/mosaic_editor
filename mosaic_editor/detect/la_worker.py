"""LocateAnything-3B 推論ワーカー (サブプロセスとして実行).

このスクリプトは transformers==4.57.1 (vendor/transformers_la を PYTHONPATH
先頭に挿入) で動かすことを前提とする。LocateAnything-3B の remote code が
transformers v5 と非互換のため、メインプロセス (SAM3 用に v5 が必要) から
分離して実行する。

プロトコル: stdin/stdout の JSON Lines。
  起動完了:  {"ready": true}
  要求:      {"image": <png path>, "prompt": "cat1</c>cat2",
              "generation_mode": "hybrid", "max_new_tokens": 8192}
  応答:      {"ok": true, "raw": "<model output text>"}
            {"ok": false, "error": "..."}
  終了要求:  {"exit": true}

注意: ログ・進捗の類は stderr のみに出すこと (stdout はプロトコル専用)。
"""
import json
import sys


def log(msg: str) -> None:
    print(f"[la_worker] {msg}", file=sys.stderr, flush=True)


def main() -> None:
    import torch
    from transformers import AutoModel, AutoProcessor, AutoTokenizer
    from PIL import Image

    model_id = "nvidia/LocateAnything-3B"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    import transformers
    log(f"transformers={transformers.__version__} device={device} dtype={dtype}")
    log("loading tokenizer/processor...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    log("loading model...")
    model = AutoModel.from_pretrained(
        model_id, torch_dtype=dtype, trust_remote_code=True,
    ).to(device).eval()
    log("model loaded")

    print(json.dumps({"ready": True}), flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            print(json.dumps({"ok": False, "error": f"bad request: {e}"}),
                  flush=True)
            continue
        if req.get("exit"):
            break
        try:
            image = Image.open(req["image"]).convert("RGB")
            prompt = ("Locate all the instances that matches the following "
                      f"description: {req['prompt']}.")
            messages = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ]}]
            with torch.no_grad():
                text = processor.py_apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
                images, videos = processor.process_vision_info(messages)
                inputs = processor(
                    text=[text], images=images, videos=videos,
                    return_tensors="pt").to(device)
                response = model.generate(
                    pixel_values=inputs["pixel_values"].to(dtype),
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    image_grid_hws=inputs.get("image_grid_hws", None),
                    tokenizer=tokenizer,
                    max_new_tokens=int(req.get("max_new_tokens", 8192)),
                    use_cache=True,
                    generation_mode=req.get("generation_mode", "hybrid"),
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    repetition_penalty=1.1,
                )
            raw = response[0] if isinstance(response, (tuple, list)) else response
            print(json.dumps({"ok": True, "raw": str(raw)}), flush=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(json.dumps({"ok": False, "error": str(e)}), flush=True)


if __name__ == "__main__":
    main()
