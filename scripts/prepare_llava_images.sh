#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(realpath "$(dirname "$0")/..")}"
BASE="$ROOT_DIR/data/imgconv_data/llava"
IMG="$BASE/llava_images"

mkdir -p "$BASE" "$IMG"
ln -sfn "$ROOT_DIR/data/genseg_data/coco2017" "$IMG/coco"

download_zip() {
  local out_file="$1"
  local url="$2"
  local check_zip="${3:-1}"

  mkdir -p "$(dirname "$out_file")"
  while true; do
    echo "[DOWNLOAD] $(basename "$out_file")"
    wget -c --tries=0 --timeout=30 --waitretry=5 -O "$out_file" "$url"

    if [[ "$check_zip" == "0" ]]; then
      break
    fi

    if unzip -tq "$out_file" >/dev/null 2>&1; then
      break
    fi

    echo "[WARN] zip invalid: $out_file, redownload from scratch."
    local backup="${out_file}.bad.$(date +%s)"
    mv "$out_file" "$backup"
  done
}

unzip_once() {
  local zip_file="$1"
  local target_dir="$2"

  mkdir -p "$target_dir"
  echo "[UNZIP] $(basename "$zip_file") -> $target_dir"
  unzip -oq "$zip_file" -d "$target_dir"
}

count_files() {
  local dir="$1"
  if [[ -d "$dir" ]]; then
    find "$dir" -maxdepth 1 -type f | wc -l | tr -d " "
  else
    echo "MISSING"
  fi
}

# 1) VG
if [[ ! -d "$IMG/vg/VG_100K" ]] || [[ ! -d "$IMG/vg/VG_100K_2" ]]; then
  download_zip "$BASE/vg_images.zip" "https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip"
  download_zip "$BASE/vg_images2.zip" "https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip"
  unzip_once "$BASE/vg_images.zip" "$IMG/vg"
  unzip_once "$BASE/vg_images2.zip" "$IMG/vg"
fi
truncate -s 0 "$BASE/vg_images.zip" 2>/dev/null || true
truncate -s 0 "$BASE/vg_images2.zip" 2>/dev/null || true

# 2) TextVQA
if [[ ! -d "$IMG/textvqa/train_images" ]]; then
  download_zip "$BASE/textvqa_train_val_images.zip" "https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip"
  unzip_once "$BASE/textvqa_train_val_images.zip" "$IMG/textvqa"
fi
ln -sfn textvqa "$IMG/text_vqa"
truncate -s 0 "$BASE/textvqa_train_val_images.zip" 2>/dev/null || true

# 3) OCR-VQA
if [[ ! -d "$IMG/ocr_vqa/images" ]]; then
  download_zip "$BASE/ocr_vqa_images_llava_v15.zip" "https://huggingface.co/datasets/weizhiwang/llava_v15_instruction_images/resolve/main/ocr_vqa_images_llava_v15.zip"
  unzip_once "$BASE/ocr_vqa_images_llava_v15.zip" "$IMG/ocr_vqa"

  if [[ ! -d "$IMG/ocr_vqa/images" ]]; then
    CAND="$(find "$IMG/ocr_vqa" -maxdepth 6 -type d -name images | head -n 1 || true)"
    if [[ -n "${CAND:-}" ]]; then
      mkdir -p "$IMG/ocr_vqa/images"
      if [[ "$CAND" != "$IMG/ocr_vqa/images" ]]; then
        rsync -a --ignore-existing "$CAND/" "$IMG/ocr_vqa/images/"
      fi
    fi
  fi
fi
truncate -s 0 "$BASE/ocr_vqa_images_llava_v15.zip" 2>/dev/null || true

# 4) GQA
if [[ ! -d "$IMG/gqa/images" ]]; then
  download_zip "$BASE/gqa_images.zip" "https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip"
  unzip_once "$BASE/gqa_images.zip" "$IMG/gqa"
fi
truncate -s 0 "$BASE/gqa_images.zip" 2>/dev/null || true

echo "[CHECK]"
for d in \
  "$IMG/coco/train2017" \
  "$IMG/gqa/images" \
  "$IMG/ocr_vqa/images" \
  "$IMG/textvqa/train_images" \
  "$IMG/vg/VG_100K" \
  "$IMG/vg/VG_100K_2"; do
  echo "$d -> $(count_files "$d") files"
done

echo "[DONE] LLaVA images prepared."
