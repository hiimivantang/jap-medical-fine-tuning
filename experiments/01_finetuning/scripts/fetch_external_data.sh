#!/usr/bin/env bash
# Download the licence-verified third-party datasets into data/raw/<source>/ and record SHA-256 checksums.
#
# Every source is pinned (git commit or fixed download id). Licences were verified on the primary source
# pages on 2026-10-19 (see docs/DATASETS.md). Excluded on licence grounds and therefore NOT downloaded:
# JMED-LLM NRNER/JCSTS (CC BY-NC-SA), JMED-LLM SMDIS (NII MedWeb terms forbid profit use).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
RAW="$REPO_ROOT/data/raw"
mkdir -p "$RAW"

JMED_LLM_COMMIT="3774a34e4a26a7b8a7851f421ecad91a04c2f59b"
JGLUE_COMMIT="6f071c09316baae89c3d083a90985b4b1cb9968c"

fetch() {  # fetch <url> <dest>
  mkdir -p "$(dirname "$2")"
  curl -fsSL --retry 3 -o "$2" "$1"
}

# JMED-LLM (sociocom) — commercial-use-compatible subsets only.
for f in mrner_disease mrner_medicine crade rrtnm jmmlu_med; do
  fetch "https://raw.githubusercontent.com/sociocom/JMED-LLM/${JMED_LLM_COMMIT}/datasets/${f}.csv" \
        "$RAW/jmed_llm/${f}.csv"
done

# JGLUE v1.3 test splits (general Japanese regression; evaluation only, CC BY-SA 4.0).
fetch "https://raw.githubusercontent.com/yahoojapan/JGLUE/${JGLUE_COMMIT}/datasets/jcommonsenseqa-v1.3/test-v1.3.json" \
      "$RAW/jglue/jcommonsenseqa-test-v1.3.json"
fetch "https://raw.githubusercontent.com/yahoojapan/JGLUE/${JGLUE_COMMIT}/datasets/jnli-v1.3/test-v1.3.json" \
      "$RAW/jglue/jnli-test-v1.3.json"

# NTCIR-16 Real-MedNLP test collection (MedTxt-CR-JA / MedTxt-RR-JA), CC BY 4.0.
fetch "https://sociocom.naist.jp/download/real-mednlp-test-collection/?wpdmdl=4864" "$RAW/real_mednlp/test_collection.zip"
# NTCIR-17 MedNLP-SC RR-TNM, CC BY 4.0.
fetch "https://sociocom.naist.jp/download/mednlp-sc-rr-tnm/?wpdmdl=6910" "$RAW/rr_tnm/rr_tnm.zip"

# Extract archives inside their own directories (data only; nothing from them is executed).
(cd "$RAW/real_mednlp" && rm -rf extracted && mkdir extracted && unzip -q -j test_collection.zip -d extracted)
(cd "$RAW/rr_tnm" && rm -rf extracted && mkdir extracted && unzip -q rr_tnm.zip -d extracted)

(cd "$RAW" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
echo "Wrote $(wc -l < "$RAW/SHA256SUMS") checksums to $RAW/SHA256SUMS"
