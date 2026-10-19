### test_synth

| Condition | A: entity F1 | B: doc-type macro-F1 | C: field accuracy |
|---|---|---|---|
| E1 Qwen3-8B | 95.1 [94.0–96.2] | 62.2 [56.5–67.4] | 90.6 [89.4–91.8] |
| E2 Swallow-8B | 87.0 [84.9–89.0] | 40.4 [34.0–46.1] | 87.6 [86.1–89.0] |
| E2t Swallow-8B (think) | 96.6 [94.3–98.4] | 73.3 [63.7–82.5] | 96.1 [94.5–97.4] |
| E3 Qwen3-8B + LoRA | 99.7 [99.4–99.9] | 98.8 [97.4–100.0] | 99.3 [98.9–99.7] |
| E4 GPT-4.1 | 96.2 [95.0–97.3] | 71.1 [66.4–75.7] | 90.2 [89.0–91.4] |
| E4b GPT-5.4-mini | 95.6 [94.4–96.7] | 76.6 [71.2–81.3] | 94.0 [93.1–94.9] |

### test_xgen

| Condition | A: entity F1 | B: doc-type macro-F1 | C: field accuracy |
|---|---|---|---|
| E1 Qwen3-8B | 94.7 [93.5–95.8] | 33.5 [26.7–40.3] | 90.9 [89.6–92.1] |
| E2 Swallow-8B | 87.8 [85.6–89.8] | 21.3 [14.3–28.2] | 89.5 [88.2–90.8] |
| E2t Swallow-8B (think) | 96.2 [94.6–97.6] | 45.8 [36.3–54.0] | 92.0 [89.8–93.9] |
| E3 Qwen3-8B + LoRA | 98.9 [98.4–99.4] | 78.9 [71.8–85.0] | 97.3 [96.4–98.0] |
| E4 GPT-4.1 | 95.8 [94.8–96.8] | 55.6 [47.7–62.3] | 91.0 [89.6–92.2] |
| E4b GPT-5.4-mini | 95.6 [94.7–96.6] | 57.9 [49.8–64.7] | 92.3 [91.1–93.5] |

### test_ext

| Condition | MRNER-disease F1 | MRNER-medicine F1 | CRADE acc. | RRTNM acc. |
|---|---|---|---|---|
| E1 Qwen3-8B | 26.5 [23.0–30.1] | 24.9 [19.1–31.9] | 50.0 [40.0–60.0] | 77.0 [68.0–85.0] |
| E2 Swallow-8B | 17.1 [14.0–20.1] | 17.1 [12.8–22.7] | 30.0 [21.0–39.0] | 68.0 [59.0–77.0] |
| E2t Swallow-8B (think) | 11.9 [7.2–16.6] | 50.2 [39.4–60.5] | 56.7 [43.3–68.3] | 80.0 [70.0–90.0] |
| E3 Qwen3-8B + LoRA | 47.3 [42.2–52.4] | 21.5 [15.5–28.7] | 42.0 [32.0–52.0] | 69.0 [60.0–78.0] |
| E4 GPT-4.1 | 11.0 [7.3–14.7] | 38.9 [30.6–47.6] | 52.0 [42.0–62.0] | 91.0 [85.0–96.0] |
| E4b GPT-5.4-mini | 19.3 [15.1–23.4] | 42.2 [33.9–50.5] | 55.0 [45.0–65.0] | 97.0 [93.0–100.0] |

### test_general

| Condition | JCommonsenseQA acc. | JNLI acc. | JMMLU-Med acc. |
|---|---|---|---|
| E1 Qwen3-8B | 90.3 [88.6–92.0] | 65.8 [63.9–67.7] | 65.0 [55.0–74.0] |
| E2 Swallow-8B | 92.6 [91.0–94.1] | 63.8 [62.0–65.7] | 71.0 [62.0–80.0] |
| E3 Qwen3-8B + LoRA | 89.7 [87.9–91.4] | 67.0 [65.2–68.9] | 71.0 [62.0–80.0] |
| E4 GPT-4.1 | 97.0 [96.0–97.9] | 82.9 [81.4–84.4] | 88.0 [81.0–94.0] |

### A_secondary_synth

| Condition | micro_precision | micro_recall | micro_f1 | exact_match | json_valid_rate | schema_valid_rate |
|---|---|---|---|---|---|---|
| E1 Qwen3-8B | 94.4 | 95.8 | 95.1 | 60.1 | 100.0 | 100.0 |
| E2 Swallow-8B | 85.7 | 88.3 | 87.0 | 31.8 | 100.0 | 86.5 |
| E2t Swallow-8B (think) | 95.7 | 97.5 | 96.6 | 76.7 | 98.3 | 98.3 |
| E3 Qwen3-8B + LoRA | 99.7 | 99.7 | 99.7 | 97.3 | 100.0 | 100.0 |
| E4 GPT-4.1 | 93.6 | 98.9 | 96.2 | 74.9 | 100.0 | 100.0 |
| E4b GPT-5.4-mini | 92.5 | 98.9 | 95.6 | 69.5 | 100.0 | 100.0 |

### C_secondary_synth

| Condition | field_accuracy | exact_match | list_micro_f1 | missing_fields_total | unsupported_rate | schema_valid_rate |
|---|---|---|---|---|---|---|
| E1 Qwen3-8B | 90.6 | 35.0 | 90.1 | 1 | 0.1 | 99.6 |
| E2 Swallow-8B | 87.6 | 27.4 | 87.0 | 11 | 1.3 | 93.3 |
| E2t Swallow-8B (think) | 96.1 | 65.0 | 96.9 | 0 | 0.0 | 71.7 |
| E3 Qwen3-8B + LoRA | 99.3 | 94.2 | 99.5 | 0 | 0.0 | 100.0 |
| E4 GPT-4.1 | 90.2 | 35.9 | 88.8 | 0 | 0.1 | 100.0 |
| E4b GPT-5.4-mini | 94.0 | 47.5 | 93.1 | 0 | 0.0 | 100.0 |

### B_secondary_synth

| Condition | accuracy | macro_f1 | invalid_rate |
|---|---|---|---|
| E1 Qwen3-8B | 61.0 | 62.2 | 0.0 |
| E2 Swallow-8B | 41.3 | 40.4 | 0.0 |
| E2t Swallow-8B (think) | 76.7 | 73.3 | 0.0 |
| E3 Qwen3-8B + LoRA | 98.7 | 98.8 | 0.0 |
| E4 GPT-4.1 | 71.7 | 71.1 | 0.0 |
| E4b GPT-5.4-mini | 75.8 | 76.6 | 0.0 |

### H1

| Split | Task | Δ (pp) | 95% CI (pp) | p (bootstrap) | p (Holm) | McNemar p |
|---|---|---|---|---|---|---|
| test_synth | A: entity F1 | +4.6 | [+3.6, +5.7] | 0.0000 | 0.0000 | – |
| test_synth | B: doc-type macro-F1 | +36.6 | [+31.4, +42.4] | 0.0000 | 0.0000 | 0.0000 |
| test_synth | C: field accuracy | +8.7 | [+7.5, +9.9] | 0.0000 | 0.0000 | – |
| test_xgen | A: entity F1 | +4.2 | [+3.2, +5.3] | 0.0000 | 0.0000 | – |
| test_xgen | B: doc-type macro-F1 | +45.5 | [+37.2, +53.3] | 0.0000 | 0.0000 | 0.0000 |
| test_xgen | C: field accuracy | +6.4 | [+5.4, +7.4] | 0.0000 | 0.0000 | – |
| test_ext | CRADE acc. | -8.0 | [-18.0, +1.0] | 0.1146 | 0.2292 | 0.1516 |
| test_ext | MRNER-disease F1 | +20.8 | [+15.1, +26.6] | 0.0000 | 0.0000 | – |
| test_ext | MRNER-medicine F1 | -3.4 | [-9.6, +3.0] | 0.2952 | 0.2952 | – |
| test_ext | RRTNM acc. | -8.0 | [-16.0, +0.0] | 0.0512 | 0.1536 | 0.0768 |
| test_general | JCommonsenseQA acc. | -0.6 | [-1.6, +0.4] | 0.2380 | 0.2380 | 0.2810 |
| test_general | JMMLU-Med acc. | +6.0 | [+2.0, +11.0] | 0.0054 | 0.0162 | 0.0312 |
| test_general | JNLI acc. | +1.2 | [+0.4, +2.0] | 0.0064 | 0.0162 | 0.0073 |

### H2

| Split | Task | Δ (pp) | 95% CI (pp) | p (bootstrap) | p (Holm) | McNemar p |
|---|---|---|---|---|---|---|
| test_synth | A: entity F1 | +12.7 | [+10.7, +14.8] | 0.0000 | 0.0000 | – |
| test_synth | B: doc-type macro-F1 | +58.4 | [+52.6, +65.0] | 0.0000 | 0.0000 | 0.0000 |
| test_synth | C: field accuracy | +11.8 | [+10.4, +13.3] | 0.0000 | 0.0000 | – |
| test_xgen | A: entity F1 | +11.1 | [+9.2, +13.2] | 0.0000 | 0.0000 | – |
| test_xgen | B: doc-type macro-F1 | +57.6 | [+49.2, +65.9] | 0.0000 | 0.0000 | 0.0000 |
| test_xgen | C: field accuracy | +7.7 | [+6.6, +8.9] | 0.0000 | 0.0000 | – |
| test_ext | CRADE acc. | +12.0 | [+3.0, +21.0] | 0.0138 | 0.0414 | 0.0227 |
| test_ext | MRNER-disease F1 | +30.2 | [+24.6, +36.0] | 0.0000 | 0.0000 | – |
| test_ext | MRNER-medicine F1 | +4.3 | [-1.8, +11.2] | 0.1654 | 0.3308 | – |
| test_ext | RRTNM acc. | +1.0 | [-5.0, +8.0] | 0.8754 | 0.8754 | 1.0000 |

### E4

| Split | Task | Δ (pp) | 95% CI (pp) | p (bootstrap) | p (Holm) | McNemar p |
|---|---|---|---|---|---|---|
| test_synth | A: entity F1 | +3.5 | [+2.5, +4.6] | 0.0000 | 0.0000 | – |
| test_synth | B: doc-type macro-F1 | +27.7 | [+22.9, +32.6] | 0.0000 | 0.0000 | 0.0000 |
| test_synth | C: field accuracy | +9.1 | [+7.9, +10.4] | 0.0000 | 0.0000 | – |
| test_xgen | A: entity F1 | +3.1 | [+2.3, +4.0] | 0.0000 | 0.0000 | – |
| test_xgen | B: doc-type macro-F1 | +23.3 | [+16.2, +30.6] | 0.0000 | 0.0000 | 0.0000 |
| test_xgen | C: field accuracy | +6.3 | [+5.2, +7.4] | 0.0000 | 0.0000 | – |
| test_ext | CRADE acc. | -10.0 | [-20.0, +1.0] | 0.0800 | 0.0800 | 0.0987 |
| test_ext | MRNER-disease F1 | +36.3 | [+30.1, +42.8] | 0.0000 | 0.0000 | – |
| test_ext | MRNER-medicine F1 | -17.4 | [-25.0, -9.8] | 0.0000 | 0.0000 | – |
| test_ext | RRTNM acc. | -22.0 | [-31.0, -14.0] | 0.0000 | 0.0000 | 0.0000 |
| test_general | JCommonsenseQA acc. | -7.3 | [-9.2, -5.5] | 0.0000 | 0.0000 | 0.0000 |
| test_general | JMMLU-Med acc. | -17.0 | [-25.0, -9.0] | 0.0000 | 0.0000 | 0.0002 |
| test_general | JNLI acc. | -15.9 | [-17.6, -14.1] | 0.0000 | 0.0000 | 0.0000 |