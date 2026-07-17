---
license: mit
---
# Volleylitics Whistle Detection Dataset

## Overview

This dataset contains full match audio recordings and whistle annotations
for training and evaluating whistle detection models in indoor volleyball environments.

The dataset is designed for:

- DSP-based whistle detection
- Audio classification (CNN)
- Acoustic event detection research

---

## Audio Format

- Format: WAV
- Sample rate: 22.05kHz 

---

## Structure

match1.wav 
match2.wav  
...  

whistles_all.json
whistles_all_anchored.json
whistles_all_reanchored.json

---

## Annotation Format

Each annotation JSON contains:

```json
{
        "match_id": "match1",
        "whistle_id": 0,
        "time": 395.705,
        "type": "other",
        "t_raw": 395.705,
        "global_id": 0,
        "t_anchor": 395.663
    },
    {
        "match_id": "match1",
        "whistle_id": 1,
        "time": 396.473,
        "type": "other",
        "t_raw": 396.473,
        "global_id": 1,
        "t_anchor": 396.303
    },
```
t_raw is the time of a manual labeling to a whistle sound.
t_anchor is the time of anchoring by some features (band_peak)