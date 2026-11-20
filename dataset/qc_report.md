# QC report

- Images: **1124** (checked on disk: 1124)
- Largest / smallest group: 2.19

## Result

**PASSED** - no hard failures.

### Hard failures

None.


### Warnings

None.


## Per group

| Group | train | val | test |
|---|---|---|---|
| BA | 143 | 31 | 30 |
| MA | 179 | 38 | 38 |
| UY | 214 | 46 | 46 |
| YI | 153 | 33 | 33 |
| ZH | 98 | 21 | 21 |

## Pixel share by class

- background: 76.8%
- top: 4.0%
- bottom: 8.1%
- dress: 6.7%
- headwear: 2.3%
- footwear: 0.2%
- waistwear: 2.0%

## Checks performed

classes.json matches config; metadata columns and required fields; unique ids; verified source labels; mask is single-channel uint8, 512x512, valid ids, same name as image; image RGB 512x512; background pixels exactly (255,255,255); labels only on the subject; subject centre within 10% and longer side >= 60%; per-group and total minimums; split ratios; no source page or near-duplicate across splits; every class used.
