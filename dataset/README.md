# Ethnic Clothing Part Segmentation

Semantic segmentation dataset of Chinese ethnic clothing: 1124 images, 5 ethnic groups, 6 clothing-part classes. Built 2026-11-21 with ethnoseg 1.0.0.

## Ethnic categories

| Code | Ethnicity | 民族 | train | val | test | Total |
|---|---|---|---|---|---|---|
| YI | Yi | 彝族 | 153 | 33 | 33 | 219 |
| BA | Bai | 白族 | 143 | 31 | 30 | 204 |
| ZH | Zhuang | 壮族 | 98 | 21 | 21 | 140 |
| UY | Uygur | 维吾尔族 | 214 | 46 | 46 | 306 |
| MA | Manchu | 满族 | 179 | 38 | 38 | 255 |
| | **Total** | | 787 | 169 | 168 | **1124** |

Labels come only from the source page (see `verify_status` and `evidence` in metadata.csv); they were never inferred from how the clothing or the person looks.

## Mask classes (`annotations/classes.json`)

| ID | Class | SAM 3 prompts | Pixel share |
|---|---|---|---|
| 0 | background | - | 76.8% |
| 1 | top | jacket, blouse, shirt, vest | 4.0% |
| 2 | bottom | skirt, trousers | 8.1% |
| 3 | dress | dress, robe | 6.7% |
| 4 | headwear | hat, headdress, headscarf | 2.3% |
| 5 | footwear | shoe, boot | 0.2% |
| 6 | waistwear | apron, belt, sash | 2.0% |

## Annotation rules

- 0 = background **and** every non-clothing pixel of the subject (skin, hair, hands).
- Only visible pixels are labelled; hidden parts are not guessed.
- Overlaps: classes are painted in the order dress, bottom, top, waistwear, footwear, headwear; a later class wins, so headwear/footwear/waistwear are drawn on top of the garment they sit on.
- `top` = upper garment ending around the waist/hip; `bottom` = skirt or trousers; `dress` = one-piece garment or robe covering upper and lower body.
- Two-piece rule: when both top and bottom cover >= 4% of the subject, remaining `dress` pixels are reassigned to top or bottom by vertical position.
- `waistwear` = apron, belt or sash worn over another garment.
- Each image has a same-name mask of identical size (512x512); masks are single-channel PNG.

## Directory and fields

- `images/<split>/<ID>.png`: RGB, pure white background (255,255,255), subject centred, longer side = 90% of 512. PNG avoids JPEG noise in the white background.
- `annotations/masks/<split>/<ID>.png`: class ids 0-6.
- `annotations/classification_labels.csv`: image_id, path, ethnicity, garment_type, split.
- `metadata.csv`: image_id, path, mask_path, ethnicity, ethnicity_zh, ethnicity_code, garment_type, split, source_url, image_url, publisher, page_title, collected_at, license, verify_status, evidence_type, evidence, source_category, search_query, processing, review, reviewer, reviewed_at, cluster.
- `garment_type`: parts present (>= 1% of clothing pixels), joined with `+`.
- `cluster`: images from the same source page; a cluster is never split across train/val/test.

## Processing

`sam3(facebook/sam3) subject+parts @1280px, part_thr=0.4; white bg; fit=0.9; pad 512x512; human review`

- Split: train 70%, val 15%, test 15% per group, seed 2026.

## Licence

Images come from public web pages and are assumed to be all-rights-reserved. Use `source_url` / `image_url` in metadata.csv to obtain the originals; do not redistribute.

## Software

- Python 3.12.10 on Windows (local stages)
- SAM 3 via Hugging Face transformers (Sam3Model), run on Modal GPU

## How to rebuild

See the project README (run steps 1-11).
