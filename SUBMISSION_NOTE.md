# Submission note

Two different packages exist. Do not mix them up.

| Package | Where | Contains photos? | Public? |
|---|---|---|---|
| Public repository | GitHub | **No**: code, metadata.csv (source URLs + evidence), README, QC report, charts | yes |
| `submission_full/` | this laptop only (made by `python scripts/make_submission.py`, run config "15 Make submission folder") | **Yes**: `dataset/images/` and the derived masks | **no** |

## Licence-restricted parts

- `submission_full/dataset/images/` — processed copies of photos from public web pages. We could not confirm a
  licence that allows redistribution, so we treat every image as "all rights reserved".
- `submission_full/dataset/annotations/masks/` — derived from those photos.
- The Word report, the PDF and the slides contain no source photos, but they contain names, a phone number and
  an e-mail address, so they are also kept out of the public repository.

The brief (section 7) says: *if the licence does not allow redistribution, do not put the source images in the
submission package; give source records, licence details and access instructions instead.*

**Before submitting `submission_full/`, ask the teacher (黄珊珊) whether the processed images may be handed in
privately for grading.** If not, delete `submission_full/dataset/images/` and `submission_full/dataset/annotations/masks/`
before handing it in: `metadata.csv` (source_url, image_url, publisher, evidence, licence note) plus the code are
the lawful-access instructions.

## The public repository already has images in its history

Changing `.gitignore` does not remove files that were committed before. Safest order:

1. On GitHub: Settings → General → Danger Zone → **Change visibility → Private** (do this first).
2. Either keep it private, or make a **new, clean public repository** from the current tree
   (with the new `.gitignore`, so no images are added) and keep the old one private.
3. If you keep the old repository, untrack the restricted files once:
   `git rm -r --cached dataset/images dataset/annotations/masks docs/*.docx docs/*.pptx` and commit.
   The old commits still contain the images, which is why step 1 matters.

Nobody rewrote the git history; that choice is yours.
