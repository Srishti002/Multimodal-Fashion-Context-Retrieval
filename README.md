# Multimodal-Fashion-Context-Retrieval

A search engine that retrieves fashion images from natural language descriptions understanding not just *what* someone is wearing, but *where* they are and the *vibe* of the outfit.

## Why not just use CLIP?

Vanilla CLIP retrieval struggles with two things this project specifically targets:

- **Compositionality** — it can't reliably tell "red shirt, blue pants" from "blue shirt, red pants," since it embeds a whole image as one vector and a whole caption as one vector, with no binding between which color belongs to which item.
- **Fine-grained fashion attributes** — general-purpose training doesn't distinguish garment types as precisely as a fashion-aware pipeline needs.

This project fixes both **without any training or fine-tuning** — it stays fully zero-shot — by embedding each garment *region* separately instead of the whole image, and decomposing queries into per-garment sub-phrases matched against the correct region.

## Architecture

```
Dataset/                          
├── output_directory            # Contains Fashionpedia filtered images
├── new_images                  # Images extracted from pexels (home+office)
├── final_dataset.json          # Final dataset json file

results/                          
5 queries results

indexer.py                     #Build FAISS index: region crops + whole-image embeddings

retriever.py                   #Decompose query → region-specific search → aggregate → top-k

visualize_results.py           #Render top-k results as an image grid for eyeballing
```

**Indexer**: for each image with segmentation masks, every main garment instance (shirt, dress, pants, jacket, etc. not sub-parts like sleeve/collar) is cropped and embedded separately with CLIP. Every image, masked or not, also gets one whole-image embedding for scene/context matching. All vectors go into a FAISS flat index (cosine similarity), with metadata tracking region type, category, color, and scene per vector.

**Retriever**: a query like *"a red tie and a white shirt in a formal setting"* is split into garment sub-phrases (`red tie`, `white shirt`) plus a scene phrase (the full query). Each sub-phrase is embedded and searched only against its matching region type, so color is checked against the correct crop, not the whole image. Scores are aggregated with a weighted average (not summed), so images with more regions aren't unfairly favored.

## Dataset

~980 images, built from:
- **[Fashionpedia](https://github.com/cvdfoundation/fashionpedia)** (val split, CC-BY-4.0) — 800 images, curated for color and category balance, with real segmentation masks.
- **[Pexels](https://www.pexels.com/api/)** — 180 images (90 office + 90 home), added because Fashionpedia's images skew heavily toward studio/street photography with almost no office/home representation. These lack segmentation masks; the retriever falls back to whole-image matching for garment queries on these images
Fashionpedia has **no color or location annotations at all** — both were derived: color via k-means clustering on masked pixels, scene via zero-shot CLIP classification. Full methodology, including the color-imbalance problem found and how it was corrected, is in the report.

## Setup

```bash
conda create -n fashion-retrieval python=3.10 -y
conda activate fashion-retrieval
pip install torch transformers pillow faiss-cpu numpy pycocotools scikit-learn requests matplotlib
```

## Usage

**Visualize results:**
```bash
python visualize_results.py --index vector_index.faiss --metadata vector_metadata.json --query "A red tie and a white shirt in a formal setting" --top_k 10
```

## Evaluation

Results for all 5 required queries, including honest failure analysis

## Known limitations

- CLIP conflates adjacent hues (yellow/mustard/tan) — color matching isn't pixel-precise.
- No action/pose understanding — "sitting on a bench" is not distinguished from "standing near a bench."
- Categories with very few dataset instances (e.g. ties: 3 total) have weak retrieval simply from lack of data, not an architecture flaw.
- SigLIP-2 was the original intended embedding model for its stronger fine-grained retrieval performance, but a confirmed upstream `transformers` checkpoint bug forced a fallback to CLIP
