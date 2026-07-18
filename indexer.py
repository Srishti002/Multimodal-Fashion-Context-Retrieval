import json
import faiss
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

MODEL_NAME = "openai/clip-vit-base-patch32"


def load_model():
    device="cuda" if torch.cuda.is_available() else "cpu"
    model=CLIPModel.from_pretrained(MODEL_NAME, use_safetensors=True).to(device).eval()
    processor=CLIPProcessor.from_pretrained(MODEL_NAME)
    return model, processor, device


def embed_image(model, processor, device, pil_image):
    inputs=processor(images=pil_image, return_tensors="pt").to(device)
    with torch.no_grad():
        out=model.get_image_features(**inputs)
        feats=out.pooler_output if hasattr(out, "pooler_output") else out
        feats=feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy().astype("float32").squeeze(0)


def crop_from_bbox(image, bbox):
    x, y, w, h = bbox
    x, y, w, h = int(x), int(y), int(w), int(h)
    if w <= 0 or h <= 0:
        return None
    return image.crop((x, y, x + w, y + h))


def main(dataset_json, output_index, output_metadata):
    model, processor, device=load_model()

    with open(dataset_json) as f:
        dataset=json.load(f)

    embedding_dim=None
    vectors=[]
    metadata=[]

    total=len(dataset)
    for i, (unified_id, entry) in enumerate(dataset.items()):
        image_path=entry["image_path"]
        try:
            image=Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Skipping {image_path}: {e}")
            continue

        #Whole-image "scene" vector (every image gets one)
        scene_vec=embed_image(model, processor, device, image)
        if embedding_dim is None:
            embedding_dim=scene_vec.shape[0]

        vectors.append(scene_vec)
        metadata.append({
            "unified_id": unified_id,
            "image_path": image_path,
            "region_type": "scene",
            "category": None,
            "color_name": None,
            "scene": entry["scene"],
            "source": entry["source"],
        })

        #Per garment region vectors (only if masks available)
        if entry["has_segmentation_mask"]:
            for garment in entry["garments"]:
                crop=crop_from_bbox(image, garment["bbox"])
                if crop is None or crop.width < 5 or crop.height < 5:
                    continue  

                region_vec=embed_image(model, processor, device, crop)
                vectors.append(region_vec)
                metadata.append({
                    "unified_id": unified_id,
                    "image_path": image_path,
                    "region_type": garment["category"],
                    "category": garment["category"],
                    "color_name": garment["color_name"],
                    "scene": entry["scene"],
                    "source": entry["source"],
                })

        if (i + 1) % 50 == 0:
            print(f"Indexed {i + 1}/{total} images ({len(vectors)} vectors so far)...")

    print(f"\nTotal vectors: {len(vectors)} from {total} images")

    #Building FAISS index 
    vectors_np=np.stack(vectors).astype("float32")
    index=faiss.IndexFlatIP(embedding_dim)
    index.add(vectors_np)

    faiss.write_index(index, output_index)
    with open(output_metadata, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved FAISS index to {output_index}")
    print(f"Saved metadata to {output_metadata}")

    from collections import Counter
    region_counts = Counter(m["region_type"] for m in metadata)
    print("\nVectors by region type (top 15):")
    for name, count in region_counts.most_common(15):
        print(f"  {name:35s} {count}")


if __name__ == "__main__":
  
    main("final_dataset.json", "vector_index.faiss", "vector_metadata.json")