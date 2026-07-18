import json
import re
from collections import defaultdict

import faiss
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

MODEL_NAME = "openai/clip-vit-base-patch32"

KNOWN_GARMENTS = [
    "shirt, blouse", "top, t-shirt, sweatshirt", "sweater", "cardigan",
    "jacket", "vest", "pants", "shorts", "skirt", "coat", "dress",
    "jumpsuit", "cape", "bag, wallet", "belt", "glasses", "hat",
    "headband, head covering, hair accessory", "tie", "glove", "watch",
    "shoe", "scarf",
]

GARMENT_ALIASES = {
    "shirt": "shirt, blouse", "blouse": "shirt, blouse",
    "top": "top, t-shirt, sweatshirt", "t-shirt": "top, t-shirt, sweatshirt",
    "tshirt": "top, t-shirt, sweatshirt", "sweatshirt": "top, t-shirt, sweatshirt",
    "hoodie": "top, t-shirt, sweatshirt",
    "sweater": "sweater", "cardigan": "cardigan",
    "jacket": "jacket", "blazer": "jacket", "raincoat": "coat", "coat": "coat",
    "vest": "vest", "pants": "pants", "trousers": "pants", "jeans": "pants",
    "shorts": "shorts", "skirt": "skirt", "dress": "dress", "gown": "dress",
    "jumpsuit": "jumpsuit", "cape": "cape", "bag": "bag, wallet",
    "wallet": "bag, wallet", "belt": "belt", "glasses": "glasses",
    "sunglasses": "glasses", "hat": "hat", "cap": "hat",
    "headband": "headband, head covering, hair accessory",
    "tie": "tie", "necktie": "tie", "glove": "glove", "gloves": "glove",
    "watch": "watch", "shoe": "shoe", "shoes": "shoe", "sneakers": "shoe",
    "boots": "shoe", "scarf": "scarf",
}

KNOWN_COLORS = [
    "black", "white", "gray", "grey", "red", "maroon", "pink", "orange",
    "yellow", "beige", "brown", "olive", "green", "teal", "blue", "navy",
    "purple", "denim",
]

SCENE_KEYWORDS = [
    "office", "street", "park", "home", "studio", "urban", "city",
    "business", "formal", "casual", "professional", "weekend",
]


def decompose_query(query):
    
    query_lower=query.lower()
    words=re.findall(r"[a-z]+", query_lower)

    garment_matches=[]
    for i, w in enumerate(words):
        if w in GARMENT_ALIASES:
            category=GARMENT_ALIASES[w]
            #look at the 1-2 words immediately before this garment word for a color
            color=None
            for back in (1, 2):
                if i - back >= 0 and words[i - back] in KNOWN_COLORS:
                    color=words[i - back]
                    break
            phrase=f"{color + ' ' if color else ''}{w}"
            garment_matches.append({"category": category, "color": color, "phrase": phrase})

    scene_phrase=query

    return garment_matches, scene_phrase


def load_model():
    device="cuda" if torch.cuda.is_available() else "cpu"
    model=CLIPModel.from_pretrained(MODEL_NAME, use_safetensors=True).to(device).eval()
    processor=CLIPProcessor.from_pretrained(MODEL_NAME)
    return model, processor, device


def embed_text(model, processor, device, text):
    inputs=processor(text=[text], padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        out=model.get_text_features(**inputs)
        feats=out.pooler_output if hasattr(out, "pooler_output") else out
        feats=feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy().astype("float32").squeeze(0)

#Search the full index, then filter results to a specific region_type if given
def search(index, metadata, query_vec, region_filter=None, top_n=50):
    query_vec=query_vec.reshape(1, -1)
    scores,indices=index.search(query_vec, min(top_n, index.ntotal))

    results=[]
    for score, idx in zip(scores[0], indices[0]):
        if idx ==-1:
            continue
        m=metadata[idx]
        if region_filter is not None and m["region_type"] != region_filter:
            continue
        results.append((m["unified_id"], float(score), m))
    return results


def retrieve(query, index, metadata, model, processor, device, top_k=10):
    garment_matches, scene_phrase=decompose_query(query)

    #image_id=list of (score, weight) tuples collected across all sub phrases
    image_scores=defaultdict(list)
    image_info={}

    #Scene sub-phrase: search against ALL scene vectors 
    scene_vec=embed_text(model, processor, device, scene_phrase)
    scene_results=search(index, metadata, scene_vec, region_filter="scene", top_n=index.ntotal)
    for image_id, score, m in scene_results:
        image_scores[image_id].append((score, 1.0))
        image_info[image_id]=m

    #Garment sub-phrases: search against matching region type, fallback to scene 
    for gm in garment_matches:
        phrase_text=gm["phrase"]
        phrase_vec=embed_text(model, processor, device, phrase_text)

        region_results=search(index, metadata, phrase_vec, region_filter=gm["category"], top_n=index.ntotal)
        region_image_ids={r[0] for r in region_results}
        for image_id, score, m in region_results:
            image_scores[image_id].append((score, 1.0))
            image_info[image_id]=m

        fallback_results=search(index, metadata, phrase_vec, region_filter="scene", top_n=index.ntotal)
        for image_id, score, m in fallback_results:
            if image_id in region_image_ids:
                continue  
            image_scores[image_id].append((score * 0.8, 0.8))
            image_info[image_id]=m

    #Aggregate: weighted average score per image 
    final_scores=[]
    for image_id, score_weight_pairs in image_scores.items():
        weighted_sum=sum(s * w for s, w in score_weight_pairs)
        total_weight=sum(w for _, w in score_weight_pairs)
        avg_score=weighted_sum / total_weight if total_weight > 0 else 0
        final_scores.append((image_id, avg_score, image_info[image_id]))

    final_scores.sort(key=lambda x: x[1], reverse=True)
    return final_scores[:top_k], garment_matches, scene_phrase


def main(index_path, metadata_path, query, top_k):
    index=faiss.read_index(index_path)
    with open(metadata_path) as f:
        metadata=json.load(f)
    model, processor, device=load_model()

    results, garment_matches, scene_phrase=retrieve(query, index, metadata, model, processor, device, top_k)

    print(f"\nQuery: \"{query}\"")
    print(f"Decomposed garment sub-phrases: {[g['phrase'] for g in garment_matches]}")
    print(f"Scene phrase: \"{scene_phrase}\"")
    print(f"\nTop {top_k} results:")
    for rank, (image_id, score, info) in enumerate(results, 1):
        print(f"  {rank}. {image_id}  score={score:.4f}  path={info['image_path']}  scene={info['scene']}  source={info['source']}")


if __name__ == "__main__":

    main("vector_index.faiss", "vector_metadata.json", "A person in a bright yellow raincoat", 10)

