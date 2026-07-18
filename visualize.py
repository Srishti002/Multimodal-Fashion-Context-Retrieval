import sys

import faiss
import json
import matplotlib.pyplot as plt
from PIL import Image


sys.path.insert(0, ".")
from retriever import retrieve, load_model 


def main(index_path, metadata_path, query, top_k):
    index=faiss.read_index(index_path)
    with open(metadata_path) as f:
        metadata=json.load(f)
    model, processor, device=load_model()

    results, garment_matches, scene_phrase=retrieve(
        query, index, metadata, model, processor, device, top_k
    )

    print(f"Query: \"{query}\"")
    print(f"Decomposed garment sub-phrases: {[g['phrase'] for g in garment_matches]}")

    cols=5
    rows=(len(results) + cols - 1) // cols
    fig, axes=plt.subplots(rows, cols, figsize=(4 * cols, 4.5 * rows))
    axes=axes.flatten() if len(results) > 1 else [axes]

    for i, (image_id, score, info) in enumerate(results):
        ax=axes[i]
        try:
            img=Image.open(info["image_path"]).convert("RGB")
            ax.imshow(img)
        except Exception as e:
            ax.text(0.5, 0.5, f"Failed to load:\n{e}", ha="center", va="center")
        ax.set_title(f"#{i+1} score={score:.3f}\n{info['scene']} / {info['source']}", fontsize=9)
        ax.axis("off")

    #Hiding any unused subplot slots
    for j in range(len(results), len(axes)):
        axes[j].axis("off")

    fig.suptitle(f'Query: "{query}"', fontsize=13)
    plt.tight_layout()
    plt.savefig("query_results2.png", dpi=120, bbox_inches="tight")
    print("\nSaved grid to query_results2.png")
    plt.show()


if __name__ == "__main__":

    main("vector_index.faiss", "vector_metadata.json","A person in a bright yellow raincoat", 10)