#hedhi feha el model kifeh ta3malou bedhabt bech ye5dem
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util
from datasets import load_dataset
import pickle
from typing import List, Tuple, Dict
import random
import warnings

# Ignore potential warnings from libraries
warnings.filterwarnings('ignore')

# Ensure necessary libraries are installed
try:
    from sentence_transformers import SentenceTransformer, util
    from datasets import load_dataset
    import pickle
    import numpy as np
    import matplotlib.pyplot as plt
except ImportError:
    print("Installing necessary libraries...")
    from sentence_transformers import SentenceTransformer, util
    from datasets import load_dataset
    import pickle
    import numpy as np
    import matplotlib.pyplot as plt
    # In case of Colab, pickle5 might be needed for Python < 3.8
    try:
        import pickle5 as pickle
    except ImportError:
        pass


# Define the class for image search evaluation
class ImageSearchEvaluator:
    def __init__(self, embeddings_path: str, metadata_path: str):
        """Initialize the evaluator with pre-computed embeddings and metadata."""
        print("Loading SentenceTransformer model...")
        self.model = SentenceTransformer('clip-ViT-B-32')

        print(f"Loading image embeddings from {embeddings_path}...")
        try:
            self.image_embeddings = np.load(embeddings_path)
        except FileNotFoundError:
            print(f"Error: Embeddings file not found at {embeddings_path}. "
                  "Please make sure you have pre-computed and saved the embeddings.")
            print("Hint: Run the embedding generation code first.")
            self.image_embeddings = None # Set to None to indicate failure
            return # Exit init if file not found

        print(f"Loading metadata from {metadata_path}...")
        try:
            with open(metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
        except FileNotFoundError:
            print(f"Error: Metadata file not found at {metadata_path}. "
                  "Please make sure you have pre-computed and saved the metadata.")
            print("Hint: Run the embedding generation code first.")
            self.metadata = None # Set to None to indicate failure
            self.image_embeddings = None # Also set embeddings to None if metadata fails
            return # Exit init if file not found
        except Exception as e:
             print(f"Error loading metadata: {e}")
             self.metadata = None
             self.image_embeddings = None
             return

        # Load original dataset for images to display
        print("Loading image dataset...")
        try:
            self.dataset = load_dataset("ashraq/fashion-product-images-small", split="train")
            self.images = self.dataset["image"]
        except Exception as e:
            print(f"Error loading dataset: {e}")
            self.images = None
            # Continue without images if dataset loading fails, search functions will still work


        if self.image_embeddings is not None:
            print(f"Loaded {len(self.image_embeddings)} embeddings")
            print(f"Embedding dimension: {self.image_embeddings.shape[1]}")
        if self.metadata is not None:
            print(f"Loaded metadata for {len(self.metadata['product_names'])} items.")
        if self.images is not None:
             print(f"Loaded {len(self.images)} images from the dataset.")


    def search_by_text(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Search images by text query."""
        if self.image_embeddings is None or self.metadata is None:
            print("Evaluator not initialized properly due to missing files.")
            return []

        print(f"Encoding query: '{query}'")
        query_embedding = self.model.encode([query], convert_to_tensor=True)

        print("Calculating similarities...")
        similarities = util.cos_sim(query_embedding, self.image_embeddings)[0]

        print(f"Finding top {top_k} results...")
        # top_k + 1 to potentially exclude the query image itself if it's in the dataset
        top_results = similarities.argsort(descending=True)[:top_k]

        # Convert tensor results to list of tuples (index, score)
        results = [(idx.item(), similarities[idx].item()) for idx in top_results]

        print(f"Found {len(results)} results.")
        return results

    def demo_search_and_print_images(self, query: str, top_k: int = 5):
        """Demonstrate search results for a query and display the images."""
        if self.image_embeddings is None or self.metadata is None or self.images is None:
            print("Evaluator not initialized properly or images not loaded.")
            return

        print(f"\n🔍 Demo Search Results for: '{query}'")
        print("-" * 50)

        results = self.search_by_text(query, top_k)

        if not results:
            print("No results found.")
            return

        fig, axes = plt.subplots(1, top_k, figsize=(20, 10))
        # Ensure axes is always an array even if top_k is 1
        if top_k == 1:
            axes = [axes]

        fig.suptitle(f"Search Results for: '{query}'", fontsize=16)

        for i, (idx, score) in enumerate(results):
            if idx >= len(self.images):
                 print(f"Warning: Image index {idx} out of bounds. Skipping.")
                 continue

            image = self.images[idx]
            product_name = self.metadata['product_names'][idx]
            category = self.metadata['categories'][idx]
            article_type = self.metadata['article_types'][idx]
            color = self.metadata['colors'][idx]

            ax = axes[i]
            ax.imshow(image)
            # Limit title length for readability
            title_text = f"Score: {score:.3f}\n{article_type} ({category})\nColor: {color}"
            ax.set_title(title_text, fontsize=10)
            ax.axis('off')

            print(f"{i+1:2d}. Score: {score:.3f} | "
                  f"Product: {product_name[:40]:<40} | "
                  f"Category: {category:<15} | "
                  f"Type: {article_type:<15} | "
                  f"Color: {color}")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
        plt.show()


# --- Main execution block for testing queries ---

# Define paths to your embeddings and metadata files
# Make sure these paths are correct relative to your notebook
EMBEDDINGS_PATH = 'image_embeddings_clip.npy' # Update if your file name is different
METADATA_PATH = 'metadata.pkl'                 # Update if your file name is different

# Initialize the evaluator
# This will load the model, embeddings, metadata, and dataset
evaluator = ImageSearchEvaluator(
    embeddings_path=EMBEDDINGS_PATH,
    metadata_path=METADATA_PATH
)

# Check if the evaluator was initialized successfully (files were loaded)
if evaluator.image_embeddings is not None and evaluator.metadata is not None:
    print("\n✅ Evaluator initialized successfully.")

    # --- Test queries ---
    print("\n" + "="*50)
    print("TESTING SEARCH QUERIES")
    print("="*50)

    test_queries = [
        'black dress',
        'casual shirt for men',
        'running shoes',
        'blue jeans',
        'red t-shirt',
        'formal wear',
        'sunglasses',
        'backpack'
    ]

    for query in test_queries:
        # Use the demo function to search and display images
        evaluator.demo_search_and_print_images(query, top_k=5)

    print(f"\n✅ Query testing completed.")
    print(f"📁 Ensure '{EMBEDDINGS_PATH}' and '{METADATA_PATH}' exist and are accessible.")

else:
    print("\n❌ Evaluator could not be initialized. Please check file paths and ensure embeddings/metadata are generated.")

