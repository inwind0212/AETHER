# AETHER: AlphaEarth--POI Enriched Representation Learning

This repository contains the official implementation of **AETHER**, a
framework for aligning **AlphaEarth (AE) satellite embeddings** with
**urban semantic signals derived from Points-of-Interest (POIs)**.

AETHER enriches Earth observation embeddings by learning a joint
representation between **remote sensing features** and **urban semantic
information** extracted from POIs using contrastive learning.

The framework optimizes two objectives:

-   **Image--Image Alignment:** ensures spatial consistency across
    AlphaEarth embeddings.
-   **Image--Text Alignment:** aligns AE embeddings with semantic
    representations derived from POI descriptions.

The resulting representations can be used for multiple **urban
downstream tasks**.

------------------------------------------------------------------------

# 1 Installation

Clone the repository:

    git clone <repo-url>
    cd AETHER

Create a Python environment:

    python3 -m venv .venv
    source .venv/bin/activate

Install dependencies:

    pip install -r requirements.txt

------------------------------------------------------------------------

# 2 Data Preparation

## AlphaEarth Embeddings

Download AlphaEarth embeddings and place them in:

    data/AE/London_Embedding_2024AEF.tif

The raster should contain **64-dimensional AlphaEarth embeddings per
pixel**.

------------------------------------------------------------------------

## POI Dataset

Prepare a POI dataset containing:

  Field         Description
  ------------- ----------------------------------------
  name          POI name
  description   textual description used for embedding
  geometry      point geometry

Example file:

    data/poi/ld_poi_with_name.geojson

Example record:

    {
      "name": "University College London",
      "description": "University campus",
      "geometry": Point(...)
    }

------------------------------------------------------------------------

# 3 Training AETHER

Training is controlled by a configuration file.

Run:

    python train.py --cfg config.yaml

The training pipeline performs:

1.  Load POI dataset
2.  Extract AlphaEarth features around POIs
3.  Encode POI descriptions into text embeddings
4.  Train contrastive alignment

The objective:

    L = λ L_image-image + (1-λ) L_image-text

------------------------------------------------------------------------

# 4 Downstream Tasks

Aligned embeddings can be evaluated on multiple urban tasks.

Run:

    python downstream_task/main.py --cfg downstream_task/configs/ld_luc_AETHER.yaml

Supported tasks include:

-   Land‑Use Classification (LUC)
-   Socio‑Demographic Mixture Prediction (SDM)
-   GDP Prediction
-   House Price Regression
-   Region-level semantic retrieval

------------------------------------------------------------------------

# 5 Repository Structure

    AETHER/
    │
    ├── train.py
    ├── model.py
    ├── dataio.py
    ├── config.yaml
    │
    ├── downstream_task/
    │   ├── main.py
    │   ├── configs/
    │   ├── tasks/
    │   └── common/
    │
    ├── cache/
    ├── cache_downstream/
    ├── outputs/
    │
    └── README.md

------------------------------------------------------------------------

# 6 Output Structure

Training outputs are stored in:

    outputs/<experiment_name>/

Example:

    outputs/
      tri2_AEProj-TextProj_LD_t3l_pix4-aug9_tau0.07-0.07_lam0.20_h256_d128_bs512_lr1e-3_epo100/
          ckpts/
              best.pth
              last.pth

Each run also records metadata:

    run_meta.json

------------------------------------------------------------------------

# 7 Text Embeddings

Two text encoding options are supported.

### OpenAI

Set API key:

    export OPENAI_API_KEY=your_key_here

Model example:

    text-embedding-3-large

### HuggingFace

Example model:

    sentence-transformers/all-MiniLM-L6-v2

------------------------------------------------------------------------

# 8 Caching

To speed up experiments, intermediate results are cached:

    cache/
       poi/
       text/
       area/

Cached data include:

-   processed POI datasets
-   text embeddings
-   AlphaEarth feature extraction results

Caches are reused automatically if configuration hashes match.

------------------------------------------------------------------------

# 9 Reproducibility

Each run stores:

-   configuration hash
-   dataset signatures
-   cache keys
-   hyperparameters

inside:

    run_meta.json

This allows full reproducibility of experiments.

------------------------------------------------------------------------

# 10 Citation

If you use AETHER in your research, please cite:

    @article{liu2026aether,
      title={AETHER: AlphaEarth–POI Enriched Representation Learning},
      author={Liu, ...},
      journal={ISPRS Journal of Photogrammetry and Remote Sensing},
      year={2026}
    }

------------------------------------------------------------------------

# License

This project is released under the MIT License.
