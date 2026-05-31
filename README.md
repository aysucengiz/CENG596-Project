# Clinical Trial Search

This project provides a local clinical-trial retrieval pipeline and a Streamlit UI for:
- BM25 retrieval with PyTerrier/Terrier
- dense retrieval with sentence-transformers + FAISS
- BM25+dense fusion with weighted RRF tuning
- optional cross-encoder reranking
- evaluation over the TREC Clinical Trials query set

## Files

- `clinical_trials_local_pipeline.py`: backend pipeline and CLI
- `streamlit_app.py`: Streamlit interface

## Environment

Python 3.11 is recommended. Java 17 is required for PyTerrier.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Set `JAVA_HOME` to your JDK 17 installation and add `%JAVA_HOME%\bin` to `Path`. Or add it in the environment manually:
```powershell
 $env:JAVA_HOME = "C:\Program Files\Eclipse Adoptium\jdk-17.0.19.10-hotspot"                       
$env:Path = "$env:JAVA_HOME\bin;$env:Path"
```

## Run the Streamlit app

```powershell
python -m streamlit run "C:\Users\vicy\Desktop\CENG592\streamlit_app.py"
```

## Run the CLI pipeline

BM25 only:

```powershell
python "C:\Users\vicy\Desktop\CENG592\clinical_trials_local_pipeline.py" --skip-dense --skip-rerank
```

BM25 + dense:

```powershell
python "C:\Users\vicy\Desktop\CENG592\clinical_trials_local_pipeline.py" --dense-device cpu --skip-rerank
```

BM25 + dense + reranking:

```powershell
python "C:\Users\vicy\Desktop\CENG592\clinical_trials_local_pipeline.py" --dense-device cpu --rerank-device cpu
```

## Caches

The code creates caches locally for speed. By default they are kept outside the repo:
- prepared data: `runtime/prepared_cache`
- BM25 index: `runtime/bm25_index`
- dense embeddings: `runtime/embeddings`
- ir_datasets cache: `runtime/ir_datasets_cache`
- PyTerrier jars: user-level PyTerrier cache (not stored in this repo)

These caches are excluded from git in `.gitignore` (except embeddings).
