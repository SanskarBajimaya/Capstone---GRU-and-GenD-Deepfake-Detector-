# Local-only DVC setup

## Purpose and safety boundary

This repository uses DVC for local version receipts while keeping private research data out of future Git commits. Git stores source code and approved metadata. DVC stores content in the machine-local `.dvc/cache` and, when explicitly configured, in a separate local backup directory. No cloud or internet DVC remote is configured.

A `.dvc` file is a small content receipt, not the dataset. It cannot reconstruct private data by itself. Keep the local DVC cache and an independently protected backup. Team members still need independently authorized access to the underlying data.

## Current audit and migration boundary

The following paths were present when DVC was initialized:

| Path | Contents | Files | Approximate size | Git status | DVC status | Referenced by code |
| --- | --- | ---: | ---: | --- | --- | --- |
| `data/faceforensic/original_sequences` | FaceForensics++ real videos | 510 | 941.51 MB | Tracked | Not tracked (blocked) | Yes |
| `data/faceforensic/manipulated_sequences` | FaceForensics++ Deepfakes videos | 510 | 973.02 MB | Tracked | Not tracked (blocked) | Yes |
| `datasets/CDFv2` | Extracted Celeb-DF v2 sample frames | 12 | 0.64 MB | Tracked | Not tracked (blocked) | Used by dataset/evaluation workflows |
| `datasets/FF` | Extracted FaceForensics++ sample frames | 20 | 1.57 MB | Tracked | Not tracked (blocked) | Used by dataset workflows |
| `temporal/features` | Legacy cached CLIP tensors | 20 | 1.28 MB | Tracked | Not tracked (blocked) | Yes |
| `temporal/features_64` | 64-frame CLIP master features | Not present | — | Ignored for future generation | No receipt | Yes |
| `temporal/features_pe32` | 32-frame PE master features | Not present | — | Ignored for future generation | No receipt | Yes |
| `temporal/manifests` | Video manifest and train/validation/test split metadata | 2 | 0.41 MB | Tracked | Approved Git metadata, not a DVC output | Yes |
| `temporal/models/best_gru.pt` | Legacy 16-frame GRU checkpoint | 1 | 1.69 MB | Tracked | Not tracked (blocked) | Legacy code only |
| `temporal/models/gru32_seed*_best.pt` | CLIP-GRU-32 checkpoints | Not present | — | Directory ignored for future local files | No receipt | Frontend and evaluation |
| `temporal/models_pe/gru32_pe_seed*_best.pt` | PE-GRU-32 checkpoints | Not present | — | Directory ignored for future local files | No receipt | Frontend and evaluation |
| `temporal/results` | Approved experiment reports and tables | 39 | 0.15 MB | Tracked | Not tracked | Research reporting |
| `temporal/results_pe` | Approved PE/comparison reports and tables | 15 | 0.08 MB | Tracked; tensor caches ignored | No receipt | Evaluation/comparison |
| `weights/models/buffalo_l/det_10g.onnx` | RetinaFace detector weight | 1 | 16.14 MB | Ignored local file | No receipt | Frontend preprocessing |
| Approved Hugging Face cache outside this repository | Original GenD-CLIP and GenD-PE weights | Environment-dependent | Environment-dependent | Outside Git | Outside this DVC project | Frontend and feature builders |

Git and DVC must not own the same file. Therefore, no current dataset or artifact was automatically added to DVC. Ignore rules prevent new files under private/generated paths from being added accidentally, but ignore rules do not remove existing data from Git history or from GitHub.

Migrating an already Git-tracked dataset requires team approval and a dedicated reviewed change. First make and test an authorized backup. Then remove only the approved target from the Git index while retaining the working files, verify every file remains on disk, and run DVC on the unchanged path. For example, the reviewable top-level migrations would use `git rm -r --cached -- data` followed by `python -m dvc add data`, or the equivalent pair for `datasets`. These commands are a proposal, not part of this setup; never run them without approval. Review the resulting `data.dvc` or `datasets.dvc` receipt, ignore changes, and staged deletion list before committing. Address historical exposure separately if policy requires it. Do not rewrite history without a separately approved incident/remediation plan.

## Local storage layout

- Working data remains at its existing project path so research scripts continue to resolve it.
- DVC's local cache is `.dvc/cache` and is ignored by Git.
- Machine-specific DVC settings belong in `.dvc/config.local`, which is ignored by Git.
- DVC analytics are disabled in that local configuration (`core.analytics=false`).
- No DVC remote is currently configured. The audited machine exposed only one filesystem volume, so another directory on that volume would not be an independent physical backup.

Do not place a local remote inside the dataset, project cache, or repository. If a separate protected drive becomes available, first confirm its capacity and then configure it without committing its absolute path:

```powershell
python -m dvc remote add --local -d private-store "X:\approved\gend-dvc-store"
python -m dvc remote list
git check-ignore -v .dvc/config.local
```

Replace the example with an approved path on a separate local device. Review expected copy size before running `python -m dvc push`. A DVC push copies data to that configured store; a GitHub push never transfers DVC data.

## Adding an approved, untracked local dataset

Before adding any target, confirm that it is not already tracked by Git or another DVC output, that sufficient cache space is available, and that its current path is the path expected by the code:

```powershell
git ls-files -- "path/to/approved-data"
Get-ChildItem -Recurse -Filter *.dvc
python -m dvc status
Get-PSDrive -PSProvider FileSystem
python -m dvc add "path/to/approved-data"
python -m dvc status
```

Review the generated `.dvc` receipt and `.gitignore` change. DVC may use links or copies between the working tree and cache; do not move, rename, or delete the working data as part of this workflow.

To record a later approved update:

1. Update the data at the same local path.
2. Run `python -m dvc add "path/to/approved-data"` again.
3. Review the changed `.dvc` receipt and ignore rules.
4. Verify that no raw files are staged.
5. Commit only the approved receipt and metadata after team review.
6. Push the Git commit to the approved feature branch separately.

This versions metadata; it does not automate dataset sharing.

## Restoring from local storage

If the content is already in `.dvc/cache`, restore the working copy with:

```powershell
python -m dvc checkout
```

If an authorized separate local remote is configured and contains the data:

```powershell
python -m dvc pull
```

If both the working data and all DVC cache/backup copies are lost, the `.dvc` receipt alone cannot recover the data. Reacquire it from an authorized source, verify it, and rebuild the receipt if necessary. Back up private data to encrypted, access-controlled local storage according to the dataset licenses and institutional policy; test restoration periodically.

## Checkpoints and inference on another computer

Checkpoints are separate from training datasets. The frontend requires these exact local files for the temporal configurations:

- `temporal/models/gru32_seed0_best.pt` through `gru32_seed4_best.pt`
- `temporal/models_pe/gru32_pe_seed0_best.pt` through `gru32_pe_seed4_best.pt`

All ten were absent during this setup. Do not substitute `temporal/models/best_gru.pt`, which is a legacy 16-frame checkpoint. When the approved GRU-32 checkpoints become available, keep their paths unchanged and track each checkpoint directory as a separate DVC target only after confirming the files are not Git-tracked. Distribution must use an authorized private channel or an approved private DVC remote, never the public repository.

For standard frontend inference on another computer:

- GitHub supplies the application, research code, manifests, and approved metadata.
- Install Python dependencies from the repository requirements. The native video stack and a supported PyTorch runtime must also be available.
- Original GenD-CLIP (`yermandy/GenD_CLIP_L_14`) and GenD-PE (`yermandy/GenD_PE_L`) weights are loaded from their approved Hugging Face model sources or an existing authorized local model cache.
- The RetinaFace detector weight is downloaded to `weights/models/buffalo_l/det_10g.onnx` when network/model-source access permits, or it must be supplied from its approved source.
- The exact ten local GRU-32 checkpoints above are required for CLIP-GRU, PE-GRU, and the temporal ensemble.
- Private FaceForensics++ and Celeb-DF training/evaluation datasets are not required to analyze a newly uploaded video.

Cloning GitHub alone therefore does not guarantee that all six detector configurations can run. Missing checkpoint-backed configurations remain unavailable by design.

## Preventing accidental publication

Before every commit, inspect both staged and untracked files:

```powershell
git status --short --branch
git diff --cached --name-only
git diff --cached --stat
git ls-files data datasets temporal/models temporal/models_pe weights
git check-ignore -v data datasets temporal/features_64 temporal/features_pe32 temporal/models temporal/models_pe weights .dvc/cache .dvc/config.local .env
```

Do not use `git add -f` on ignored research paths. If any raw video, extracted private frame, generated feature tensor, checkpoint, DVC cache object, credential, or local config appears staged, stop and remove only that path from the index before proceeding. Never use `.gitignore` as evidence that an already committed file was removed from Git history.

## Future private shared remote

A shared remote is intentionally out of scope. If the team later approves one, it needs a documented storage owner, dataset-license review, encryption, access controls, credential rotation, backup/retention policy, capacity estimate, and an explicit list of permitted DVC targets. Store credentials outside Git, keep machine secrets in local configuration, test access with non-sensitive data, and obtain review before any `dvc push`.
