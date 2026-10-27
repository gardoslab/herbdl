#!/bin/bash -l
# Distribution rundown: long-tail and uniform options from the cleaned pool (CPU only).
# Same env and module-load caveat as run_build_index.sh.
module load miniconda
module load academic-ml/fall-2026
conda activate fall-2026-pyt
export PYTHONUSERBASE=/projectnb/herbdl/workspaces/faridkar/.local-fall2026-pyt

cd "$(dirname "$0")" || exit 1
echo "host=$(hostname)  cwd=$(pwd)  python=$(command -v python)  TMPDIR=${TMPDIR:-unset}"
python -c "import pandas, numpy, pyarrow; print('pandas', pandas.__version__, 'numpy', numpy.__version__, 'pyarrow', pyarrow.__version__)" || exit 1
python rundown.py
