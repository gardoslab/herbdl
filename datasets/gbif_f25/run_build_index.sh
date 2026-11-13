#!/bin/bash -l
# Job 1: GBIF-F25 species index snapshot (CPU only). Same shared env as scaling/run_scaling.sh.
# No `set -e` around module loads: SCC's miniconda profile returns non-zero (see run_scaling.sh).
module load miniconda
module load academic-ml/fall-2026
conda activate fall-2026-pyt
export PYTHONUSERBASE=/projectnb/herbdl/workspaces/faridkar/.local-fall2026-pyt

cd "$(dirname "$0")" || exit 1
echo "host=$(hostname)  cwd=$(pwd)  python=$(command -v python)  TMPDIR=${TMPDIR:-unset}"
python -c "import pandas, numpy, pyarrow; print('pandas', pandas.__version__, 'numpy', numpy.__version__, 'pyarrow', pyarrow.__version__)" || exit 1
python build_index.py
