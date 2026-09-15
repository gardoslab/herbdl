# SCC GPU queues — eligibility, availability, and wall-time caps

A practical guide to answering, for a **given job configuration and project**:
*which nodes can run it, which queues I'm allowed to use, and the longest wall time I can request* —
and why a job can sit in `qw` while GPUs sit idle.

> SCC has scheduler job-info collection **off**, so `qstat -j` won't tell you *why* a job waits
> or its queue position. You infer it from the data below.

---

## TL;DR for project `herbdl`

The free GPUs you see are mostly **buy-in** nodes owned by other groups, and `herbdl`'s access
to them is **time-capped**. What `herbdl` can do with a single-node `cc≥8.0 / 80G` GPU job:

| Where | herbdl access | max `h_rt` | Notes |
|-------|---------------|-----------|-------|
| Shared `a100` / `h200` / `l40s` … queues | yes | **48h** | The only way to request 48h — but these pools are **scarce/usually full** |
| `cds-gpu-long` (CDS buy-in) | yes | **24h** | Often the fastest real option — CDS nodes (e.g. `scc-a16`) are frequently idle |
| `cds-gpu`, public `*-pub` queues | yes | **12h** | Widest access to *idle* buy-in GPUs across the cluster |
| `ece*`, `chapmangroup*`, other groups' own queues | **no** | — | Idle but unusable by `herbdl` |

**Consequence:** a **48h** request is eligible *only* for the scarce shared `a100`/`h200`
queues, so it queues behind them even while CDS/public GPUs are idle. **Request ≤24h** (or
≤12h for the widest access) and lean on per-epoch checkpoints + auto-resume across jobs.
`finetuning/SWIN/submit_concrete.sh` defaults `H_RT=24:00:00` for this reason.

---

## The helper: feedback for a job config + project

[`scc_gpu_check.sh`](scc_gpu_check.sh) consolidates the whole analysis. Run it on a login node:

```bash
# defaults: PROJECT=herbdl NEED=2 (gpus/node) MINCC=8.0 MINMEM=80 (GB)
bash docs/scc_gpu_check.sh

# any job config / project:
PROJECT=herbdl NEED=4 MINCC=8.0 MINMEM=80 bash docs/scc_gpu_check.sh
```

It prints four things:
1. **GPU queues the project may use** and each one's wall-time cap.
2. **The longest allowed wall time** (max cap across eligible queues).
3. **Nodes with ≥`NEED` free matching GPUs right now** (free *on one host* — what a single-node
   `-pe omp` job needs).
4. **Where you can actually run right now** — each free node × the eligible queues on it × cap.

### Example output (`herbdl`, 2 GPUs, cc≥8.0, 80G)

```
### Job: project=herbdl  gpus=2 (single node)  cc>=8.0  gpu_mem>=80G

## GPU queues herbdl may use (wall-time cap):
   a100                     h_rt<=48:00:00
   h200                     h_rt<=48:00:00
   cds-gpu-long             h_rt<=24:00:00
   cds-gpu                  h_rt<=12:00:00
   cds-gpu-pub              h_rt<=12:00:00
   academic-gpu-pub         h_rt<=12:00:00
   ... (many *-pub queues at 12h)
   -> LONGEST allowed wall time: 48h (via a100)

## Nodes with >=2 free matching GPUs right now:
   scc-221    free=2 cc=8 mem=80G
   scc-a06    free=3 cc=9 mem=144G
   scc-a16    free=2 cc=9 mem=144G

## Where you can actually run right now (free node -> eligible queue -> cap):
   scc-221    -> chapmangroup-gpu-pub   (h_rt<=12:00:00)
   scc-a06    -> ece-pub                (h_rt<=12:00:00)
   scc-a16    -> cds-gpu                (h_rt<=12:00:00)
   scc-a16    -> cds-gpu-long           (h_rt<=24:00:00)   <- best: 24h on an idle CDS node
   scc-a16    -> cds-gpu-pub            (h_rt<=12:00:00)
```

Reading it: the "longest allowed" is **48h** (shared `a100`/`h200`) — but those are scarce.
Right now the only idle nodes are buy-in; the best *usable* slot is **`cds-gpu-long` on
`scc-a16` at ≤24h**. So a 24h job lands immediately; a 48h job waits for the shared pool.

---

## Manual commands (what the helper wraps)

```bash
# Your jobs + state (qw = waiting, r = running) and the request:
qstat -u $USER
qstat -j <JOBID>                 # hard resource_list, project, pe

# Cluster-wide free GPUs by type:
qgpus                            # A100-80G, H200, L40S, ...

# Nodes with >=N free GPUs matching cc/memory (free ON ONE host):
qhost -F gpu_compute_capability,gpu_memory,gpus | awk '
  /^scc-/{h=$1;cc=m=f="";next}
  /compute_capability=/{s=$0;sub(/.*=/,"",s);cc=s+0}
  /gpu_memory=/{s=$0;sub(/.*=/,"",s);m=s+0}
  /hc:gpus=/{s=$0;sub(/.*=/,"",s);f=s+0; if(cc>=8 && m>=80 && f>=2) printf "%-12s cc=%s mem=%dG free=%d\n",h,cc,m,f}'
#   knobs: cc>=8 (8.0=A100, 9.0=H200), m>=80 (GB), f>=N (gpus free on one node)

# Queues on a node + their state:
qhost -q -h <NODE>

# A queue's project access + wall-time cap (lists wrap with '\'):
qconf -sq <QUEUE> | grep -E 'qname|projects|h_rt|pe_list'
```

`hc:gpus` is the **available** (free) count on a host, so a node only shows up if it can satisfy
your single-node request *right now*. `projects NONE` = open to everyone; otherwise the project
must be in the list. `BIP`/`BP`/`IP` in `qhost -q` are queue types (buy-in/batch/interactive).

---

## Why a job sits in `qw` even though GPUs are free

A single-node DDP job (`-pe omp`) needs **all its GPUs free on one host**, in a **queue your
project may use**, whose **`h_rt` cap ≥ your request**. A job waits if *any* of these fail:

1. **Fragmentation** — GPUs are free, but not `NEED` of them on the *same* node
   (e.g. 4-GPU job when nodes have only 0–2 free each).
2. **Ownership** — the free GPUs are another group's buy-in nodes (`ece*`, `chapmangroup*`);
   your project isn't in their `projects` list.
3. **Wall-time cap** — the only queues you *can* borrow (`*-pub`, `cds-gpu*`) cap at 12–24h,
   so a longer request is ineligible and waits for the scarce shared long-GPU pool.
4. **Other resources** — not enough free CPU slots (`-pe omp N`) or host memory on the node.

For long training, the fix is almost always **request a shorter wall time** (≤24h, often ≤12h
for widest access) and **resume across jobs** from per-epoch checkpoints, rather than waiting
days for a 48h slot on the shared pool.
