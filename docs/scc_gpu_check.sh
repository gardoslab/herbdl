#!/bin/bash
# Report, for a project + single-node GPU job config: usable queues, longest allowed
# wall time, and nodes with enough free GPUs right now.
PROJECT="${PROJECT:-herbdl}"; NEED="${NEED:-2}"; MINCC="${MINCC:-8.0}"; MINMEM="${MINMEM:-80}"
echo "### Job: project=$PROJECT  gpus=$NEED (single node)  cc>=$MINCC  gpu_mem>=${MINMEM}G"; echo

# GPU hosts (have a gpus complex)
gpu_hosts=" $(qhost -F gpus 2>/dev/null | awk '/^scc-/{h=$1} /:gpus=/{print h}' | sort -u | tr '\n' ' ') "
# GPU queues = queue instances living on GPU hosts (one qhost -q call)
gpu_queues=$(qhost -q 2>/dev/null | awk -v gh="$gpu_hosts" '
  /^scc-/{h=$1; isg=(index(gh," "h" ")>0); next}
  isg && /BIP|BP|IP/ {print $1}' | sort -u)

elig_of(){ # $1 queue -> "yes|no HRT"
  local flat hrt projects
  flat=$(qconf -sq "$1" 2>/dev/null | sed ':a;/\\$/{N;s/\\\n//;ta}')
  hrt=$(echo "$flat" | awk '/^h_rt /{print $2}')
  projects=$(echo "$flat" | awk '/^projects/{$1="";print;exit}')
  if echo " $projects " | grep -qwE "NONE|$PROJECT"; then echo "yes $hrt"; else echo "no $hrt"; fi
}

echo "## GPU queues $PROJECT can use (and their wall-time cap):"
best=0; bestq=""
while read -r q; do
  [ -z "$q" ] && continue
  read -r elig hrt < <(elig_of "$q")
  [ "$elig" = yes ] || continue
  printf "   %-24s h_rt<=%s\n" "$q" "$hrt"
  s=$(echo "$hrt"|awk -F: '{print $1*3600+$2*60+$3}'); [ "$s" -gt "$best" ] && { best=$s; bestq=$q; }
done <<< "$gpu_queues"
echo "   -> LONGEST allowed wall time: $((best/3600))h (via $bestq)"; echo

echo "## Nodes with >=$NEED free GPUs matching cc/mem right now:"
qhost -F gpu_compute_capability,gpu_memory,gpus 2>/dev/null | awk -v cc=$MINCC -v mem=$MINMEM -v need=$NEED '
  /^scc-/{h=$1;c=m=f="";next}
  /compute_capability=/{s=$0;sub(/.*=/,"",s);c=s+0}
  /gpu_memory=/{s=$0;sub(/.*=/,"",s);m=s+0}
  /hc:gpus=/{s=$0;sub(/.*=/,"",s);f=s+0; if(c>=cc&&m>=mem&&f>=need) printf "   %-10s free=%d cc=%s mem=%dG\n",h,f,c,m}'

echo; echo "## Where you can actually run right now (free node x eligible queue x cap):"
qhost -F gpu_compute_capability,gpu_memory,gpus 2>/dev/null | awk -v cc=$MINCC -v mem=$MINMEM -v need=$NEED '
  /^scc-/{h=$1;c=m=f="";next}
  /compute_capability=/{s=$0;sub(/.*=/,"",s);c=s+0}
  /gpu_memory=/{s=$0;sub(/.*=/,"",s);m=s+0}
  /hc:gpus=/{s=$0;sub(/.*=/,"",s);f=s+0; if(c>=cc&&m>=mem&&f>=need) print h}' | while read -r node; do
  for q in $(qhost -q -h "$node" 2>/dev/null | awk 'NR>3 && /BIP|BP|IP/{print $1}'); do
    read -r elig hrt < <(elig_of "$q")
    [ "$elig" = yes ] && printf "   %-10s -> %-22s (h_rt<=%s)\n" "$node" "$q" "$hrt"
  done
done
