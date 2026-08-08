#!/bin/sh


PARTITION="$1"
GROUP_SIZE="$2"
SCRIPT="$3"

for taskID in $(seq 0 $((GROUP_SIZE - 1))); do
	groupID=$taskID
	str="taskID=${taskID}_SCRIPT=$(basename "$SCRIPT" .py)"
	sbatch -o "${str}.out" -p "$PARTITION" run.sh "$groupID" "$GROUP_SIZE" "$SCRIPT"
done


