#!/bin/bash

echo "Analyzing largest historical blobs (fast mode)..."

git rev-list --objects --all \
  | git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' \
  | grep '^blob ' \
  | sort -k3 -n \
  | tail -n 20 \
  | awk '{ printf "%12s bytes   %s\n", $3, $4 }'

echo "Done."
