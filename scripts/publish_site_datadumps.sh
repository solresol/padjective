#!/usr/bin/env bash
set -euo pipefail

source_dir=${1:?expected the built datadumps directory}
canonical=${2:?expected the canonical rsync destination}
legacy=${3:?expected the compatibility rsync destination}
[[ -d "$source_dir" && ! -L "$source_dir" ]] || {
    echo "No built SQL dumps; leaving published dumps unchanged."
    exit 0
}

shopt -s dotglob nullglob
files=("$source_dir"/*)
if (( ${#files[@]} == 0 )); then
    echo "No built SQL dumps; leaving published dumps unchanged."
    exit 0
fi

links=$(mktemp -d)
trap 'rm -rf -- "$links"' EXIT
for file in "${files[@]}"; do
    name=${file##*/}
    if [[ ! -f "$file" || -L "$file" || ! "$name" =~ ^[a-z0-9_][a-z0-9_.-]*\.sql$ ]]; then
        echo "Unexpected data dump entry: $name" >&2
        exit 1
    fi
    # This relative target works both in OpenBSD httpd's chroot and on disk.
    ln -s -- "../../../datadumps.ifost.org.au/htdocs/padjective/$name" "$links/$name"
done

# Publish bytes first and retain the canonical site's historical exports.
rsync -az -- "$source_dir/" "$canonical"
# Do not copy the private staging directory's mode onto the public directory.
# Keep the book site's original URL and directory index without another copy.
rsync -rlt --delete -- "$links/" "$legacy"
