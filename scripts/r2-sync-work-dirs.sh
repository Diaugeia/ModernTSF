#!/bin/zsh
set -euo pipefail

bucket="${R2_BUCKET:-moderntsf-artifacts}"
source_root="${1:-work_dirs}"
prefix="${R2_PREFIX:-work_dirs}"
parallelism="${R2_PARALLELISM:-4}"

if [[ ! -d "$source_root" ]]; then
  print -u2 "Source directory does not exist: $source_root"
  exit 1
fi

source_root="${source_root:A}"
export bucket source_root prefix

find "$source_root" -type f -print0 | xargs -0 -n 1 -P "$parallelism" /bin/zsh -c '
  file="$1"
  relative="${file#$source_root/}"
  print "Uploading $relative"
  wrangler r2 object put "$bucket/$prefix/$relative" --file "$file" --remote
' _

print "Upload complete: r2://$bucket/$prefix/"
