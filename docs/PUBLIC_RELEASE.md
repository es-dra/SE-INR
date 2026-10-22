# Publish a clean snapshot

Run the following commands from the source repository after the manuscript and compact result files are ready for release. They export committed `main` only; uncommitted changes, Git history and tags are not exported. The source repository, its history and its `archive-2026-10` tag are unchanged.

Choose a new, empty destination outside the source repository. `mkdir` intentionally fails if that destination already exists. Review the exported files for private information, data redistribution restrictions and omitted external dependencies before making the new repository public.

```bash
set -euo pipefail
release_dir=/absolute/path/to/SE-INR-public
source_commit=$(git rev-parse --verify main)
mkdir "$release_dir"
git archive --format=tar "$source_commit" | tar -xf - -C "$release_dir"
git -C "$release_dir" init -b main
git -C "$release_dir" add --all
git -C "$release_dir" -c user.name='Release Maintainer' -c user.email='release@example.invalid' \
  commit -m "Public source snapshot ($source_commit)"

# These checks must print 1 and no tags, respectively.
git -C "$release_dir" rev-list --count HEAD
git -C "$release_dir" tag --list
```

Use an appropriate public author identity in the commit command. The sample identity avoids copying the source repository's personal Git identity. No remote is added and nothing is pushed automatically. After reviewing the snapshot, create a new empty remote repository and publish it separately; do not force-push this snapshot over the source repository.

Files over 1 MB and trained weights remain external. Their recorded paths and SHA256 values identify the corresponding artifacts; they do not themselves grant access or redistribution rights. Preserve licence notices, frozen protocols, scoring code, compact results and failed predictions in the release.
