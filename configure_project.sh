#!/bin/bash

# Argument parsing for --dry-run, --backup, and --restore
DRY_RUN=0
BACKUP=0
RESTORE=0
POSITIONAL=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --backup)
      BACKUP=1
      shift
      ;;
    --restore)
      RESTORE=1
      shift
      ;;
    *)
      POSITIONAL+=("$1")
      shift
      ;;
  esac
done
set -- "${POSITIONAL[@]}"


# --restore logic
if [ $RESTORE -eq 1 ]; then
  restored=0
  for f in pyproject.toml README.md environment.yml Makefile; do
    if [ -f "$f.bak" ]; then
      cp "$f.bak" "$f"
      echo "Restored $f from $f.bak"
      rm -f "$f.bak"
      restored=1
    fi
  done
  if [ $restored -eq 0 ]; then
    echo "No backup files found to restore."
  fi
  exit 0
fi

if [ -z "$1" ]; then
  echo "Usage: $0 [--dry-run] [--backup] [--restore] <new_project_name>"
  exit 1
fi

NEW_NAME="$1"
NEW_NAME_SNAKE=$(echo "$NEW_NAME" | tr '-' '_')
NEW_NAME_KEBAB=$(echo "$NEW_NAME" | tr '_' '-')



# Backup function
backup_file() {
  local f="$1"
  if [ "$BACKUP" -eq 1 ] && [ -f "$f" ]; then
    cp "$f" "$f.bak"
  fi
}

# DRY RUN function for awk
awk_dry_run() {
  local newname="$1"
  local file="$2"
  echo "[DRY-RUN] Would update 'name' in [project] section of $file to '$newname'"
  awk -v newname="$newname" '
    BEGIN { in_project=0 }
    /^\[project\]/ { in_project=1; print; next }
    /^\[/ { in_project=0 }
    in_project && /^name[ ]*=/ {
      print "name = \"" newname "\""; next
    }
    { print }
  ' "$file" | diff -u "$file" - || true
}

# Robustly replace 'name' field only in [project] section of pyproject.toml
if [ "$DRY_RUN" -eq 1 ]; then
  awk_dry_run "$NEW_NAME_KEBAB" pyproject.toml
else
  backup_file pyproject.toml
  awk -v newname="$NEW_NAME_KEBAB" '
    BEGIN { in_project=0 }
    /^\[project\]/ { in_project=1; print; next }
    /^\[/ { in_project=0 }
    in_project && /^name[ ]*=/ {
      print "name = \"" newname "\""; next
    }
    { print }
  ' pyproject.toml > pyproject.toml.tmp && mv pyproject.toml.tmp pyproject.toml
fi



# Replace in README.md (title and intro) if present
if [ -f README.md ]; then
  if [ "$BACKUP" -eq 1 ]; then backup_file README.md; fi
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "[DRY-RUN] Would replace template[-_]basic with $NEW_NAME_KEBAB and $NEW_NAME_SNAKE in README.md"
    grep -E -i 'template[-_]basic' README.md | cat
  else
    sed -i "s/template[-_]basic/$NEW_NAME_KEBAB/gI" README.md || { echo "Error updating README.md (kebab)"; exit 1; }
    sed -i "s/template[-_]basic/$NEW_NAME_SNAKE/gI" README.md || { echo "Error updating README.md (snake)"; exit 1; }
  fi
fi



# Replace in environment.yml (if present, robust to whitespace)
if [ -f environment.yml ]; then
  if [ "$BACKUP" -eq 1 ]; then backup_file environment.yml; fi
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "[DRY-RUN] Would replace environment name in environment.yml with $NEW_NAME_KEBAB"
    grep -E -i '^([[:space:]]*name:[[:space:]]*)template[-_]basic' environment.yml | cat
  else
    sed -i -E "s/^([[:space:]]*name:[[:space:]]*)template[-_]basic/\1$NEW_NAME_KEBAB/gI" environment.yml || { echo "Error updating environment.yml"; exit 1; }
  fi
fi

# Replace environment name in Makefile if present
if [ -f Makefile ]; then
  if [ "$BACKUP" -eq 1 ]; then backup_file Makefile; fi
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "[DRY-RUN] Would replace environment name in Makefile with $NEW_NAME_KEBAB"
    grep -E 'mamba (env create|activate)[^\\n]*template[-_]basic' Makefile | cat
  else
    sed -i -E "s/(--name[[:space:]]*)template[-_]basic/\1$NEW_NAME_KEBAB/gI" Makefile || { echo "Error updating Makefile (env name)"; exit 1; }
    sed -i -E "s/(mamba[[:space:]]+activate[[:space:]]*)template[-_]basic/\1$NEW_NAME_KEBAB/gI" Makefile || { echo "Error updating Makefile (activate)"; exit 1; }
  fi
fi

# Rename src/template_basic to src/$NEW_NAME_SNAKE if needed, check for destination
if [ -d "src/template_basic" ] && [ "$NEW_NAME_SNAKE" != "template_basic" ]; then
  if [ -e "src/$NEW_NAME_SNAKE" ]; then
    echo "Destination directory src/$NEW_NAME_SNAKE already exists. Aborting rename." >&2
    exit 1
  fi
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "[DRY-RUN] Would rename src/template_basic to src/$NEW_NAME_SNAKE"
  else
    mv "src/template_basic" "src/$NEW_NAME_SNAKE" || { echo "Error renaming src/template_basic"; exit 1; }
  fi
fi



# Update imports in tests and src
if [ -d src ] || [ -d tests ]; then
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "[DRY-RUN] Would update imports in Python files in src/ and tests/"
    find src tests -type f -name "*.py" -exec grep -Hn template_basic {} +
  else
    find src tests -type f -name "*.py" -exec sed -i "s/template_basic/$NEW_NAME_SNAKE/g" {} + || { echo "Error updating imports in Python files"; exit 1; }
  fi
fi

if [ "$DRY_RUN" -eq 1 ]; then
  echo "[DRY-RUN] No files were modified."
else
  echo "Project name updated to $NEW_NAME"
fi
