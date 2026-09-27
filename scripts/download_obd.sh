#!/usr/bin/env bash

set -euo pipefail

TARGET="${1:-data/obd}"
BASE="https://raw.githubusercontent.com/st-tech/zr-obp/master/obd"

echo "Загрузка Open Bandit Dataset в ${TARGET}"

for policy in random bts; do
  for campaign in all men women; do
    dir="${TARGET}/${policy}/${campaign}"
    mkdir -p "${dir}"
    for file in "${campaign}.csv" "item_context.csv"; do
      url="${BASE}/${policy}/${campaign}/${file}"
      echo "  ${policy}/${campaign}/${file}"
      curl -fsSL -o "${dir}/${file}" "${url}"
    done
  done
done

echo
echo "Готово. Проверка:"
python3 - "$TARGET" << 'PY'
import sys, pathlib, csv
target = pathlib.Path(sys.argv[1])
path = target / "random" / "all" / "all.csv"
with path.open(encoding="utf-8") as handle:
    header = next(csv.reader(handle))
    rows = sum(1 for _ in handle)
required = {"item_id", "position", "click", "propensity_score"}
missing = required - set(header)
if missing:
    raise SystemExit(f"ОШИБКА: в {path} нет колонок {sorted(missing)}")
print(f"  {path}: {rows} строк, обязательные колонки на месте")
PY
