# ExploreKit

ExploreKit — командный прототип управляемого exploration для рекомендаций
с офлайн-оценкой политик (OPE). Бандиты крутят exploration поверх базового
ранкера, симулятор знает скрытый P(click), OPE оценивает политику по логам
с propensity, dashboard показывает сохранённые прогоны.

Жадная выдача зацикливается на уже проверенных айтемах (feedback loop).
Новинки почти не получают показов, статистика по ним не копится. Exploration
даёт шанс слабым по текущему скору рукам; OPE позволяет сравнить политики
по историческим логам, не включая каждую в прод. Сценарий раскрутки новинок —
Cold-Item Booster: молодые айтемы получают дополнительный шанс показа.

Отчёт: [report/report.html](report/report.html). Графики лежат в
`report/figures/`.

## Кто за что отвечает

Четыре зоны стыкуются через общий `Decision` и YAML-конфиг.

1. Бандиты (`explorekit/policies/`) — ε-greedy, LinUCB, Thompson Sampling,
   `Decision` с propensity, фабрика политик.
2. OPE (`explorekit/ope/`) — IPS, SNIPS, Doubly Robust, bootstrap CI, ESS,
   reliability.
3. Симулятор и Cold-Item Booster (`explorekit/simulator/`,
   `explorekit/cold_start/`) — скрытый CTR, волны новинок, метрики speedup
   и CTR cost.
4. Интеграция (`explorekit/logging/`, `explorekit/integration/`,
   `explorekit/results/`, `experiments/run_experiment.py`, `dashboard/`) —
   логи, runner, Parquet, dashboard.

Цепочка одного прогона:

```text
simulator → bandit.select_action / action_distribution
         → DecisionLog (propensity)
         → OPE (IPS / SNIPS / DR + CI, ESS, reliability)
         → ExperimentResult → dashboard
```

Политика создаётся фабрикой `build_active_policy`. Integration не копирует
реализации бандитов и OPE: логи уходят в `explorekit.ope.build_ope_input`,
результат — в `ExperimentResult`.

## Быстрый старт

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q
```

Четыре пресета, seed и `n_rounds` берутся из YAML (`seed: 42`, `n_rounds: 5000`):

```bash
python experiments/run_experiment.py --config configs/baseline.yaml
python experiments/run_experiment.py --config configs/conservative.yaml
python experiments/run_experiment.py --config configs/moderate.yaml
python experiments/run_experiment.py --config configs/aggressive.yaml
```

Главная команда из ТЗ — moderate:

```bash
python experiments/run_experiment.py --config configs/moderate.yaml
```

Она печатает True CTR, Expected CTR, IPS, SNIPS, DR, 95% CI, ESS,
reliability, exploration share, cold-item speedup и CTR cost. Результат
пишется в `outputs/results/` (каталог в git не коммитится) и подхватывается
dashboard.

Сводная таблица cold-start по тем же четырём YAML:

```bash
python experiments/cold_item_experiment.py
python experiments/cold_item_experiment.py --n-rounds 800
```

Dashboard и Docker:

```bash
streamlit run dashboard/app.py
```

```bash
docker build -t explorekit .
docker run --rm -p 8501:8501 explorekit
docker run --rm explorekit \
  python experiments/run_experiment.py --config configs/moderate.yaml
```

Smoke интеграции:

```bash
python scripts/smoke_integration.py --n-rounds 1000
```

## Контракт Decision

Канонический `Decision` в `explorekit.policies.base`:

```python
Decision(
    chosen_item: int,
    propensity: float,
    probabilities: np.ndarray,
    is_exploration: bool,
    policy_name: str,
)
```

`chosen_item` — int-индекс в `[0, n_actions)`. Строковые `item_id`, если они
есть, мапятся в индекс до вызова политики (`explorekit.integration.ActionIndexMap`).

`DecisionLog` содержит timestamp, request_id, user_segment, context,
candidate_items, base_scores, chosen_item, propensity, probabilities,
policy_name, policy_params, reward, is_exploration, is_cold_item.

Validator проверяет `0 < propensity <= 1`, `chosen_item` входит в
`candidate_items`, `propensity == probabilities[chosen_item]`,
`sum(probabilities) == 1`, нет NaN/inf, `reward in {0, 1}`.
`candidate_items` может быть подмножеством action space: длина не обязана
совпадать с `probabilities`; `base_scores` и `probabilities` длины `n_actions`.

## Бандиты

Пресеты в `configs/*.yaml` включают `epsilon_greedy`. В
`configs/policies.yaml` также есть LinUCB и Thompson Sampling.

- ε-greedy: с вероятностью `1 − ε` берёт argmax base score, с вероятностью
  `ε` — равномерно по кандидатам (или по `exploration_subset`, если booster
  его задал). Propensity = вероятность выбранной руки.
- LinUCB: линейный UCB по контексту; `action_distribution` даёт реальный
  propensity выбранного действия, не эвристику.
- Thompson Sampling: сэмплы θ, propensity ≈ доля побед руки в MC-сэмплах.

Интерфейс: `select_action`, `action_distribution`, `update`. Для OPE
нужен `action_distribution(context, base_scores, candidate_items)`.

Для baseline в YAML стоит `min_epsilon: 0.0`. Без override боевой
`configs/policies.yaml` держит минимум 0.01, и «жадный» пресет превратился
бы в 1% exploration.

## OPE

IPS, SNIPS и Doubly Robust читают логи с propensity логирующей политики и
распределение оцениваемой. Клиппинг весов — `max_weight: 15` в YAML.
95% CI — bootstrap. ESS = (∑w)² / ∑w²; статус Reliable / Use with caution /
Unreliable смотрит ESS/N, долю обрезанных весов и common support.

На синтетике истина известна: `make_replication` возвращает `true_value`.
Прогон `experiments/ope_synthetic.py` (12 независимых репликаций, n = 2000,
seed 42, ε = 0.10) дал bias около +0.001 и CI coverage 100% у IPS, SNIPS и
DR. В `configs/ope.yaml` заложен полный чек на 50 репликаций × 5000 раундов;
эти 50×5000 в отчёт не подставлялись.

Open Bandit Dataset в репозиторий не входит.
Загрузчик — `explorekit/datasets/open_bandit.py`, эксперимент —
`experiments/open_bandit.py`. Скачать опубликованные CSV:

```bash
bash scripts/download_obd.sh
```

Источник: [Open Bandit Dataset](https://github.com/st-tech/zr-obp)
([Saito et al., 2021](https://arxiv.org/abs/2008.07146)). Для полного
объёма укажите каталог в YAML, код менять не нужно:

```yaml
open_bandit:
  data_dir: /path/to/full/obd
```

## Симулятор и Cold-Item Booster

Синтетическая среда в `explorekit/simulator/`. Политики не видят настоящую
вероятность клика: `P(click | user, item) = sigmoid(u·i + affinity + intercept)`.
Base ranker получает зашумлённую оценку и занижает скоры молодых айтемов.
80 айтемов, 8 признаков контекста, 4 сегмента пользователей, 5 категорий.
День 0 — тёплое ядро, день 5 и день 10 — волны новых айтемов. True CTR
жадной политики калибруется около 5% (`target_ctr: 0.05`).

```python
from explorekit.simulator.environment import make_replication

rep = make_replication(seed)
rep.ope_input   # reward, behavior/evaluation propensity, q_hat для DR
rep.true_value  # ожидаемый CTR оцениваемой политики без booster
```

В пресетах:

```yaml
simulation:
  provider: explorekit.simulator.environment
  logging_epsilon: 0.5
  cold_penalty: 1.2
  target_ctr: 0.05
```

Cold-Item Booster (`explorekit/cold_start/booster.py`) помечает айтем как
cold, если возраст `< max_age_days` или показов `< min_impressions`. Cold
items получают сдвиг base score на `log(boost_multiplier)` и становятся
`exploration_subset` для ε-greedy. При ε = 0 booster выключен: baseline —
чистое exploitation.

Два стека, это не баг:

- True CTR и OPE — стационарная оцениваемая политика, `exploration_subset=None`,
  без booster. Это то, что должен восстановить офлайн-оценщик.
- speedup и CTR cost — serving с booster (кроме baseline). CTR cost в
  процентных пунктах относительно greedy без booster.

Логи пишет ε-greedy с `logging_epsilon: 0.5`, чтобы ESS не падал на 80 руках.
time-to-N цензурируется горизонтом прогона; speedup — среднее по айтемам
отношений времён, не отношение средних.

Цифры ниже — реальные прогоны `configs/{baseline,conservative,moderate,aggressive}.yaml`,
seed 42, n_rounds = 5000. CTR в тексте в процентах; внутри системы на `[0, 1]`.
CTR cost в п.п. Speedup на этих пресетах 1.28–1.91x, не 4x.

| preset | ε | True CTR | CTR cost | speedup | OPE | ESS |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| baseline | 0.00 | 5.10% | 0.00 pp | 1.00x | Reliable | 2531 |
| conservative | 0.05 | 4.86% | 0.18 pp | 1.28x | Reliable | 2793 |
| moderate | 0.10 | 4.63% | 0.42 pp | 1.43x | Reliable | 3079 |
| aggressive | 0.25 | 3.94% | 1.10 pp | 1.91x | Reliable | 4021 |

Графики: `report/figures/exploration_vs_ctr.png`,
`report/figures/exploration_vs_speedup.png`,
`report/figures/true_vs_ope.png`. Перерисовать:

```bash
python experiments/plot_zone3.py
```

### Прогон на Open Bandit Dataset

Датасет: 1,374,327 показов, 80 действий, CTR логирующей
политики 0.347% (4,768 кликов).

| policy | ε | IPS | SNIPS | DR | ESS/N | status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| conservative | 0.05 | 0.085% | 0.352% | 0.351% | 1.4% | Unreliable |
| moderate | 0.10 | 0.102% | 0.351% | 0.351% | 1.6% | Unreliable |
| aggressive | 0.25 | 0.153% | 0.350% | 0.350% | 2.2% | Unreliable |

IPS-оценки ненадёжны (ESS/N = 1.4–2.2%, max weight 60–76, обрезано 1.3%) —
следствие крайне низкого CTR (0.347%) и слабого перекрытия политик.
SNIPS и DR дают стабильные оценки ~0.35%, но не различают политики:
reward model на `LogisticRegression` не улавливает различия между
айтемами при таком CTR. Диагностика корректно помечает все оценки как
`Unreliable` — это ожидаемое поведение OPE при малом перекрытии.

## ExperimentResult и dashboard

Каждый run фиксирует logging/evaluation policy, N, seed, True CTR,
IPS/SNIPS/DR, 95% CI, ESS, reliability, exploration share, cold-item
speedup, CTR cost.

```bash
streamlit run dashboard/app.py
```

Dashboard читает только `outputs/results/*.json`: Overview, Exploration vs
CTR, Exploration vs cold-item speedup, IPS / SNIPS / DR / ESS / Reliability.
Числа не захардкожены. Пока прогонов нет, покажет подсказку запустить
`run_experiment`.

Если simulator вернул `logs`, runner пишет их в
`outputs/logs/date=YYYY-MM-DD/part-....parquet` (нужен `pyarrow`).

## Тесты и воспроизводимость

```bash
python -m pytest -q
ruff check .
black --check .
```

Python 3.11+ (`requires-python` в `pyproject.toml`; Docker — 3.11).
Случайность только через `np.random.default_rng(seed)`. Seed и остальные
параметры — в YAML, не в коде. `outputs/`, `data/obd/` и `.venv` в git
не коммитятся. Сверка формул OPE с пакетом `obp` пропускается, если пакет
не установлен.

## Структура

```text
explorekit/
├── explorekit/
│   ├── policies/
│   ├── ope/
│   ├── simulator/
│   ├── cold_start/
│   ├── datasets/
│   ├── logging/
│   ├── integration/
│   └── results/
├── configs/
├── dashboard/
├── experiments/
├── notebooks/
├── scripts/
├── tests/
├── report/                 # report.html и figures/
├── README.md
├── requirements.txt
├── pyproject.toml
└── Dockerfile
```

## Скринкаст демонстрации

[Смотреть на Яндекс.Диске](https://disk.yandex.ru/d/WxWf2lBwUdACvw) — 5 минут