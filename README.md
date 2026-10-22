# AutoGluon vs FEDOT

Сравнение двух AutoML-фреймворков на табличных задачах при одинаковом бюджете времени:

- **AutoGluon** обучает набор моделей (градиентный бустинг, случайный лес, нейросети и др.) и собирает их в стекинг-ансамбль;
- **FEDOT** эволюционно подбирает структуру композитного пайплайна в виде графа.

Сравниваются качество, время и устойчивость результата. Подробности постановки и выводы: [REPORT.md](REPORT.md).

## Результаты

Бюджет 120 с на задачу и фреймворк, `seed=42`, hold-out 20%.

| Задача | Метрика | AutoGluon | FEDOT |
|---|---|---|---|
| Классификация (Breast Cancer) | Accuracy | 0.956 | 0.956 |
| | F1 | 0.966 | 0.966 |
| | ROC-AUC | 0.994 | 0.995 |
| | время, с | 8 | 123 |
| Регрессия (California Housing) | MAE | **0.275** | 0.533 |
| | RMSE | **0.428** | 0.745 |
| | время, с | 121 | 120 |

![Сравнение](results/reference_run/comparison_results.png)

Это один прогон на одном seed: разницу в классификации (0.994 против 0.995 по ROC-AUC) нельзя считать значимой.
Значения в таблице сняты с графика эталонного прогона, см. [results/reference_run](results/reference_run).

## Структура

```
automl_benchmark/    код эксперимента (python -m automl_benchmark)
  config.py          параметры: бюджет, seed, наборы операций FEDOT
  datasets.py        загрузка данных и сплиты
  runners.py         запуск AutoGluon и FEDOT
  metrics.py         метрики и выбор лучшего
  plots.py           итоговый график
notebooks/           исследовательский ноутбук: теория AutoML, три задачи, временные ряды
results/             результаты прогонов
tests/               тесты метрик, сплитов и построения графика
REPORT.md            отчёт
```

## Запуск

FEDOT поддерживает не самые свежие версии Python: если установка падает, используйте Python 3.10.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m automl_benchmark --time-limit 120 --seed 42 --output results/latest
```

Результаты (`results.json` и `comparison_results.png`) появятся в `results/latest`.
Один полный прогон занимает около 8 минут: по 120 с на каждую из двух задач для каждого фреймворка.

Ноутбук из `notebooks/` рассчитан на Google Colab (устанавливает зависимости первой ячейкой);
дополнительно к двум задачам из скрипта он содержит классификацию Adult Income и прогноз временного ряда AirPassengers.
Сохранённых выходов в ноутбуке нет, их нужно получить запуском.

## Тесты

```bash
pip install -r requirements-dev.txt
pytest
```

Тесты не требуют установки AutoGluon и FEDOT.
