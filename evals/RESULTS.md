# Fixture retrieval eval results

Offline harness using `FakeEmbeddings` (no live OpenAI calls).

- Command: `python evals/run_eval.py`
- k = 4
- Questions = 4
- Mean precision@4 = **0.3333**
- Mean recall@4 = **1.0**
- Faithfulness pass rate = **1.0** (5 checks, including empty-context refusal)

## Per question

- `q_acme_founded`: P@4=0.3333, R@4=1.0, faithful=True; retrieved=['helios.txt', 'acme.txt', 'northwind.txt']
- `q_northwind_skus`: P@4=0.3333, R@4=1.0, faithful=True; retrieved=['helios.txt', 'acme.txt', 'northwind.txt']
- `q_helios_site`: P@4=0.3333, R@4=1.0, faithful=True; retrieved=['northwind.txt', 'acme.txt', 'helios.txt']
- `q_helios_cadence`: P@4=0.3333, R@4=1.0, faithful=True; retrieved=['helios.txt', 'northwind.txt', 'acme.txt']
