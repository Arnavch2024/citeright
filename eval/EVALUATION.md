# Evaluation

Mode: **offline (all numbers synthetic)**. Offline figures are synthetic; live latency and cost are estimates.
Citations are valid when every cited `source_id`/`url` is in `sources[]` (retrieved set).
Cost is a rough USD estimate from token counts, not a bill.

| query | variant | schema | citations | behavior checks | tool path | verified ratio | claims dropped | conflicts | confidence | latency | approx cost |
|---|---|---|---|---|---|---:|---:|---:|---|---:|---:|
| vector_dbs | full agent | valid | valid | schema; citations; tools:['web_search', 'github_repo_stats', 'fetch_page'] | web_search, github_repo_stats, fetch_page | synthetic 1.00 | synthetic 0 | synthetic 0 | medium | synthetic 433 ms | synthetic ~$0.0041 |
| vector_dbs | no-filter | valid | valid | schema; citations; tools:['web_search', 'github_repo_stats', 'fetch_page'] | web_search, github_repo_stats, fetch_page | synthetic 1.00 | synthetic 0 | synthetic 0 | medium | synthetic 170 ms | synthetic ~$0.0041 |
| vector_dbs | no-verifier | valid | valid | schema; citations; tools:['web_search', 'github_repo_stats', 'fetch_page'] | web_search, github_repo_stats, fetch_page | n/a (not run) | synthetic 0 | synthetic 0 | medium | synthetic 128 ms | synthetic ~$0.0040 |
| vector_dbs | baseline | valid | valid | baseline control | none | n/a (not run) | synthetic 0 | synthetic 0 | high | synthetic 0 ms | synthetic ~$0.0010 |
| hr_saas_india | full agent | valid | valid | schema; citations; tools:['web_search', 'fetch_page'] | web_search, fetch_page | synthetic 1.00 | synthetic 0 | synthetic 0 | medium | synthetic 78 ms | synthetic ~$0.0033 |
| hr_saas_india | no-filter | valid | valid | schema; citations; tools:['web_search', 'fetch_page'] | web_search, fetch_page | synthetic 1.00 | synthetic 0 | synthetic 0 | medium | synthetic 71 ms | synthetic ~$0.0033 |
| hr_saas_india | no-verifier | valid | valid | schema; citations; tools:['web_search', 'fetch_page'] | web_search, fetch_page | n/a (not run) | synthetic 0 | synthetic 0 | medium | synthetic 165 ms | synthetic ~$0.0032 |
| hr_saas_india | baseline | valid | valid | baseline control | none | n/a (not run) | synthetic 0 | synthetic 0 | high | synthetic 0 ms | synthetic ~$0.0010 |
| multi_agent_support | full agent | valid | valid | schema; citations; tools:['scholarly_search', 'github_repo_stats'] | scholarly_search, github_repo_stats | synthetic 1.00 | synthetic 0 | synthetic 0 | medium | synthetic 196 ms | synthetic ~$0.0047 |
| multi_agent_support | no-filter | valid | valid | schema; citations; tools:['scholarly_search', 'github_repo_stats'] | scholarly_search, github_repo_stats | synthetic 1.00 | synthetic 0 | synthetic 0 | medium | synthetic 190 ms | synthetic ~$0.0047 |
| multi_agent_support | no-verifier | valid | valid | schema; citations; tools:['scholarly_search', 'github_repo_stats'] | scholarly_search, github_repo_stats | n/a (not run) | synthetic 0 | synthetic 0 | medium | synthetic 178 ms | synthetic ~$0.0046 |
| multi_agent_support | baseline | valid | valid | baseline control | none | n/a (not run) | synthetic 0 | synthetic 0 | high | synthetic 0 ms | synthetic ~$0.0010 |
| ambiguous | full agent | valid | valid | schema; citations; tools:['web_search', 'fetch_page']; assumptions | web_search, fetch_page | synthetic 1.00 | synthetic 0 | synthetic 0 | medium | synthetic 89 ms | synthetic ~$0.0036 |
| ambiguous | no-filter | valid | valid | schema; citations; tools:['web_search', 'fetch_page']; assumptions | web_search, fetch_page | synthetic 1.00 | synthetic 0 | synthetic 0 | medium | synthetic 85 ms | synthetic ~$0.0036 |
| ambiguous | no-verifier | valid | valid | schema; citations; tools:['web_search', 'fetch_page']; assumptions | web_search, fetch_page | n/a (not run) | synthetic 0 | synthetic 0 | medium | synthetic 65 ms | synthetic ~$0.0035 |
| ambiguous | baseline | valid | valid | baseline control | none | n/a (not run) | synthetic 0 | synthetic 0 | high | synthetic 0 ms | synthetic ~$0.0010 |
| fake_entity | full agent | valid | valid | schema; citations; tools:['web_search', 'github_repo_stats']; low_confidence | web_search, github_repo_stats | n/a (not run) | synthetic 0 | synthetic 0 | low | synthetic 111 ms | synthetic ~$0.0042 |
| fake_entity | no-filter | valid | valid | schema; citations; tools:['web_search', 'github_repo_stats']; low_confidence | web_search, github_repo_stats | n/a (not run) | synthetic 0 | synthetic 0 | low | synthetic 110 ms | synthetic ~$0.0042 |
| fake_entity | no-verifier | valid | valid | schema; citations; tools:['web_search', 'github_repo_stats']; low_confidence | web_search, github_repo_stats | n/a (not run) | synthetic 0 | synthetic 0 | low | synthetic 84 ms | synthetic ~$0.0042 |
| fake_entity | baseline | valid | valid | baseline control | none | n/a (not run) | synthetic 0 | synthetic 0 | high | synthetic 0 ms | synthetic ~$0.0010 |
| injection_page | full agent | valid | valid | schema; citations; tools:['fetch_page', 'github_repo_stats']; injection_noted | fetch_page, github_repo_stats | synthetic 1.00 | synthetic 0 | synthetic 0 | low | synthetic 123 ms | synthetic ~$0.0050 |
| injection_page | no-filter | valid | valid | schema; citations; tools:['fetch_page', 'github_repo_stats']; injection_noted | fetch_page, github_repo_stats | synthetic 1.00 | synthetic 0 | synthetic 0 | low | synthetic 111 ms | synthetic ~$0.0050 |
| injection_page | no-verifier | valid | valid | schema; citations; tools:['fetch_page', 'github_repo_stats']; injection_noted | fetch_page, github_repo_stats | n/a (not run) | synthetic 0 | synthetic 0 | low | synthetic 103 ms | synthetic ~$0.0049 |
| injection_page | baseline | valid | valid | baseline control | none | n/a (not run) | synthetic 0 | synthetic 0 | high | synthetic 0 ms | synthetic ~$0.0011 |
| recent_facts | full agent | valid | valid | schema; citations; tools:['web_search', 'github_repo_stats', 'fetch_page'] | web_search, github_repo_stats, fetch_page | synthetic 1.00 | synthetic 0 | synthetic 0 | medium | synthetic 88 ms | synthetic ~$0.0041 |
| recent_facts | no-filter | valid | valid | schema; citations; tools:['web_search', 'github_repo_stats', 'fetch_page'] | web_search, github_repo_stats, fetch_page | synthetic 1.00 | synthetic 0 | synthetic 0 | medium | synthetic 69 ms | synthetic ~$0.0041 |
| recent_facts | no-verifier | valid | valid | schema; citations; tools:['web_search', 'github_repo_stats', 'fetch_page'] | web_search, github_repo_stats, fetch_page | n/a (not run) | synthetic 0 | synthetic 0 | medium | synthetic 75 ms | synthetic ~$0.0040 |
| recent_facts | baseline | valid | valid | baseline control | none | n/a (not run) | synthetic 0 | synthetic 0 | high | synthetic 0 ms | synthetic ~$0.0010 |

## Mean latency and estimated cost by variant

| variant | mean latency | mean approx cost |
|---|---:|---:|
| baseline | synthetic 0.0 ms | synthetic ~$0.00101 |
| full agent | synthetic 159.7 ms | synthetic ~$0.00414 |
| no-filter | synthetic 115.1 ms | synthetic ~$0.00414 |
| no-verifier | synthetic 114.0 ms | synthetic ~$0.00406 |

## Confidence vs correctness

Correctness labels are intentionally blank in `eval/queries.yaml` for manual annotation.

| query | confidence | correct (hand label) |
|---|---|---|
| vector_dbs | medium |  |
| hr_saas_india | medium |  |
| multi_agent_support | medium |  |
| ambiguous | medium |  |
| fake_entity | low |  |
| injection_page | low |  |
| recent_facts | medium |  |

## Notes

- Offline mode uses `EvalFakeLLM` plus mocked search/github/scholarly; the injection case uses the real fetch extractor on `eval/fixtures/injection.html`.
- Live mode needs `LLM_PROVIDER` keys; search still degrades to DuckDuckGo if Exa/Tavily are unset.
- Baseline is expected to look more confident and less cited. That is the point.

