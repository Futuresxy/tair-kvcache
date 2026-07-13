# ShareGPT 300 HiCache Matrix Summary

| policy | pass | completed | prefix cache reused | disk prefetch | mean TTFT ms | mean TPOT ms | mean E2E ms | storage keys |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lru | cold | 300 | 4.72% | 0.00% | 64872.36 | 29.24 | 83994.61 | 241024 |
| lru | warm | 300 | 19.74% | 28.93% | 65534.61 | 34.67 | 88155.49 | 243469 |
| lfu | cold | 300 | 4.74% | 0.00% | 64925.02 | 29.25 | 84053.07 | 241023 |
| lfu | warm | 300 | 15.27% | 29.25% | 64176.10 | 34.05 | 86401.07 | 243546 |
