# Local vs api_local predictions

API: `http://127.0.0.1:8000` | tolerance on probability: 1e-06

| # | local class | api class | local P | api P | abs diff | match |
|---|---|---|---|---|---|---|
| 0 | malignant | malignant | 0.99959789 | 0.99959789 | 0.00e+00 | yes |
| 1 | benign | benign | 0.99654403 | 0.99654403 | 0.00e+00 | yes |
| 2 | malignant | malignant | 0.93680229 | 0.93680229 | 0.00e+00 | yes |
| 3 | malignant | malignant | 0.67733333 | 0.67733333 | 0.00e+00 | yes |
| 4 | malignant | malignant | 0.99968638 | 0.99968638 | 0.00e+00 | yes |
| 5 | benign | benign | 0.95472356 | 0.95472356 | 0.00e+00 | yes |
| 6 | benign | benign | 0.99639179 | 0.99639179 | 0.00e+00 | yes |
| 7 | malignant | malignant | 0.99959789 | 0.99959789 | 0.00e+00 | yes |
| 8 | malignant | malignant | 0.99959789 | 0.99959789 | 0.00e+00 | yes |
| 9 | malignant | malignant | 0.99959789 | 0.99959789 | 0.00e+00 | yes |

**Result: ALL MATCH**