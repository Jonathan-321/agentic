| Condition | Task type    | Context bucket | Pass rate | Avg steps | N |
|-----------|--------------|----------------|-----------|-----------|---|
| baseline  | long_horizon | n/a            | 100%      | 2.0       | 20 |
| baseline  | needle       | <=800          | 100%      | 3.0       | 8 |
| baseline  | needle       | <=2500         | 37.5%     | 3.0       | 8 |
| baseline  | needle       | <=5000         | 25.0%     | 3.0       | 8 |
| baseline  | needle       | >5000          | 0%        | 3.0       | 8 |
| summary   | needle       | >5000          | 0%        | 3.0       | 8 |
| retrieval | needle       | >5000          | 100%      | 3.0       | 8 |
| both      | needle       | >5000          | 100%      | 3.0       | 8 |
