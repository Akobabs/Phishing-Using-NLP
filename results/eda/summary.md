# EDA Summary

## Enron
**Summary**: {'Size': 0, 'Columns': ['text', 'label'], 'Missing Values': {'text': 0, 'label': 0}, 'Duplicates': np.int64(0)}
**Label Distribution**: {}
**Text Stats**: {}
**Top Phishing Words**: []
**Top Legitimate Words**: []
**Top Phishing Bigrams**: []
**Top Legitimate Bigrams**: []
## PhishTank
**Summary**: {'Size': 64391, 'Columns': ['text', 'label'], 'Missing Values': {'text': 0, 'label': 0}, 'Duplicates': np.int64(0)}
**Label Distribution**: {1: 64391}
**Text Stats**: {'Char Length Mean': np.float64(46.2749141960833), 'Char Length Median': np.float64(27.0), 'Word Length Mean': np.float64(1.0321473497849079), 'Word Length Median': np.float64(1.0)}
**Top Phishing Words**: [('auth', 84), ('sso', 82), ('cdn.webflow.io/', 82), ('help', 58), ('auth.webflow.io/', 52), ('secure', 48), ('start', 47), ('app', 36), ('cdn', 30), ('coinbasepro', 23)]
**Top Legitimate Words**: []
**Top Phishing Bigrams**: [(('coinbasepro', 'cdn'), 9), (('cdn', 'auth.webflow.io/'), 8), (('auth', 'start'), 7), (('secure', 'blockfi'), 7), (('sso', 'coinbasepro'), 6), (('d1aqfkf.xn', 'p1ai/bitrix/'), 6), (('p1ai/bitrix/', '...'), 6), (('secure-app', 'coinbasepro'), 5), (('auth', 'sso'), 5), (('secure', 'suite'), 4)]
**Top Legitimate Bigrams**: []
## UCI
**Summary**: {'Size': 0, 'Columns': ['label'], 'Missing Values': {'label': 0}, 'Duplicates': np.int64(0)}
**Label Distribution**: {}
**Feature Stats**: {}
**Feature Correlations**: 
Empty DataFrame
Columns: []
Index: []
**Feature Importance**: 
Series([], )
## Combined (Enron + PhishTank)
**Summary**: {'Size': 64391, 'Columns': ['text', 'label'], 'Missing Values': {'text': 0, 'label': 0}, 'Duplicates': np.int64(0)}
**Label Distribution**: {1: 64391}
**Text Stats**: {'Char Length Mean': np.float64(46.2749141960833), 'Char Length Median': np.float64(27.0), 'Word Length Mean': np.float64(1.0321473497849079), 'Word Length Median': np.float64(1.0)}
**Top Phishing Words**: [('auth', 84), ('sso', 82), ('cdn.webflow.io/', 82), ('help', 58), ('auth.webflow.io/', 52), ('secure', 48), ('start', 47), ('app', 36), ('cdn', 30), ('coinbasepro', 23)]
**Top Legitimate Words**: []
**Top Phishing Bigrams**: [(('coinbasepro', 'cdn'), 9), (('cdn', 'auth.webflow.io/'), 8), (('auth', 'start'), 7), (('secure', 'blockfi'), 7), (('sso', 'coinbasepro'), 6), (('d1aqfkf.xn', 'p1ai/bitrix/'), 6), (('p1ai/bitrix/', '...'), 6), (('secure-app', 'coinbasepro'), 5), (('auth', 'sso'), 5), (('secure', 'suite'), 4)]
**Top Legitimate Bigrams**: []
