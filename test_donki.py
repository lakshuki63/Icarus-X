import requests

url = "https://api.nasa.gov/DONKI/FLR?startDate=2020-01-01&endDate=2023-12-31&api_key=DEMO_KEY"
resp = requests.get(url)
data = resp.json()
classes = [d.get('classType', '')[0] for d in data if 'classType' in d]
from collections import Counter
print(Counter(classes))
