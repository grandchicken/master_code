import re
pattern = '1-caramel-dumplings.*'
s = '1-caramel-dumplings_4_2.jpg'
s2 = 'wcljqnjkkwh'
result = re.search(pattern,s2)
print(result)
print(result.group())

