import os as _os
_REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
EV  = _os.path.join(_REPO, 'evaluation')
OUT = _os.path.join(_REPO, 'evaluation', 'outputs')
TEX = _os.path.join(_REPO, 'paper', 'main.tex')
import re

p = TEX
s = open(p, encoding='utf-8').read()

marker = "\\" + "begin{thebibliography}"
idx = s.find(marker)
body, bib = s[:idx], s[idx:]

order_bib = re.findall(r"\\bibitem\{([^}]+)\}", bib)
seen, first = set(), []
for m in re.finditer(r"\\cite\{([^}]+)\}", body):
    for k in m.group(1).split(','):
        k = k.strip()
        if k and k not in seen:
            seen.add(k)
            first.append(k)

print('bibitems:', len(order_bib), '| distinct cited:', len(first))
print('uncited bibitems:', [k for k in order_bib if k not in seen] or 'none')
print('cited but no bibitem:', [k for k in first if k not in order_bib] or 'none')

bad = [(i + 1, k, first.index(k) + 1)
       for i, k in enumerate(order_bib) if k in seen and first.index(k) != i]
print('\nout-of-order entries:', len(bad))
for cur, key, want in bad:
    print(f'  currently [{cur:2d}] {key:<22} -> should be [{want}]')

print('\ncorrect order (first-citation):')
print('  ' + ', '.join(first))
