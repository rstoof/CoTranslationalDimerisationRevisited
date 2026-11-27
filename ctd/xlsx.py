"""Minimal .xlsx reader (stdlib only): returns {sheet_name: [[cell,...],...]}."""
import re, sys, zipfile
import xml.etree.ElementTree as ET
NS = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main',
      'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
      'p': 'http://schemas.openxmlformats.org/package/2006/relationships'}
def col2idx(c):
    n = 0
    for ch in c: n = n * 26 + ord(ch) - 64
    return n - 1
def read(path):
    z = zipfile.ZipFile(path)
    ss = []
    if 'xl/sharedStrings.xml' in z.namelist():
        for si in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('m:si', NS):
            ss.append(''.join(t.text or '' for t in si.iter('{%s}t' % NS['m'])))
    wb = ET.fromstring(z.read('xl/workbook.xml'))
    rels = {r.get('Id'): r.get('Target') for r in ET.fromstring(z.read('xl/_rels/workbook.xml.rels')).findall('p:Relationship', NS)}
    out = {}
    for s in wb.find('m:sheets', NS):
        tgt = rels[s.get('{%s}id' % NS['r'])]
        tgt = tgt.lstrip('/') if tgt.startswith('/') else 'xl/' + tgt
        rows = []
        for row in ET.fromstring(z.read(tgt)).iter('{%s}row' % NS['m']):
            cells = {}
            for c in row.findall('m:c', NS):
                ref = re.match(r'[A-Z]+', c.get('r')).group(0)
                v = c.find('m:v', NS); t = c.get('t')
                if t == 's' and v is not None: val = ss[int(v.text)]
                elif t == 'inlineStr': val = ''.join(x.text or '' for x in c.iter('{%s}t' % NS['m']))
                elif v is not None:
                    try: val = float(v.text)
                    except ValueError: val = v.text
                else: val = None
                cells[col2idx(ref)] = val
            if cells:
                r = [None] * (max(cells) + 1)
                for k, v in cells.items(): r[k] = v
                rows.append(r)
        out[s.get('name')] = rows
    return out
if __name__ == '__main__':
    for name, rows in read(sys.argv[1]).items():
        print('=== sheet', repr(name), len(rows), 'rows')
        for r in rows[:int(sys.argv[2]) if len(sys.argv) > 2 else 8]: print('   ', r)
