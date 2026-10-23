import sys, zipfile, csv, re
from lxml import etree
src, out = sys.argv[1], sys.argv[2]
NS='{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
z=zipfile.ZipFile(src)
ss=[]
for _,si in etree.iterparse(z.open('xl/sharedStrings.xml'), tag=NS+'si'):
    ss.append(''.join(si.itertext())); si.clear()
def colidx(ref):
    n=0
    for ch in ref:
        if ch.isalpha(): n=n*26+ord(ch)-64
        else: break
    return n-1
w=csv.writer(open(out,'w',newline='',encoding='utf8'))
cnt=0
for _,row in etree.iterparse(z.open('xl/worksheets/sheet1.xml'), tag=NS+'row'):
    vals={}
    for c in row:
        r=c.get('r'); t=c.get('t'); i=colidx(r)
        if t=='inlineStr':
            v=''.join(c.itertext())
        else:
            ve=c.find(NS+'v'); v=ve.text if ve is not None else ''
            if t=='s' and v!='': v=ss[int(v)]
        vals[i]=v
    if vals:
        m=max(vals)+1
        w.writerow([vals.get(k,'') for k in range(m)])
    cnt+=1
    row.clear()
    while row.getprevious() is not None: del row.getparent()[0]
    if cnt%100000==0: print(cnt, flush=True)
print('done', cnt)
