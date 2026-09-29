import sqlite3, json
con = sqlite3.connect(r'data\vector_store\chroma.sqlite3')
cur = con.cursor()
tables = [r[0] for r in cur.execute("select name from sqlite_master where type='table'").fetchall()]
print('TABLES:', tables)
for t in ('collections','segements','segments','embeddings','embedding_metadata','max_seq_id'):
    pass
for t in tables:
    try:
        n = cur.execute(f'select count(*) from "{t}"').fetchone()[0]
        print(f'{t}: {n} rows')
    except Exception as e:
        print(t, 'ERR', e)
try:
    print('COLLECTIONS:', cur.execute('select id,name from collections').fetchall())
except Exception as e:
    print('collections err', e)
try:
    rows = cur.execute('select id, string_value from embedding_metadata where key in ("standard_number","standard_code","source","dataset_version") limit 40').fetchall()
    for r in rows: print(r)
except Exception as e:
    print('meta err', e)
