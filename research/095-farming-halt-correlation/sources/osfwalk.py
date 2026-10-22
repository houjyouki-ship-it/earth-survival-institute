import json,urllib.request
def walk(url,depth=0):
    while url:
        d=json.load(urllib.request.urlopen(url))
        for f in d['data']:
            a=f['attributes']
            print('  '*depth+a['kind'],a['materialized_path'],a.get('size'),f['links'].get('download',''))
            if a['kind']=='folder' and depth<3:
                walk(f['relationships']['files']['links']['related']['href'],depth+1)
        url=d['links'].get('next')
walk("https://api.osf.io/v2/nodes/yrbse/files/osfstorage/")
