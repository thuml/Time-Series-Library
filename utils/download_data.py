import requests

if __name__=="__main__":
    source_url = 'https://cloud.tsinghua.edu.cn/d/e1ccfff39ad541908bae/files/?p=%2Fall_six_datasets.zip&dl=1'
    headers = {'User-Agent': 'Mozilla/5.0'}
    res = requests.get(source_url, headers=headers)

    with open('dataset/datasets.zip', 'wb') as f:
        f.write(res.content)
