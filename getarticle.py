import requests as rt
from bs4 import BeautifulSoup

class GuardianNews():

    def __init__(self):
        self.agent = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/76.0.3809.132 Safari/537.36'
        self.header = {
            'user-agent': self.agent,
            'referer': 'https://guardian.ng'
        }
        self.index = 0
        self.save_path = './1-750/'

    def getUrl(self, url):
        resp = rt.get(url, headers=self.header)
        soup = BeautifulSoup(resp.content, 'html.parser')
        content = soup.body.main.section.find_all(name='span', attrs='title')
        pUrl = []
        for item in content:
            pUrl.append(item.a.get('href'))
        return pUrl
    
    def write(self, url):
        print(url)
        r = rt.get(url, headers=self.header)
            
        soup = BeautifulSoup(r.content, 'html.parser')
        main = soup.body.find(name='main', attrs='page-main')
        article = main.find(name='div', attrs='content').find(name='article')
        content_list = article.find_all(name='p')
        self.index += 1
        with open(self.save_path + f"{self.index}.txt", 'a+') as f1:
            for content in content_list:
                if(content.string != None):
                    f1.write(content.string + '\n')

        

    def batchGet(self, index, down, up):
        categoryDict = {
            'Sino-US': 'https://guardian.ng/?s=Sino-US+relations',
            'China-US': 'https://guardian.ng/?s=China+US',
            'Taiwan': 'https://guardian.ng/?s=Taiwan+China',
            'COVID-19': 'https://guardian.ng/?s=COVID-19'
        }
        dic = {}
        if (index == -1):
            dic = categoryDict
        else:
            dic[index] = categoryDict[index]

        for category in dic:
            url = dic[category]
            for num in range(down, up):
                url_list = [url, '&page=', str(num)]
                url1 = ''
                url1 = url1.join(url_list)
                url2 = self.getUrl(url1)
                for u in url2:
                    self.write(u)
    

    def go(self):
        self.batchGet('China-US', 1, 140)
        self.batchGet('Taiwan', 1, 82)
        self.batchGet('Vaccine', 1, 120)

if __name__ == "__main__":
    News = GuardianNews()
    News.go()