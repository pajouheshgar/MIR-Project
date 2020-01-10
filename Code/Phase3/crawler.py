import scrapy
from Code.Utils.Config import Config


class SemanticScholarSpider(scrapy.Spider):
    name = 'SemanticScholar_Crawler'

    start_urls = list(filter(lambda x: len(x) > 0, map(str.strip, open(Config.START_URL_DATA_DIR).readlines())))
    parsed_ids = set()
    limit_to = 7000
    total_added_to_queue = 3

    USER_AGENTS = [
        ('Mozilla/5.0 (X11; Linux x86_64) '
         'AppleWebKit/537.36 (KHTML, like Gecko) '
         'Chrome/57.0.2987.110 '
         'Safari/537.36'),  # chrome
        ('Mozilla/5.0 (X11; Linux x86_64) '
         'AppleWebKit/537.36 (KHTML, like Gecko) '
         'Chrome/61.0.3163.79 '
         'Safari/537.36'),  # chrome
        ('Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:55.0) '
         'Gecko/20100101 '
         'Firefox/55.0')  # firefox
    ]
    headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:48.0) Gecko/20100101 Firefox/48.0'}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print(self.start_urls)

    def parse(self, response):
        id = response.request.url.split('/')[-1]
        self.parsed_ids.add(id)
        if len(self.parsed_ids) % 150 == 149:
            print("Changing User Agents")
            self.headers['User-Agent'] = " ".join(self.USER_AGENTS[len(self.parsed_ids) // 150])
        title = response.xpath("//meta[@name='citation_title']/@content")[0].extract()
        abstract = response.xpath("//meta[@name='description']/@content")[0].extract()
        date = response.xpath("//meta[@name='citation_publication_date']/@content")[0].extract()
        print(id, title, date)
        authors = []
        for author in response.css('.author-list__author-name'):
            authors.append(author.css('span span::text').extract_first())

        reference_ids = []
        reference_urls = []
        # print(response.css('#references .citation__title a::attr(href)').extract_first())
        for reference in response.css('#references .citation__title'):
            reference_id = reference.css('::attr(data-heap-paper-id)').extract_first()
            if len(reference_id) == 0:
                continue
            reference_ids.append(reference_id)
            try:
                reference_url = "https://www.semanticscholar.org" + reference.css('a::attr(href)').extract_first()
                reference_urls.append((reference_id, reference_url))
            except:
                pass
        reference_ids = reference_ids[:10]
        reference_urls = reference_urls[:5]
        if len(self.parsed_ids) < self.limit_to:
            for ref_id, ref_url in reference_urls:
                if ref_id not in self.parsed_ids:
                    if self.total_added_to_queue < self.limit_to:
                        self.total_added_to_queue += 1
                        yield scrapy.Request(ref_url, callback=self.parse, headers=self.headers)

        yield {
            'id': id,
            'title': title,
            'abstract': abstract,
            'date': date,
            'authors': authors,
            'references': reference_ids
        }


if __name__ == '__main__':
    from scrapy.crawler import CrawlerProcess

    process = CrawlerProcess(settings={
        'DOWNLOAD_DELAY': 0.25,
        'FEED_FORMAT': 'json',
        'FEED_URI': 'Data/Phase3/crawler2.json'
    })

    process.crawl(SemanticScholarSpider)
    process.start()
