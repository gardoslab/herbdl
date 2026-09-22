
from typing import Literal

from bs4 import BeautifulSoup as BS

import requests as r

class site_tools:

    class wikipedia:

        @staticmethod
        def description(soup) -> str:

            # Get and append 'Description', 'Distribution', 'Ecology', 'Botany', 'Appearance' and first paragraph

            paragraphs = []

            # Get first paragraph

            first_para = soup.find(lambda p: p.name == 'p' and 'mw-empty-elt' not in (p.get('class', [])))

            if first_para:

                paragraphs.append(first_para.text)

            # Get named paragraphs if available

            find_tags = ['Description', 'Distribution', 'Ecology', 'Botany', 'Appearance']

            for tag in find_tags:

                if not (desc := soup.find('h2', id=tag, string=tag)):
                    continue

                iter_ = desc.parent

                while (next_p := iter_.find_next_sibling()) and next_p.name == 'p':

                    paragraphs.append(next_p.text)

                    iter_ = next_p
            
            return "\n".join(paragraphs)

        @staticmethod
        def common_names(soup) -> list[str]:

            # Find first paragraph
            body = soup.find(class_='mw-content-ltr')

            return [b.text for b in body.find_all(lambda tag: tag.name == 'p' and not tag.has_attr('class'))[0].find_all('b')][1:]

        @staticmethod
        def valid_url(soup) -> bool:
            
            return not ((mbox := soup.find(class_='mbox-text')) and 'Wikipedia does not have an article with this exact name.' in mbox.text)

        @staticmethod
        def search_format(genus, species) -> str:

            return f'https://en.wikipedia.org/w/index.php?title={genus.replace(" ", "_")}_{species.replace(" ", "_")}'

    class ncsu:

        @staticmethod
        def description(soup) -> str: 
            
            return dt.find_next_sibling("dd").text \
                if (dt := soup.find("dt", string="Description")) else ''

        @staticmethod
        def common_names(soup) -> list[str]:

            return [name.text for name in soup.find(id='common_names').find_all('li')] \
                if soup.find(id='common_names') else []

        @staticmethod
        def valid_url(soup) -> bool:

            return '404 Not Found' not in soup.title.text

        @staticmethod
        def search_format(genus, species) -> str:

            return f'https://plants.ces.ncsu.edu/plants/{genus.replace(" ", "-")}-{species.replace(" ", "-")}/'

    class britannica:

        @staticmethod
        def valid_url(soup) -> bool:

            return not bool(soup.find(class_='md-error-search-box'))

        @staticmethod
        def search_format(common_names) -> list[str]:

            urls = [f'https://britannica.com/plant/{cn.replace(" ", "-")}' for cn in common_names]

            return urls

        @staticmethod
        def common_names(soup) -> list[str]:

            return []

        @staticmethod
        def description(soup) -> str: 
            
            base_url = 'https://britannica.com'

            article = soup.find('article')

            if article is None:

                return ''

            if (read_more := article.find('a', class_='read-more')):

                link, ref = read_more['href'].split('#')

                l = base_url + link

                wrapper_page = BS(r.get(l).text)

                return wrapper_page.find('span', id=ref).parent.text

            else:

                paragraphs = article.find_all('p', class_='topic-paragraph')

                return '\n'.join(p.text for p in paragraphs)

    class epicgardening:

        @staticmethod
        def search_format(common_names) -> list[str]:

            return [f'https://epicgardening.com/{cn.replace(" ", "-")}/' for cn in common_names]

        @staticmethod
        def common_names(soup) -> list[str]:

            return []
