import requests as r
from IPython.display import clear_output
from bs4 import BeautifulSoup as BS
import time, os, json

from typing import Dict, Callable, List

from collections import defaultdict

class scraping_tools:

    @staticmethod
    def get_all_plant_urls(search_sites: list[str], search_format: Dict[str, Callable], unique_plants: List[str], sci: bool = True) -> list[str]:

        name_urls = {}

        assert all([site in search_format for site in search_sites]), f'Not all sites sites have a search format'

        for site in search_sites:

            name_urls[site] = []

            for i, plant_name in enumerate(unique_plants):

                if sci:

                    genus, species = plant_name.split('^^^')

                    name_urls[site].append(
                        {
                            'plant': " ".join([genus, species]),
                            'url': search_format[site](genus, species),
                        }
                    ) # Change to work with common urls

                # Common Name
                else:

                    name_urls[site].append(
                        {
                            'plant': plant_name,
                            'url': search_format[site](unique_plants[plant_name]),
                        }
                    )

                        

        return name_urls

    @staticmethod
    def scrape_sites(sites: list[str], unique_plants: list[str], search_format: dict[str, callable], site_to_name: Dict[str, str], if_valid_url: Dict[str, Callable], get_desctiption: Dict[str, Callable], get_site_common_names: Dict[str, Callable], sci: bool = True) -> dict:

        urls = scraping_tools.get_all_plant_urls(sites, search_format=search_format, unique_plants=unique_plants, sci=sci)

        retrieved_sites = {}

        for site in sites:
            
            save_file: str = f'site_results.o.{site_to_name[site]}.json'

            if os.path.exists(save_file):

                with open(save_file, 'r') as sf:

                    data = json.load(sf)

                    retrieved_sites[site] = data[list(data.keys())[0]][site]

            else:

                retrieved_sites[site] = {}

                retrieved_sites[site]['stats'] = {'searched': 0, 'found': 0, 'percent': 0}

                s = retrieved_sites[site]

                site_urls: int = urls[site].__len__()

                lasttime = time.time()

                for i, u in enumerate(urls[site]):

                    # Save results 
                    if i and i % 500 == 0:

                         with open(save_file, 'w') as sf:

                            json.dump(retrieved_sites, sf)

                    if sci:

                        if (curr_time := time.time()) - lasttime > 1:

                            clear_output()

                            s['stats']['percent'] = s['stats']['found']/s['stats']['searched']

                            print(site)
                            print(f'Searching: {i}/{site_urls}')
                            print(f"Count: {s['stats']['percent']:.2f} {s['stats']['found']}/{s['stats']['searched']}")

                            lasttime = curr_time

                        response = r.get(u['url'])

                        page = response.text

                        s['stats']['searched'] += 1

                        page_soup = BS(page, 'html.parser')


                        if not if_valid_url[site](page_soup):

                            s[u['url']] = {
                                'plant': u['plant'],
                                'exists': False,
                                'common_names': [],
                                'description': '',
                                'page': '',
                            }

                            continue
                        
                        else:

                            s['stats']['found'] += 1

                            description = get_desctiption[site](page_soup)
                            common_names = get_site_common_names[site](page_soup)

                            s[u['url']] = {
                                'plant': u['plant'],
                                'exists': True,
                                'common_names': common_names,
                                'description': description,
                                'page': page,
                            }

                    # Common names
                    else:
                        
                        for url in u['url']:

                            if (curr_time := time.time()) - lasttime > 1:

                                clear_output(wait=True)

                                s['stats']['percent'] = s['stats']['found']/s['stats']['searched']

                                print(site)
                                print(f'Searching: {i}/{site_urls}')
                                print(f"Count: {s['stats']['percent']:.2f} {s['stats']['found']}/{s['stats']['searched']}")

                                lasttime = curr_time

                            response = r.get(url)

                            page = response.text

                            s['stats']['searched'] += 1

                            page_soup = BS(page, 'html.parser')


                            if not if_valid_url[site](page_soup):

                                s[url] = {
                                    'plant': u['plant'],
                                    'exists': False,
                                    'common_names': [],
                                    'description': '',
                                    'page': '',
                                }

                                continue
                            
                            else:

                                s['stats']['found'] += 1

                                description = get_desctiption[site](page_soup)
                                common_names = get_site_common_names[site](page_soup)

                                s[url] = {
                                    'plant': u['plant'],
                                    'exists': True,
                                    'common_names': common_names,
                                    'description': description,
                                    'page': page,
                                }

                with open(save_file, 'w') as sf:

                    json.dump({
                        f"{'sci' if sci else 'common'}": {
                            f'{site}': s
                        }
                    }, sf)


                clear_output()

                # outputfile = 'site_stats.o.txt'

                # with open(outputfile, 'a') as of:
                #     print(f'Searching: {i}/{site_urls}', file=of)
                #     print(f'Found: {s["stats"]["found"]}/{s["stats"]["searched"]}', file=of)

        return retrieved_sites

    @staticmethod
    def get_common_from_sci(sci_scraping) -> List[List[str]]:

        # ex key: sci -> common
        sci_and_common_names = defaultdict(list)

        for site in sci_scraping.values():

            for plant, plant_info in zip(site, site.values()):

                if plant != 'stats' and plant_info['common_names']:

                    sci_and_common_names[plant_info.get('plant', ' '.join(plant.split('/')[-1].split('=')[-1].split('_')))].extend(plant_info['common_names'])

        return sci_and_common_names