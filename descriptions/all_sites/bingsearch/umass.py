
from typing import Dict

class results:

    @staticmethod
    def load_stats() -> Dict:

        common_sites = {}

        with open('common_sites.txt', 'r') as cs:

            lines = [line.split(': ') for line in cs.read().split('\n')]

            for site, num in lines:

                common_sites[site] = num

        import json

        with open('pages.jsoin', 'r') as pages:

            bing_search_data = json.load(pages)

        with open('example_data.jsoin', 'r') as pages:

            umass_example_data = json.load(pages)

        def get_baseurl(url: str) -> str:
            """
            Returns base url ex (google.com) to group bing results
            """
            return url.split('//')[1].split('/')[0].removeprefix('www.')

        # Load dictionary for urls associated with plant or base url

        site_examples = {}

        plant_to_urls = {} # key: plant -> key: baseurl -> list[exact urls] 

        for plant_name, plant_info in zip(bing_search_data, list(bing_search_data.values())):

            plant_to_urls[plant_name] = {}

            for site in plant_info['pages']:

                base_url: str = get_baseurl(site['link'])

                if base_url not in site_examples:
                    site_examples[base_url] = []

                site_examples[base_url].append(site['link'])

                if base_url not in plant_to_urls[plant_name]:

                    plant_to_urls[plant_name][base_url] = []

                plant_to_urls[plant_name][base_url].append(site['link'])

        # Remove repeats

        for burl in site_examples:

            site_examples[burl] = set(site_examples[burl])


        # Score urls based on hits in bing search with sci name and common name

        from colorama import Fore

        site_scores = {} # key: baseurl -> key: search int (number of urls on burl) /
        # && key: found -> key: common int && key: sci int / && key: misses list[exact url]

        # Initialize all baseurls in bing search data

        [site_scores.__setitem__(burl, {'searched': 0, 'found': {'common': 0, 'sci': 0}, 'misses': []}) for burl in site_examples]

        for plant_name, plant_info in zip(bing_search_data, list(bing_search_data.values())):

            common_name = plant_name
            sci_name = umass_example_data[plant_name]['sci_name']

            for base_url, urls in zip(plant_to_urls[plant_name], list(plant_to_urls[plant_name].values())):

                for url in urls:

                    end_route = [i for i in url.split('/') if i != ''][-1].lower()

                    site_scores[base_url]['searched'] += 1

                    if end_route == sci_name.lower().replace(' ', '-'):

                        site_scores[base_url]['found']['sci'] += 1

                    elif end_route == common_name.lower().replace(' ', '-'):

                        site_scores[base_url]['found']['common'] += 1

                    else:

                        site_scores[base_url]['misses'].append(url)

        for base_url, stats in zip(site_scores, list(site_scores.values())):

            stats['scores'] = {
                'common': stats['found']['common']/stats['searched'],
                'sci': stats['found']['sci']/stats['searched'],
            }

        return site_scores

    def display_sci(site_scores = None, least_examples: int = 20) -> str:

        if not site_scores:

            site_scores = results.load_stats()

        top_sci = sorted(site_scores, key=lambda s: site_scores[s]['scores']['sci'] if site_scores[s]['searched'] >= least_examples else 0, reverse=True)

        return '\n'.join([*[f'{ts}: {site_scores[ts]["searched"]} -> {site_scores[ts]["scores"]["sci"]:.2f}' for ts in top_sci[:10]]])

    def display_common(site_scores = None, least_examples: int = 20) -> str:

        if not site_scores:

            site_scores = results.load_stats()

        top_common = sorted(site_scores, key=lambda s: site_scores[s]['scores']['common'] if site_scores[s]['searched'] >= least_examples else 0, reverse=True)

        return '\n'.join([*[f'{tc}: {site_scores[tc]["searched"]} -> {site_scores[tc]["scores"]["common"]:.2f}' for tc in top_common[:10]]])
    