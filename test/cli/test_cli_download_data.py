#!/usr/bin/python3
# -*- coding: utf-8 -*

import requests
from shimmingtoolbox.cli.download_data import URL_DICT


def test_links():
    for key in URL_DICT:
        for url in URL_DICT[key][0]:
            assert requests.head(url).status_code != 404
