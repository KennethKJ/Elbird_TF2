import logging
from io import BytesIO

import flickrapi
import requests


def get_flickr_image_by_keyword(keyword):
    """Given a keyword, search Flickr for it and return a slightly-random result as a file descriptor"""
    logging.info("Getting {} from Flickr".format(keyword))
    flickr = flickrapi.FlickrAPI(FLICKR_KEY, FLICKR_SECRET, format='etree')
    result = flickr.photos.search(per_page=100,
                                  text=keyword,
                                  tag_mode='all',
                                  content_type=1,
                                  tags=keyword,
                                  extras='url_o,url_l',
                                  sort='relevance')
    # Randomize the result set
    img_url = None
    photos = [p for p in result[0]]
    while img_url is None and len(photos) > 0:
        photo = photos[0]
        img_url = photo.get('url_o') or photo.get('url_l')
        photos.pop()
    if not img_url:
        raise Exception("Couldn't find a Flickr result for %s" % keyword)
    logging.info(img_url)
    img_file = requests.get(img_url, stream=True)
    return BytesIO(img_file.content)