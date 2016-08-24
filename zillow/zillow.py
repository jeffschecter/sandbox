#!/usr/bin/env python

import multiprocessing
import os
import re
import urllib2

import bs4

import pandas as pd
import scipy as sp
import numpy as np


def ParseZpid(zpid):
    url = "http://www.zillow.com/homedetails/{}_zpid/".format(zpid)
    html = urllib2.urlopen(url).read()
    soup = bs4.BeautifulSoup(html, "lxml")

    # Zestimate
    zestimate_text = soup.div(class_="zest-value")[0].text
    try:
        zestimate = float(re.sub(r"[^0-9]", "", zestimate_text))
    except ValueError:
        zestimate = 0

    # Last sold
    last_sold_text = ""
    for square in soup.div(class_="zsg-list_square"):
        for bullet in square.children:
            if "Last sold" in bullet.text:
                last_sold_text = bullet.text
    last_sold = float(re.sub(r"(^.*\$)|([^0-9])", "", last_sold_text))

    return zestimate, last_sold


def ParseImageContainer(cont):
    nodes = list(cont.children)
    if len(nodes) != 2:
        return None
    else:
        img = nodes[1]
        url = img.get("data-src")
        if url is None:
            url = img.get("src")
        return url


def ParseResultsPage(region, page=1):
    url = "http://www.zillow.com/{region}/sold/house_type/{page}_p/".format(
        region=region, page=page)
    doc = urllib2.urlopen(url)
    html = doc.read()
    soup = bs4.BeautifulSoup(html, "lxml")

    # ZPIDs
    links = [
        a["href"] for a
        in soup.find_all("a", class_="zsg-photo-card-overlay-link")]
    pids = [re.search(r"/(\d+)_zpid/", href).group(1) for href in links]

    # Photos
    img_urls = [
        ParseImageContainer(cont) for cont 
        in soup.find_all("div", class_="zsg-photo-card-img")]
    
    return [
        (pid, url) for pid, url in zip(pids, img_urls)
        if url is not None]


def ParseResultsPageArgWrapper(pair):
    return ParseResultsPage(pair[0], page=pair[1])


def ScanRegion(city, state_abbrev):
    region = "{}-{}".format(city, state_abbrev)
    pool = multiprocessing.Pool(20)
    pages = pool.map(
        ParseResultsPageArgWrapper,
        zip(np.repeat(region, 20), np.arange(1, 21)),
        chunksize=1)
    pool.close()
    listings = []
    for page in pages:
        listings += page
    return listings


def ProcessListing(zpid_imgurl_pair):
    try:
        zpid, url = zpid_imgurl_pair
        zestimate, last_sold = ParseZpid(zpid)
        img = urllib2.urlopen(url).read()
        return zpid, url, zestimate, last_sold, img
    except:
        return


def BulkProcessListings(zpid_imgurl_pair_list, poolsize=20):
    pool = multiprocessing.Pool(poolsize)
    listing_data = pool.map(ProcessListing, zpid_imgurl_pair_list)
    pool.close()
    return [l for l in listing_data if l is not None]


def SaveListingData(listing_data, city, state_abbrev, savedir):
    handle = "{}-{}".format(city, state_abbrev)
    df = pd.DataFrame(
        listing_data,
        columns=("zpid", "img_url", "zestimate", "last_sold", "img"))

    # Images
    img_dir = os.path.join(savedir, "raw_images")
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)
    for _, row in df.iterrows():
        img_path = os.path.join(img_dir, "{}.jpg".format(str(row.zpid)))
        with open(img_path, "w") as f:
            f.write(row.img)

    # Tabular data
    tsv_dir = os.path.join(savedir, "tabular")
    if not os.path.exists(tsv_dir):
        os.mkdir(tsv_dir)
    df["city"] = np.repeat(city, df.shape[0])
    df["state"] = np.repeat(state_abbrev, df.shape[0])
    df.drop("img", axis=1).to_csv(
        os.path.join(tsv_dir, "{}.tsv".format(handle)),
        sep='\t', index=False)


def ProcessRegion(city, state_abbrev, savedir):
    listings = ScanRegion(city, state_abbrev)
    listing_data = BulkProcessListings(listings, poolsize=100)
    SaveListingData(listing_data, city, state_abbrev, savedir)


def CropAndResize(img, size=128):
    h, w = img.shape[:2]
    shortest = min(h, w)
    cropped = img[
        (h - shortest) / 2:h + ((h - shortest) / 2),
        (w - shortest) / 2:h + ((w - shortest) / 2)]
    return sp.misc.imresize(cropped, (size, size))


def PostprocessImages(savedir):
    raw_dir = os.path.join(savedir, "raw_images")
    processed_dir = os.path.join(savedir, "processed_images")
    if not os.path.exists(processed_dir):
        os.mkdir(processed_dir)

    raw_handles = set(os.listdir(raw_dir))
    processed_handles = set(os.listdir(processed_dir))
    to_process = raw_handles - processed_handles

    for handle in to_process:
        inpath = os.path.join(raw_dir, handle)
        outpath = os.path.join(processed_dir, handle)
        sp.misc.imsave(outpath, CropAndResize(sp.misc.imread(inpath)))


def LoadTabularData(savedir):
    tsv_dir = os.path.join(savedir, "tabular")
    dfs = [
        pd.read_csv(os.path.join(tsv_dir, handle), sep="\t", header=0)
        for handle in os.listdir(tsv_dir)]
    df = pd.concat(dfs, ignore_index=True)
    df["log_sold_to_zestimate"] = np.log2(df.last_sold / df.zestimate)
    cleaned = df[(df.zestimate > 0) & (df.last_sold > 0)]
    return cleaned.drop_duplicates("zpid").reset_index(drop=True)


def LoadImageDataIntoDataFrame(df, savedir):
    img_dir = os.path.join(savedir, "processed_images")
    images = df.zpid.map(
        lambda z: sp.misc.imread(os.path.join(img_dir, "{}.jpg".format(z))))
    df["image"] = images
