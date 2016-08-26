#!/usr/bin/env python

import multiprocessing
import os
import re
import urllib2

import bs4

import pandas as pd
import numpy as np

from scipy import misc
from sklearn.feature_extraction.text import CountVectorizer


NONDIGIT_RE = re.compile(r"[^0-9.]")
LAST_SOLD_RE = re.compile(r"(^.*\$)|([^0-9])")
ADDR_RE = re.compile(r"^([^,]+), ([A-Z]{2}) (\d+)$")
ZPID_RE = re.compile(r"/(\d+)_zpid/")
NONWORD_RE = re.compile(r"\W")
NOW = pd.to_datetime("now")


# --------------------------------------------------------------------------- #
# Helpers.                                                                    #
# --------------------------------------------------------------------------- #

def ParseNumber(s):
  return float(NONDIGIT_RE.sub("", s))


# --------------------------------------------------------------------------- #
# Parse an individual home's listing.                                         #
# --------------------------------------------------------------------------- #

def ParseFacts(soup):
  facts = []
  fact_containers = soup.find_all("div", class_="fact-group-container")
  for cont in fact_containers:
    these_facts = cont.find_all("li")
    for fact in these_facts:
      if fact["class"] == [""]:
        facts.append(fact.text)
  return facts


def ParseZestimate(soup):
  zestimate_text = soup.div(class_="zest-value")[0].text
  try:
    return ParseNumber(zestimate_text)
  except ValueError:
    return 0


def ParseLastSold(facts):
  last_sold_text = ""
  for fact in facts:
    if "Last sold" in fact:
      last_sold_text = fact
      break
  return float(LAST_SOLD_RE.sub("", last_sold_text))


def ParsePermits(soup):
  permits = []
  for permit in soup.find_all("div", class_="building-permit"):
    status, sqft, cost, _ = permit.find_all("li")
    date = pd.to_datetime(permit.find("time").text)
    if date > NOW:
      date = date - pd.Timedelta(100, "Y")
    permits.append({
      "date": str(date.date()),
      "description": permit.find("p", class_="description").text,
      "status": status.text.partition("Status ")[2],
      "sqft": ParseNumber(sqft.text),
      "cost": ParseNumber(cost.text.partition("Est. Project Cost ")[2])})
  return permits


def ParseBasicInfo(soup):
  bed_span, bath_span, sqft_span = soup.find_all("span", class_="addr_bbs")
  return {
      "beds": int(bed_span.text.split()[0]),
      "baths": float(bath_span.text.split()[0]),
      "sqft": ParseNumber(sqft_span.text)}


def ParseAddress(soup):
  addr_text = soup.find("span", class_="addr_city").text
  match = ADDR_RE.search(addr_text)
  return {
      "city": match.group(1),
      "state": match.group(2),
      "zip": int(match.group(3))}


def ParseHomeDetailsPage(zpid):
  url = "http://www.zillow.com/homedetails/{}_zpid/".format(zpid)
  html = urllib2.urlopen(url).read()
  soup = bs4.BeautifulSoup(html, "lxml")
  facts = ParseFacts(soup)
  details = {
      "zestimate": ParseZestimate(soup),
      "last_sold": ParseLastSold(facts),
      "facts": facts,
      "permits": ParsePermits(soup)}
  details.update(ParseBasicInfo(soup))
  details.update(ParseAddress(soup))
  return details


def ProcessListing(zpid_imgurl_pair):
  try:
    zpid, url = zpid_imgurl_pair
    home_details = {
        "zpid": zpid,
        "img_url": url,
        "image": urllib2.urlopen(url).read()}
    home_details.update(ParseHomeDetailsPage(zpid))
    return home_details
  except Exception as e:
    {"zpid": zpid, "error": e}


# --------------------------------------------------------------------------- #
# Parse a search results page.                                                #
# --------------------------------------------------------------------------- #

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
  base_url = "http://www.zillow.com/{region}/sold/house_type/{page}_p/"
  url = base_url.format(region=region, page=page)
  doc = urllib2.urlopen(url)
  html = doc.read()
  soup = bs4.BeautifulSoup(html, "lxml")

  # ZPIDs
  links = [
      a["href"] for a
      in soup.find_all("a", class_="zsg-photo-card-overlay-link")]
  pids = [ZPID_RE.search(href).group(1) for href in links]

  # Photos
  img_urls = [
      ParseImageContainer(cont) for cont 
      in soup.find_all("div", class_="zsg-photo-card-img")]
  
  return [
      (pid, url) for pid, url in zip(pids, img_urls)
      if url is not None]


def ParseResultsPageWrapper(pair):
  return ParseResultsPage(pair[0], page=pair[1])


# --------------------------------------------------------------------------- #
# Mass data fetching with multiprocessing.                                    #
# --------------------------------------------------------------------------- #

def ScanRegion(city, state_abbrev, zipcode=None):
  region = "{}-{}".format(city, state_abbrev)
  if zipcode is not None:
    region += "-" + str(zipcode)
  pool = multiprocessing.Pool(20)
  pages = pool.map(
      ParseResultsPageWrapper,
      zip(np.repeat(region, 20), np.arange(1, 21)),
      chunksize=1)
  pool.close()
  listings = []
  for page in pages:
    listings += page
  return listings


def BulkProcessListings(zpid_imgurl_pair_list, poolsize=20):
  pool = multiprocessing.Pool(poolsize)
  listing_data = pool.map(ProcessListing, zpid_imgurl_pair_list)
  pool.close()
  return [l for l in listing_data if l is not None]


def SaveListingData(listing_data, city, state_abbrev, savedir, zipcode=None):
  handle = "{}_{}".format(city, state_abbrev)
  if zipcode is not None:
    handle += "_" + str(zipcode)
  df = pd.DataFrame(listing_data)

  # Images
  img_dir = os.path.join(savedir, "raw_images")
  if not os.path.exists(img_dir):
    os.mkdir(img_dir)
  for _, row in df.iterrows():
    img_path = os.path.join(img_dir, "{}.jpg".format(str(row.zpid)))
    with open(img_path, "w") as f:
      f.write(row.image)

  # Tabular data
  tsv_dir = os.path.join(savedir, "tabular")
  if not os.path.exists(tsv_dir):
    os.mkdir(tsv_dir)
  df.drop("image", axis=1).to_csv(
      os.path.join(tsv_dir, "{}.tsv".format(handle)),
      sep='\t', index=False)


def ProcessRegion(city, state_abbrev, savedir, zipcode=None, poolsize=100):
  listings = ScanRegion(city, state_abbrev, zipcode=zipcode)
  listing_data = BulkProcessListings(listings, poolsize=poolsize)
  SaveListingData(listing_data, city, state_abbrev, savedir, zipcode=zipcode)


# --------------------------------------------------------------------------- #
# Image normalization.                                                        #
# --------------------------------------------------------------------------- #

def CropAndResize(img, size=128):
  h, w = img.shape[:2]
  shortest = min(h, w)
  cropped = img[
      (h - shortest) / 2:h + ((h - shortest) / 2),
      (w - shortest) / 2:h + ((w - shortest) / 2)]
  return misc.imresize(cropped, (size, size))


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
    misc.imsave(outpath, CropAndResize(misc.imread(inpath)))


# --------------------------------------------------------------------------- #
# Data loading.                                                               #
# --------------------------------------------------------------------------- #

def LoadTabularData(savedir):
  tsv_dir = os.path.join(savedir, "tabular")
  dfs = [
      pd.read_csv(os.path.join(tsv_dir, handle), sep="\t", header=0)
      for handle in os.listdir(tsv_dir)]
  df = pd.concat(dfs, ignore_index=True)
  df["log_sold_to_zestimate"] = np.log2(df.last_sold / df.zestimate)
  df["facts"] = (df.facts
      .map(lambda s: s.strip("u[],'"))
      .map(lambda s: s.split("', u'"))
      .map(lambda l: [elt for elt in l if "ast s" not in elt]))
  cleaned = df[(df.zestimate > 0) & (df.last_sold > 0)]
  return cleaned.drop_duplicates("zpid").reset_index(drop=True)


def LoadImageDataIntoDataFrame(df, savedir):
  img_dir = os.path.join(savedir, "processed_images")
  images = df.zpid.map(
      lambda z: misc.imread(os.path.join(img_dir, "{}.jpg".format(int(z)))))
  df["image"] = images


# --------------------------------------------------------------------------- #
# Feature Extraction.                                                         #
# --------------------------------------------------------------------------- #

def VectorizeFacts(facts):
  facts = facts.map(
      lambda l: " ".join([NONWORD_RE.sub("", elt) for elt in l]))
  vectorizer = CountVectorizer(
      min_df=0.01, max_df=0.9, analyzer="word")
  return vectorizer, vectorizer.fit_transform(facts).toarray()
