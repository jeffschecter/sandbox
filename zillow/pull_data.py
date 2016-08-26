#!/usr/bin/env python

import os
import sys

import gflags

import pandas as pd

import zillow


FLAGS = gflags.FLAGS
gflags.DEFINE_string(
    "savedir", "data/",
    "Directory to save and load data.")
gflags.DEFINE_list(
    "states", None,
    "Only process zipcodes in these states.")
gflags.DEFINE_list(
    "cities", None,
    "Only process zipcodes in these cities.")


def LoadZipcodes(savedir, cities=None, states=None):
  zips = pd.read_csv(os.path.join(savedir, "zipcodes.csv"))
  zips["City"] = zips.City.map(lambda s: s.title())
  zips = zips[zips.ZipCodeType == "STANDARD"]
  if cities is not None:
    cities = set(cities)
    zips = zips[zips.City.map(lambda c: c in cities)]
  if states is not None:
    states = set(states)
    zips = zips[zips.State.map(lambda s: s in states)]
  return zip(zips.City, zips.State, zips.Zipcode)


def LoadAlreadyProcessed(savedir):
  tab_dir = os.path.join(savedir, "tabular")
  if os.path.exists(tab_dir):
    handles = os.listdir(tab_dir)
    handles = [handle.partition(".")[0] for handle in handles]
    parts = [handle.split("_") for handle in handles]
    parts = [p for p in parts if len(p) == 3]
    return set([(city, state, int(zipcode)) for city, state, zipcode in parts])
  else:
    return set([])


def main(savedir, cities=None, states=None):
  region_triples = LoadZipcodes(savedir, cities=cities, states=states)
  already_processed = LoadAlreadyProcessed(savedir)
  for city, state, zipcode in region_triples:
    if (city, state, zipcode) in already_processed:
      print "-- Skipping {c}, {s} zip {z}".format(c=city, s=state, z=zipcode)
    else:
      print "++ Scanning {c}, {s} zip {z}".format(c=city, s=state, z=zipcode)
      sys.stdout.flush()
      zillow.ProcessRegion(city, state, savedir, zipcode=zipcode)
      zillow.PostprocessImages(savedir)


if __name__ == "__main__":
  FLAGS(sys.argv)
  main(FLAGS.savedir, cities=FLAGS.cities, states=FLAGS.states)
