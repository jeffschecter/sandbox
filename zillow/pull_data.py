#!/usr/bin/env python

import os

import zillow


CITIES = """Aberdeen
Advance
Ahoskie
Alamance
Albemarle
Alliance
Altamahaw
Andrews
Angier
Ansonville
Apex
Aquadale
Arapahoe
Archdale
Archer Lodge
Asheboro
Asheville
Ashley Heights
Askewville
Atkinson
Atlantic
Atlantic Beach
Aulander
Aurora
Autryville
Avery Creek
Avon
Ayden
Badin
Bailey
Bakersville
Bald Head Island
Balfour
Banner Elk
Barker Heights
Barker Ten Mile
Bath
Bayboro
Bayshore
Bayview
Bear Grass
Beaufort
Beech Mountain
Belhaven
Bell Arthur
Belmont
Belville
Belvoir
Belwood
Bennett
Benson
Bent Creek
Bermuda Run
Bessemer City
Bethania
Bethel
Bethlehem
Beulaville
Biltmore Forest
Biscoe
Black Creek
Black Mountain
Bladenboro
Blowing Rock
Blue Clay Farms
Boardman
Bogue
Boiling Spring Lakes
Boiling Springs
Bolivia
Bolton
Bonnetsville
Boone
Boonville
Bostic
Bowmore
Brevard
Brices Creek
Bridgeton
Broad Creek
Broadway
Brogden
Brookford
Brunswick
Bryson City
Buies Creek
Bunn
Bunnlevel
Burgaw
Burlington
Burnsville
Butner
Butters
Buxton
Cajah's Mountain
Calabash
Calypso
Camden
Cameron
Candor
Canton
Cape Carteret
Caroleen
Carolina Beach
Carolina Shores
Carrboro
Carthage
Cary
Casar
Cashiers
Castalia
Castle Hayne
Caswell Beach
Catawba
Cedar Point
Cedar Rock
Centerville
Cerro Gordo
Chadbourn
Chapel Hill
Charlotte
Cherokee
Cherryville
Chimney Rock Village
China Grove
Chocowinity
Claremont
Clarkton
Clayton
Clemmons
Cleveland
Cliffside
Clinton
Clyde
Coats
Cofield
Coinjock
Colerain
Columbia
Columbus
Como
Concord
Conetoe
Connelly Springs
Conover
Conway
Cooleemee
Cordova
Cornelius
Cove City
Cove Creek
Cramerton
Creedmoor
Creswell
Cricket
Crossnore
Cullowhee
Dallas
Dana
Danbury
Davidson
Davis
Deercroft
Delco
Dellview
Delway
Denton
Denver
Dillsboro
Dobbins Heights
Dobson
Dortches
Dover
Drexel
Dublin
Duck
Dundarrach
Dunn
Durham
Earl
East Arcadia
East Bend
East Flat Rock
East Laurinburg
East Rockingham
East Spencer
Eastover
Eden
Edenton
Edneyville
Efland
Elizabeth City
Elizabethtown
Elk Park
Elkin
Ellenboro
Ellerbe
Elm City
Elon
Elrod
Elroy
Emerald Isle
Enfield
Engelhard
Enochville
Erwin
Etowah
Eureka
Everetts
Evergreen
Fair Bluff
Fairfield
Fairfield Harbour
Fairmont
Fairplains
Fairview
Fairview town
Faison
Faith
Falcon
Falkland
Fallston
Farmville
Fayetteville
Fearrington Village
Five Points
Flat Rock
Flat Rock CDP
Fletcher
Forest City
Forest Hills
Forest Oaks
Foscoe
Fountain
Four Oaks
Foxfire
Franklin
Franklinton
Franklinville
Fremont
Frisco
Fruitland
Fuquay-Varina
Gamewell
Garland
Garner
Garysburg
Gaston
Gastonia
Gatesville
Germanton
Gerton
Gibson
Gibsonville
Glen Alpine
Glen Raven
Glenville
Gloucester
Godwin
Goldsboro
Goldston
Gorman
Graham
Grandfather
Granite Falls
Granite Quarry
Grantsboro
Green Level
Greenevers
Greensboro
Greenville
Grifton
Grimesland
Grover
Gulf
Half Moon
Halifax
Hallsboro
Hamilton
Hamlet
Hampstead
Harkers Island
Harmony
Harrells
Harrellsville
Harrisburg
Hassell
Hatteras
Havelock
Haw River
Hayesville
Hays
Hemby Bridge
Henderson
Hendersonville
Henrietta
Hertford
Hickory
Hiddenite
High Point
High Shoals
Highlands
Hightsville
Hildebran
Hillsborough
Hobgood
Hobucken
Hoffman
Holden Beach
Hollister
Holly Ridge
Holly Springs
Hookerton
Hoopers Creek
Hope Mills
Horse Shoe
Hot Springs
Hudson
Huntersville
Icard
Indian Beach
Indian Trail
Ingold
Iron Station
Ivanhoe
JAARS
Jackson
Jackson Heights
Jacksonville
James City
Jamestown
Jamesville
Jefferson
Jonesville
Kannapolis
Keener
Kelford
Kelly
Kenansville
Kenly
Kernersville
Kill Devil Hills
King
Kings Grant
Kings Mountain
Kingstown
Kinston
Kittrell
Kitty Hawk
Knightdale
Kure Beach
La Grange
Lake Junaluska
Lake Lure
Lake Norman of Catawba
Lake Park
Lake Royale
Lake Santeetlah
Lake Waccamaw
Landis
Lansing
Lasker
Lattimore
Laurel Hill
Laurel Park
Laurinburg
Lawndale
Leggett
Leland
Lenoir
Lewiston Woodville
Lewisville
Lexington
Liberty
Light Oak
Lilesville
Lillington
Lincolnton
Linden
Littleton
Locust
Long View
Louisburg
Love Valley
Lowell
Lowesville
Lowgap
Lucama
Lumber Bridge
Lumberton
Macclesfield
Macon
Madison
Maggie Valley
Magnolia
Maiden
Mamers
Manns Harbor
Manteo
Mar-Mac
Marble
Marietta
Marion
Mars Hill
Marshall
Marshallberg
Marshville
Marvin
Matthews
Maury
Maxton
Mayodan
Maysville
McAdenville
McDonald
McFarlan
McLeansville
Mebane
Mesic
Micro
Middleburg
Middlesex
Midland
Midway
Millers Creek
Millingport
Mills River
Milton
Mineral Springs
Minnesott Beach
Mint Hill
Misenheimer
Mocksville
Momeyer
Moncure
Monroe
Montreat
Mooresboro
Mooresville
Moravian Falls
Morehead City
Morganton
Morrisville
Morven
Mount Airy
Mount Gilead
Mount Holly
Mount Olive
Mount Pleasant
Mountain Home
Mountain View
Moyock
Mulberry
Murfreesboro
Murphy
Murraysville
Myrtle Grove
Nags Head
Nashville
Navassa
Neuse Forest
New Bern
New London
Newland
Newport
Newton
Newton Grove
Norlina
Norman
North Topsail Beach
North Wilkesboro
Northchase
Northlakes
Northwest
Norwood
Oak City
Oak Island
Oak Ridge
Oakboro
Ocean Isle Beach
Ocracoke
Ogden
Old Fort
Old Hundred
Oriental
Orrum
Ossipee
Oxford
Pantego
Parkton
Parmele
Patterson Springs
Peachland
Peletier
Pembroke
Pikeville
Pilot Mountain
Pine Knoll Shores
Pine Level
Pinebluff
Pinehurst
Pinetops
Pinetown
Pineville
Piney Green
Pink Hill
Pinnacle
Pittsboro
Plain View
Pleasant Garden
Pleasant Hill
Plymouth
Polkton
Polkville
Pollocksville
Porters Neck
Potters Hill
Powellsville
Princeton
Princeville
Proctorville
Prospect
Pumpkin Center
Raeford
Raemon
Raleigh
Ramseur
Randleman
Ranlo
Raynham
Red Cross
Red Oak
Red Springs
Reidsville
Rennert
Rex
Rhodhiss
Rich Square
Richfield
Richlands
Riegelwood
River Bend
River Road
Roanoke Rapids
Robbins
Robbinsville
Robersonville
Rockfish
Rockingham
Rockwell
Rocky Mount
Rocky Point
Rodanthe
Rolesville
Ronda
Roper
Rose Hill
Roseboro
Rosman
Rougemont
Rowland
Roxboro
Roxobel
Royal Pines
Ruffin
Rural Hall
Ruth
Rutherford College
Rutherfordton
Salem
Salemburg
Salisbury
Saluda
Salvo
Sandy Creek
Sandyfield
Sanford
Saratoga
Sawmills
Saxapahaw
Scotch Meadows
Scotland Neck
Sea Breeze
Seaboard
Seagrove
Sedalia
Selma
Seven Devils
Seven Lakes
Seven Springs
Severn
Shallotte
Shannon
Sharpsburg
Shelby
Siler City
Silver City
Silver Lake
Simpson
Sims
Skippers Corner
Smithfield
Sneads Ferry
Snow Hill
South Henderson
South Mills
South Rosemary
South Weldon
Southern Pines
Southern Shores
Southmont
Southport
Sparta
Speed
Spencer
Spencer Mountain
Spindale
Spivey's Corner
Spring Hope
Spring Lake
Spruce Pine
St. Helena
St. James
St. Pauls
St. Stephens
Staley
Stallings
Stanfield
Stanley
Stantonsburg
Star
Statesville
Stedman
Stem
Stokes
Stokesdale
Stoneville
Stonewall
Stony Point
Stovall
Sugar Mountain
Summerfield
Sunbury
Sunset Beach
Surf City
Swan Quarter
Swannanoa
Swansboro
Swepsonville
Sylva
Tabor City
Tar Heel
Tarboro
Taylorsville
Taylortown
Teachey
Thomasville
Toast
Tobaccoville
Topsail Beach
Trent Woods
Trenton
Trinity
Troutman
Troy
Tryon
Turkey
Tyro
Unionville
Valdese
Valle Crucis
Valley Hill
Vanceboro
Vandemere
Vander
Vann Crossroads
Varnamtown
Vass
Waco
Wade
Wadesboro
Wagram
Wake Forest
Wakulla
Walkertown
Wallace
Wallburg
Walnut Cove
Walnut Creek
Walstonburg
Wanchese
Warrenton
Warsaw
Washington
Washington Park
Watha
Waves
Waxhaw
Waynesville
Weaverville
Webster
Weddington
Welcome
Weldon
Wendell
Wentworth
Wesley Chapel
West Canton
West Jefferson
West Marion
Westport
Whispering Pines
Whitakers
White Lake
White Oak
White Plains
Whiteville
Whitsett
Wilkesboro
Williamston
Wilmington
Wilson
Wilson's Mills
Windsor
Winfall
Wingate
Winston-Salem
Winterville
Winton
Woodfin
Woodland
Woodlawn
Wrightsboro
Wrightsville Beach
Yadkinville
Yanceyville
Youngsville
Zebulon""".split("\n")


if __name__ == "__main__":
  for city in CITIES:
    print "Scanning", city
    zillow.ProcessRegion(city, "NC", "data/")
    zillow.PostprocessImages("data/")
