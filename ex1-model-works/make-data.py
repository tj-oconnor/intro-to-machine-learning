from random import random, uniform
from pprint import pprint

TOTAL_STATS = 25


ships = []
ships.append('Starfleet Assault Cruiser')
ships.append('Starfleet Assault Frigate')
ships.append('Starfleet Fast Scout')
ships.append('Starfleet Heavy Scout')
ships.append('Starfleet Long-Range Scout')
ships.append('Starfleet Fast Attack Ship')
ships.append('Klingon Assault Ship')
ships.append('Klingon Tactical Assault Ship')
ships.append('Klingon Hunter-Killer')
ships.append('Romulan Cutter')
ships.append('Romulan Light Fighter')
ships.append('Romulan Warpshuttle')

ships_labels = {}
ships_rev_labels = {}
for i in range(0, len(ships)):
    ships_labels[i] = ships[i]

for i in range(0, len(ships)):
    ships_rev_labels[ships[i]] = i

stats = {}

for i in range(0, len(ships)):
    ships_stats = []
    for j in range(0, TOTAL_STATS):
        v = uniform(0.0, 100.0)
        z = uniform(0.0, 100.0)*v
        x = round(random()*z, 2)
        y = round(x+random()*z, 2)
        ships_stats.append([x, y])
    stats[i] = ships_stats


for i in range(0, len(ships)):
    for k in range(0, 1000):
        rand_stat = []
        for j in range(0, TOTAL_STATS):
            rand_stat.append(round(uniform(stats[i][j][0], stats[i][j][1]), 2))
        print(', '.join(str(item)
                        for item in rand_stat) + ', ' + ships_labels[i])
