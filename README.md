# Tripartite model study

## Abstract

Real-world networks share non-trivial properties, such as a skewed degree-distribution, a low global
density, a high local density, etc. However, existing random graph models do not capture all the properties
of real world networks at the same time. In particular, the bipartite configuration model achieves in
preserving a degree distribution close to the original network one, but fails in generating graphs that
keeps the overlapping structures.

F. Tarissan & L. Tabourier proposed a random model [1][2] that relies on maximal bicliques to preserve
overlaps in bipartite networks, by exploiting the tripartite version of the configuration model. The purpose of this study is to tackle the realistic random graph model problem, by evaluating the relevance of the tripartite model as a possible generalized answer to this issue. We will both validate and further current knowledge of this model, by examining other characteristics of real-world network.

Main classes:
- UnipartiteGraph.py
- BipartiteGraph.py
- Graph.py


## Bibliography

[1] Fabien Tarissan and Lionel Tabourier. A random model that relies on maximal bicliques to preserve
the overlaps in bipartite networks. In 8th International Conference on Complex Networks and their
Applications, Lisbonne, Portugal, December 2019.

[2] Émilie Coupechoux and Fabien Tarissan. Un modèle pour les graphes bipartis aléatoires avec
redondance. In 4ème Journées Modèles et l’Analyse des Réseaux : Approches Mathématiques et
Informatique (MARAMI’13), Saint-Etienne, France, October 2013.

[3] Peter Damaschke. Enumerating maximal bicliques in bipartite graphs with favorable degree
sequences. Information Processing Letters, 114:317–321, June 2014.

[4] Enver Kayaaslan. On enumerating all maximal bicliques of bipartite graphs. pages 105–108, 01 2010.
