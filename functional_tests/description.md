# Dataset description

4-sources dataset with follwing source attributes:

* Brand and Marca: Attributes with different names but whose values are homogeneous 
(the same for pages of the same entity).
* Dimensions and misure (source1, source3), complex attributes split in 'width', 'depth' and 'height' 
in source2 and source4. Splitted attributes have similar domains
* Sensor, color, internal memory (source1 and source3), joined in source2/features and source4/other (just color and 
internal memory). Atomic attributes have different domains.
* Battery chemistry, battery model (source1, source2) are atomic. Battery in source3 and source4 are heterogeneous 
attributes, sometimes provide model, sometimes chemistry and sometimes both. 
Some of the values in battery are impossible to interpret without a knowledge of the domain, as they are absent in 
battery chemistry and model source attributes. This is a lesser problem for alignment (as long as there are other 
evidences) but may be a problem for information extraction.
* Resolution/risoluzione (source1, source3): same values but different approximation (14.1 vs 14)
* Languages/lingue (source1, source3): different order of tokens + 1 additional token 
(ok as long as jaccard similarity is > 0.9).
  * Note that for JS numeric values count double, we do not test them here as it is a variable heuristic.