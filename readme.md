# Soft computing

Implementation fo various soft computing techniques taught in my undergraduate soft computing elective course.

---

## Fuzzy logic engine

Implementation of fuzzy logic concepts. Engine is implemented completely in the [Fuzzy.py](./FuzzyEngine/Fuzzy.py) file. Implemented functionality include :

1) Class for linguistic variables
2) Class for fuzzy sets
3) Class for fuzzy relations
4) Dynamic relation from linguistic variables to fuzzy sets allowing plotting different sets defined on the same variable
5) Ability to get membership function from a list or generated using a function
6) Common operations on fuzzy sets and relations
   - Inverting
   - Union
   - Intersection
   - Composition of relations
   - Cartesion product of sets
   - Alpha cuts on sets
   - Defuzzification methods on sets
     - Earliest maxima
     - Centroid
     - Mean of maxima
7) Implementation of zadehs extension principle for creating new fuzzy sets and linguistic variables ([FuzzySet.extend()](./FuzzyEngine/Fuzzy.py#103))

---

## Unit testing

Unit tests test most functionality, but is not extensive enough to conver all cases.
