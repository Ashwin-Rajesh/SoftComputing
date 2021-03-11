# Fuzzy Logic and Fuzzy controller

Implementation of fuzzy logic concepts and a fuzzy controller. Done for reinforcing concepts learned as a part of EC360 Soft Computing elective at Department of ECE, GEC Thrissur, under the guidance of Prof Gopi C.

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
