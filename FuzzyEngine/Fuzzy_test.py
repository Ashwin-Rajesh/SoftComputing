import unittest
from Fuzzy import *

class LinguisticVariableTest(unittest.TestCase):
    def setUp(self):
        self.u = LinguisticVariable(range(11), "test")
    
    def test_makeSet(self):
        self.u.make_set([1  , 0.8, 0.6, 0.4, 0.2, 0  , 0  , 0  , 0  , 0  , 0], 'Poor')

    def test_makeSetFn(self):
        self.u.make_set_fn(lambda x : 0.5 * (x % 2), 'Mod2')

class FuzzySetTest(unittest.TestCase):
    def setUp(self):
        self.u  = LinguisticVariable([1, 1.5, 2, 2.5, 3], "test")
        self.b1 = self.u.make_set([1, 0.75, 0.3, 0.15, 0], "B1")
        self.b2 = self.u.make_set([1, 0.6, 0.2, 0.1, 0], "B2")

    def test_alphaCut(self):
        self.assertEqual(self.b1.alpha_cut(0.5), [1, 1.5])
        self.assertEqual(self.b2.alpha_cut(0.5), [1, 1.5])

    def test_inv(self):
        self.assertEqual(list((~self.b1).dict.values()), [0, 0.25, 0.7, 0.85, 1])
        self.assertEqual(list((~self.b2).dict.values()), [0, 0.4, 0.8, 0.9, 1])

    def test_or(self):
        self.assertEqual(list((self.b1 | self.b2).dict.values()), [1, 0.75, 0.3, 0.15, 0])

    def test_and(self):
        self.assertEqual(list((self.b1 & self.b2).dict.values()), [1, 0.6, 0.2, 0.1, 0])

class FuzzyRelationTest(unittest.TestCase):
    def test_cartesianProd(self):
        ua, a   = FuzzySet.make({1:0.3, 2:0.7, 3:1},    'A')
        ub, b   = FuzzySet.make({10:0.4, 20:0.9},       'B')

        r = a.cartesian_product(b)

        ans = np.asarray([[0.3, 0.3], [0.4, 0.7], [0.4, 0.9]])
        self.assertTrue(np.allclose(r.arr, ans))

    def test_Composition(self):
        ua = LinguisticVariable([1, 2], 'A')
        ub = LinguisticVariable([10, 20], 'B')
        uc = LinguisticVariable([100, 200, 300], 'C')

        R = FuzzyRelation(ua, ub, np.asarray([[0.6, 0.3], [0.2, 0.9]]))
        S = FuzzyRelation(ub, uc, np.asarray([[1, 0.5, 0.3], [0.8, 0.4, 0.7]]))

        RS1 = R.max_min(S)
        RS2 = R.max_product(S)

        ans1 = np.asarray([[0.6, 0.5, 0.3], [0.8, 0.4, 0.7]])
        ans2 = np.asarray([[0.6, 0.3, 0.21], [0.72, 0.36, 0.63]])

        self.assertTrue(np.allclose(RS1.arr, ans1), str(RS1.arr) + " v/s " + str(ans1))
        self.assertTrue(np.allclose(RS2.arr, ans2), str(RS2.arr) + " v/s " + str(ans2))

if __name__ == "__main__":
    unittest.main()
