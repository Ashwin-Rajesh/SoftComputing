#! /usr/bin/env python

import numpy as np 
import matplotlib.pyplot as plt 

# Class for representing a linguistic variable
class LinguisticVariable:
    def __init__(self, inp : list, name : str):
        self.values = list(inp)
        self.name   = str(name)

        self.sets = {}

    # Create a fuzzy set by mapping the elements using the passed function to get the membership function
    def make_set_fn(self, mem_fun, name):
        self.sets[name] = FuzzySet(self, [mem_fun(i) for i in self.values], name)

        return self.sets[name]

    # Create a fuzzy set using the given list as the membership function 
    def make_set(self, mem_fun, name):
        self.sets[name] = FuzzySet(self, mem_fun, name)

        return self.sets[name]

    # Add a fuzzy set to the internal dictionary of the object
    def add_set(self, fuzzy_set):
        self.sets[fuzzy_set.name] = fuzzy_set

    def __str__(self):
        return self.name + " : " + str(self.values)

    # Plot and view the different fuzzy sets
    def view(self, show=True):
        for s in self.sets.values():
            s.view(False)
        
        if(show):
            plt.show()

    # Create a linguistic variable that can be used for apppying Zadehs extension principle
    def extend(self, function, out_name):
        inp = self.values

        out = np.unique([function(i) for i in inp])

        return LinguisticVariable(out, out_name)               

# Class for representing fuzzy sets
class FuzzySet:
    def __init__(self, ling_var : LinguisticVariable, mem_fun : list, name : str):
        self.name       = str(name)
        self.ling_var   = ling_var
        
        inp = ling_var.values

        assert len(inp) == len(mem_fun) , "Length of inputs and membership function must be equal"
        assert max(mem_fun) <= 1 , f"The membership value cannot be greater than 1. Here, maximum is : {max(mem_fun)}"

        self.dict = {inp[i]:mem_fun[i] for i in range(len(inp))}

    # Create a linguistic variable and fuzzy set from a dictionary 
    @staticmethod
    def make(dict, name):
        u = LinguisticVariable(list(dict.keys()), name)
        s = FuzzySet(u, list(dict.values()), name)

        return u, s

    # Plot and vew the membership function
    def view(self, show = True):
        plt.plot(self.dict.keys(), self.dict.values(), label = self.name)
        
        plt.xlabel(self.ling_var.name)
        plt.ylabel("Membership function")

        plt.legend()

        if(show):
            plt.show()

    # Alpha cur of a fuzzy set. Returns a python list with elements which have u > alpha
    def alpha_cut(self, alpha):
        return [k for k in self.dict if self.dict[k] >= alpha]

    # Inversion or not of a fuzzy set (u' = 1 - u)
    def __invert__(self):
        return FuzzySet(self.ling_var, [1-m for m in self.dict.values()], 'not ' + self.name)

    # Intersection of two fuzzy sets (u' = min(u1, u2))
    def __and__(self, that):
        assert self.ling_var == that.ling_var , "LinguisticVariables are not the same; Cannot do and operation!"

        return FuzzySet(self.ling_var, [min(self.dict[k], that.dict[k]) for k in self.dict], self.name + " and " + that.name)

    # Union of two fuzzy sets (u' = max(u1, u2))
    def __or__(self, that):
        assert self.ling_var == that.ling_var , "LinguisticVariables are not the same; Cannot do or operation!"

        return FuzzySet(self.ling_var, [max(self.dict[k], that.dict[k]) for k in self.dict], self.name + " or " + that.name)

    # Multiplication is assumed to be with a relation
    def __mul__(self, that):
        assert type(that) is FuzzyRelation , "The 2nd operand of multiplication must be a relation"
        assert that.src == self.ling_var , "The relation and set are not compatible"

        mem_fun = np.asarray(list(self.dict.values())).reshape(1, -1).dot(that.arr)

        return FuzzySet(self.ling_var, list(mem_fun.flatten()), that.name + " " + self.name)

    # Apply Zadehs extension principle on the fuzzy set
    # The extension principle states that for multiple inputs mapping to same
    # Output (many-to-one), we can create a fuzzy set from a fuzzy set corresponding
    # to the inputs by having the elements be mappings of elements of inputs and the
    # membership function be maximum of memebrship function of all inputs that mapped
    # to the same output
    #
    # Example : applying function x % 2 on the set {0.1/1, 0.2/1.5, 0.5/3, 0.1/3.5}
    # Will give {max(0.1, 0.5)/1, max(0.2, 0.1)/1.5}
    #
    # inp   | out    | membership
    # 1    => 1         (0.1)
    # 1.5  => 1.5    -  (0.2)
    # 3    => 1         (0.5)
    # 5.5  => 1.5    -  (0.1)
    #
    def extend(self, function, out_name, out_ling_var = None):
        inp = self.ling_var.values

        out = {}

        for i in inp:
            o = function(i)

            if o in out:
                out[o].append(i)
            else:
                out[o] = [i]
        
        if(out_ling_var == None):
            out_values = list(out.keys())
            out_values.sort()
            out_ling_var = LinguisticVariable(out_values, out_name)
        else:
            out_values = out_ling_var.values.sort()
            out_values.sort()

        out_mem    = list([max([self.dict[i] for i in out[k]]) for k in out_values])

        out_set = FuzzySet(out_ling_var, out_mem, out_name)

        out_ling_var.add_set(out_set)

        return out_ling_var, out_set

    # Cartesion product with another fuzzy set. Returns a fuzzy relation.
    def cartesian_product(self, that):
        arr = np.zeros((len(self.dict.values()), len(that.dict.values())))

        for i, k1 in enumerate(self.dict.keys()):
            for j, k2 in enumerate(that.dict.keys()):
                arr[i, j] = min(self.dict[k1], that.dict[k2])

        return FuzzyRelation(self.ling_var, that.ling_var, arr, self.name + " x " + that.name)

    # Defuzzification technique 1 - Earliest maximum membership value
    def max_membership(self):
        max_val = None
        max = 0

        for key in self.dict:
            if(self.dict[key] > max):
                max_val = key
                max = self.dict[key]

        return max_val

    # Defuzzification technique 2 - Centroid of membership v/s input
    def centroid(self):
        sum1 = 0
        sum2 = 0

        prev = None

        for key in self.dict:
            if(prev == None):
                prev = key
            else:
                x  = key
                dx = x - prev
                m  = self.dict[key]
                
                sum1 = sum1 + (m * x * dx)
                sum2 = sum2 + (m * dx)

        return sum1 / sum2

    # Defuzzification technique 3 - Mean of maximum
    def mean_max(self, that):
        max_sum     = None
        max         = 0
        max_count   = 0

        for key in self.dict:
            if(self.dict[key] > max):
                max_sum     = key
                max_count   = 1
                max         = self.dict[key]
            elif(self.dict[key] == max):
                max_count   = max_count + 1
                max_sum     = max_sum + key

        return max_sum / max_count

# Class for representing fuzzy relations
class FuzzyRelation:
    def __init__(self, src : LinguisticVariable, tgt : LinguisticVariable, arr : np.ndarray, name : str):
        self.src = src
        self.tgt = tgt

        self.name = name

        assert arr.shape[0] == len(src.values) , "The number of rows is not correct"
        assert arr.shape[1] == len(tgt.values) , "The number of columns is not correct"
        self.arr = arr

    # Max-min composition of two fuzzy relations
    def max_min(self, that):
        assert self.tgt == that.src , "The linguistic variabless used are not the same"

        arr = np.zeros((len(self.src.values), len(that.tgt.values)))

        for i in range(len(self.src.values)):
            for j in range(len(that.tgt.values)):
                arr[i][j] = max([min(self.arr[i, k], that.arr[k, j]) for k in range(len(self.tgt.values))])

        return FuzzyRelation(self.src, that.tgt, arr, self.name + " composition " + that.name)

    # Max-product composition of two fuzzy relations
    def max_product(self, that):
        assert self.tgt == that.src , "The linguistic variables used are not the same"

        arr = np.zeros((len(self.src.values), len(that.tgt.values)))

        for i in range(len(self.src.values)):
            for j in range(len(that.tgt.values)):
                arr[i][j] = max([self.arr[i, k] * that.arr[k, j] for k in range(len(self.tgt.values))])

        return FuzzyRelation(self.src, that.tgt, arr, self.name + " composition " + that.name)

    # Multiplication fo two fuzzy relations gives max-min composition
    def __mul__(self, that):
        assert self.tgt == that.src , "The linguistic variables used are not the same"

        return self.max_min(that)

    # Inverse of fuzzy relation (u' = 1 - u)
    def __inv__(self):
        return FuzzyRelation(self.src, self.tgt, np.ones_like(self.arr) - self.arr, "not " + self.name)

    # Intersection of fuzzy relations (u' = min(u1, u2))
    def __and__(self, that):
        assert self.tgt == that.tgt & self.src == that.src , "The linguistic variables do not match"

        return FuzzyRelation(self.src, self.tgt, np.minimum(self.arr, that.arr), self.name + " and " + that.name)

    # Union of fuzzy relations (u' = max(u1, u2))
    def __or__(self, that):
        assert self.tgt == that.tgt & self.src == that.src , "The linguistic variables do not match"

        return FuzzyRelation(self.src, self.tgt, np.maximum(self.arr, that.arr), self.name + " or " + that.name)

service = LinguisticVariable(range(0, 11), 'Service')
service.make_set([1  , 0.8, 0.6, 0.4, 0.2, 0  , 0  , 0  , 0  , 0  , 0], 'Poor')
service.make_set([0  , 0.2, 0.4, 0.6, 0.8, 1  , 0.8, 0.6, 0.4, 0.2, 0], 'Medium')
service.make_set([0  , 0  , 0  , 0  , 0  , 0  , 0.2, 0.4, 0.6, 0.8, 1], 'Good')
    
food_quality = LinguisticVariable(range(0, 11), 'Food Quality')
food_quality.make_set([1  , 0.8, 0.6, 0.4, 0.2, 0  , 0  , 0  , 0  , 0  , 0], 'Poor')
food_quality.make_set([0  , 0.2, 0.4, 0.6, 0.8, 1  , 0.8, 0.6, 0.4, 0.2, 0], 'Medium')
food_quality.make_set([0  , 0  , 0  , 0  , 0  , 0  , 0.2, 0.4, 0.6, 0.8, 1], 'Good')

def test_fun(inp):
    return (inp % 7)

if __name__ == "__main__":
    # s = service()
    # s.view()

    # for i in range(11):
    #     print(s.get_mem(i))

    service.view()
    food_quality.view()

