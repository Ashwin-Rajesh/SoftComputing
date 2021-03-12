from Fuzzy import *

def freezing(temp):
    if(temp >= 5):
        return 0
    elif(temp <= -10):
        return 1
    else:
        return (5 - temp)/15

def cold(temp):
    if(temp > 25):
        return 0
    elif(temp < 5):
        return 1
    else:
        return ((temp - 25) / 25)**2

def hot(temp):
    if(temp < 20):
        return 0
    else:
        return min(1, (temp - 20)/25)

def warm(temp):
    if(temp < 15):
        return 0
    else:
        return min(1, (temp - 15)/10)

temp = LinguisticVariable(range(-30, 110), "Temperature")

temp.make_set_fn(freezing,  "Freezing")
temp.make_set_fn(cold,      "Cold")
temp.make_set_fn(warm,      "Warm")
temp.make_set_fn(hot,       "Hot")

R = np.eye(len(temp.values))
R = np.concatenate((np.tile(R[:,0:1], (1, 20)), R[:,:-20]), axis=1)
hotter = FuzzyRelation(temp, temp, R, "20 degrees hotter than")

R = np.eye(len(temp.values))
R = np.concatenate((R[:,20:], np.tile(R[:,-1:], (1, 20))), axis=1)
colder = FuzzyRelation(temp, temp, R, "20 degrees colder than")

if __name__ == "__main__":
    temp.view()

    sets = temp.sets

    sets['Freezing'].view(False)
    (~sets['Freezing']).view()
    
    sets['Cold'].view(False)
    sets['Warm'].view(False)
    (sets['Cold']        & sets['Warm']).view()
    
    sets['Freezing'].view(False)
    sets['Hot'].view(False)
    (~(sets['Freezing'] | sets["Hot"])).view()

    (sets['Cold']*hotter).view(False)
    (sets['Cold']*colder).view(False)
    sets['Cold'].view()

    (sets['Hot']*hotter).view(False)
    (sets['Hot']*colder).view(False)
    sets['Hot'].view()

    abs_hot = sets['Hot'].extend(lambda x : x/2, "Half as hot")[1]
    abs_hot.view()
