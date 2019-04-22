import numpy as np
import pandas as pd


# Calculate the Prob. of class:cls


def P(data, cls_val, cls_name="class"):
    count = 0.0
    for e in data:
        if e[cls_name] == cls_val:
            count += 1
    return count/len(data)


def PT(data, cls_val, attr_name, attr_val, cls_name="class"):
    count1 = 0.0
    count2 = 0.0
    for e in data:
        if e[cls_name] == cls_val:
            count1 += 1
            if e[attr_name] == attr_val:
                count2 += 1
    return count2/count1


# Calculate the NB
def NB(data, test, cls_y, cls_n):
    PY = P(data, cls_y)
    PN = P(data, cls_n)
    print('The probability of play or not:', PY, 'vs.', PN)
    for key, val in list(test.items()):
        PY *= PT(data, cls_y, key, val)
        PN *= PT(data, cls_n, key, val)
        print(key, val, '-->play or not:-->', PY, PN)
    return {cls_y: PY, cls_n: PN}


if __name__ == '__main__':
    data = [
        {"outlook": "sunny", "temp": "hot",
            "humidity": "high", "wind": "weak", "class": "no"},
        {"outlook": "sunny", "temp": "hot", "humidity": "high",
            "wind": "strong", "class": "no"},
        {"outlook": "overcast", "temp": "hot",
            "humidity": "high", "wind": "weak", "class": "yes"},
        {"outlook": "rain", "temp": "mild", "humidity": "high",
            "wind": "weak", "class": "yes"},
        {"outlook": "rain", "temp": "cool", "humidity": "normal",
            "wind": "weak", "class": "yes"},
        {"outlook": "rain", "temp": "cool", "humidity": "normal",
            "wind": "strong", "class": "no"},
        {"outlook": "overcast", "temp": "cool",
            "humidity": "normal", "wind": "strong", "class": "yes"},
        {"outlook": "sunny", "temp": "mild",
            "humidity": "high", "wind": "weak", "class": "no"},
        {"outlook": "sunny", "temp": "cool",
            "humidity": "normal", "wind": "weak", "class": "yes"},
        {"outlook": "rain", "temp": "mild", "humidity": "normal",
            "wind": "weak", "class": "yes"},
        {"outlook": "sunny", "temp": "mild", "humidity": "normal",
            "wind": "strong", "class": "yes"},
        {"outlook": "overcast", "temp": "mild",
            "humidity": "high", "wind": "strong", "class": "yes"},
        {"outlook": "overcast", "temp": "hot",
            "humidity": "normal", "wind": "weak", "class": "yes"},
        {"outlook": "rain", "temp": "mild", "humidity": "high", "wind": "strong", "class": "no"}]
    pd.DataFrame(data)
    # The probability of play or not
    PY, PN = P(data, "yes"), P(data, "no")

    # Calculate the Prob(attr|cls)
    test = {"outlook": "sunny", "temp": "cool",
            "humidity": "high", "wind": "strong"}
    # The conditional probability of play or not
    PT(data, "yes", "outlook", "sunny"), PT(data, "no", "outlook", "sunny")

    # calculate
    NB(data, {"outlook": "sunny", "temp": "hot",
              "humidity": "high", "wind": "strong"}, "yes", "no")
