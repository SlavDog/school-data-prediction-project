from sklearn import linear_model

degrees = []
preparation = []
math_score = []

fhand = open("school-data-science-project\StudentsPerformance.csv")

for line in fhand:
    LineArr = line.split(",")
    degrees.append(LineArr[2])
    preparation.append(LineArr[4])
    math_score.append(float(LineArr[5].strip('\"')))

def Degrees_filter(item):
    if item == '"high school"' or item == '"some high school"':
        return 0
    return 1

def Prep_filter(item):
    if item == '"none"':
        return 0
    return 1

degrees_bool = list(map(Degrees_filter, degrees))
prep_bool = list(map(Prep_filter, preparation))
print(math_score)
X = [degrees_bool,prep_bool]
y = math_score

logr = linear_model.LogisticRegression()
logr.fit(X,y)


      