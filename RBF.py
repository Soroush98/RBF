import array
import random
import numpy
import math
import time
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from pandas import Series
import pandas
import matplotlib.pyplot as plt
import matplotlib.patches as pc
from deap import algorithms
from deap import base
from deap import benchmarks
from deap import creator
from deap import tools
IND_SIZE = 30
MIN_VALUE = 0
MAX_VALUE = 1
MIN_STRATEGY = 0.5
MAX_STRATEGY = 3
##initialize
m = 10      # number of clusters
n = 2      # input dimention
L = 1500  #Number of train data
generations = 10  #number of generations
train_file = "5clstrain1500.csv"                #train file
classification_test_file = "5clstest5000.csv"  #test for classification
regression_test_file = "regdata2000.csv"        #test for regression
##end of Initialize
X = []
Y = []

def evaluate(indivisual):
    W = []
    G = []
    V = []
    Ybar =[]
    if (type(Y[0]) != numpy.float64):
        Class = len(Y[0])
    else:
        Class = 1
    for i in range(n):
        V.append(0)
    for i in range(L):
        temp = []
        for j in range(m):
            for s in range(n):
                V[s] = indivisual[j*(n+1)+s]
            t = gfunc(X[i],V,indivisual[j*(n+1)+n])
            temp.append(t)

        G.append(temp)
    Failure = 0
    Gt = numpy.matrix.transpose(numpy.array(G))
    G1 = numpy.matmul(Gt,G)
    try:
        Ginv = numpy.linalg.inv(G1)
    except numpy.linalg.LinAlgError:
        print("Uninversable Matrix ..")
        return Failure*1000,
    G2 = numpy.matmul(Ginv,Gt)
    W = numpy.matmul(G2,Y)
    Ybar = numpy.matmul(G,W)
    Yfailure = []
    if (Class == 1):
        for k in range(L):
            Yfailure.append (Ybar[k] - Y[k])
        Yft = numpy.matrix.transpose(numpy.array(Yfailure))
        Failure = numpy.matmul(Yft,Yfailure)
        return Failure/2,
    else:
        for k in range(L):
            ymax = -1000000
            ybarmax = -1000000
            indy = 0
            indybar = 0
            for x1 in range(Class):
                if (Y[k][x1] > ymax):
                    ymax = Y[k][x1]
                    indy = x1
                if (Ybar[k][x1]> ybarmax):
                    ybarmax = Ybar[k][x1]
                    indybar = x1
            if (abs(indy - indybar) > 0 ):
                Failure = Failure + 1
        return Failure/(L),


def generateES(icls, scls, size, imin, imax, smin, smax):
    ind = icls(random.uniform(imin, imax) for _ in range((n+1)*m))
    ind.strategy = scls(random.uniform(smin, smax) for _ in range((n+1)*m))
    return ind

def checkStrategy(minstrategy):
    def decorator(func):
        def wrappper(*args, **kargs):
            children = func(*args, **kargs)
            for child in children:
                for i, s in enumerate(child.strategy):
                    if s < minstrategy:
                        child.strategy[i] = minstrategy
            return children
        return wrappper
    return decorator


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", array.array, typecode="d", fitness=creator.FitnessMin, strategy=None)
creator.create("Strategy", array.array, typecode="d")
toolbox = base.Toolbox()
toolbox.register("individual", generateES, creator.Individual, creator.Strategy,
                 IND_SIZE, MIN_VALUE, MAX_VALUE, MIN_STRATEGY, MAX_STRATEGY)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxESBlend, alpha=0.1)
toolbox.register("mutate", tools.mutESLogNormal, c=1.0, indpb=0.03)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)
toolbox.decorate("mate", checkStrategy(MIN_STRATEGY))
toolbox.decorate("mutate", checkStrategy(MIN_STRATEGY))
def ES():

    random.seed()
    MU, LAMBDA = 10, 100
    pop = toolbox.population(n=MU)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    pop, logbook = algorithms.eaMuCommaLambda(pop, toolbox, mu=MU, lambda_=LAMBDA,
                                              cxpb=0.6, mutpb=0.3, ngen=generations, stats=stats, halloffame=hof)

    return pop, logbook, hof



def gfunc(X,Vi,li):
    XminV = []
    for j in range(n):
        XminV.append(X[j] - Vi[j])
    XminVt = numpy.matrix.transpose(numpy.array(XminV))
    sum = numpy.matmul(XminVt,XminV)

    return numpy.exp(-1*li*sum)

def plot_fitness(logbook):
    x = logbook.select("gen")
    y = logbook.select("avg")
    plt.plot(x,y)
    plt.show()
def test_classification(pop,classes):

    Yp = []
    Xp = []
    F  = []
    TM = []
    Yc = []
    pd = pandas.read_csv(classification_test_file,header=None)
    min_max_scaler = preprocessing.MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(pd)

    for i in range(len(pd)):
        T = []
        for j in range(n):
            T .append(np_scaled[i][j])
        Xp.append(T)
        Yc.append(np_scaled[i][n])
    for x2 in range(len(pd)):
        temp = []
        for x1 in range(len(classes)):
            if (Yc[x2] == classes[x1]):
                temp.append(1)
            else:
                temp.append(-1)
        Yp.append(temp)
    G = []
    V = []
    Wp = []
    Class = len(Yp[0])
    for i in range(n):
        V.append(0)
    for i in range(len(pd)):
        temp = []
        for j in range(m):
            for s in range(n):
                V[s] = pop[0][j*(n+1)+s]
            t = gfunc(Xp[i],V,pop[0][j*(n+1)+n])
            temp.append(t)

        G.append(temp)
    Failure = 0
    Gt = numpy.matrix.transpose(numpy.array(G))
    G1 = numpy.matmul(Gt,G)
    try:
        Ginv = numpy.linalg.inv(G1)
    except numpy.linalg.LinAlgError:
        print("Uninversable Matrix ..")
        return Failure*1000,
    G2 = numpy.matmul(Ginv,Gt)
    Wp = numpy.matmul(G2,Yp)
    ybar = numpy.matmul(G,Wp)
    for k in range(len(pd)):
        ymax = -1000000
        ybarmax = -1000000
        indy = 0
        indybar = 0
        for x1 in range(len(classes)):
            if (Yp[k][x1] > ymax):
                ymax = Yp[k][x1]
                indy = x1
            if (ybar[k][x1]> ybarmax):
                ybarmax = ybar[k][x1]
                indybar = x1
        T1 = []
        T2 = []

        if (abs(indy - indybar) > 0 ):
            T1.append(Xp[k][0])
            T1.append(Xp[k][1])
            F.append(T1)
            Failure = Failure + 1
        else:
            T2.append(Xp[k][0])
            T2.append(Xp[k][1])
            TM.append(T2)

    print("Failure :"+ str(Failure/len(pd))+"%")
    print(Wp)
    print(pop[0])
    return F,TM
def test_regression(pop):
    Yp = []
    Xp = []
    pd = pandas.read_csv(regression_test_file,header=None)
    min_max_scaler = preprocessing.MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(pd)

    for i in range(len(pd)):
        T = []
        for j in range(n):
            T .append(np_scaled[i][j])
        Xp.append(T)
        Yp.append(np_scaled[i][n])
    G = []
    V = []
    Wp = []
    for i in range(n):
        V.append(0)
    for i in range(len(Yp)):
        temp = []
        for j in range(m):
            for s in range(n):
                V[s] = pop[0][j*(n+1)+s]
            t = gfunc(Xp[i],V,pop[0][j*(n+1)+n])
            temp.append(t)

        G.append(temp)
    Failure = 0
    Gt = numpy.matrix.transpose(numpy.array(G))
    G1 = numpy.matmul(Gt,G)
    try:
        Ginv = numpy.linalg.inv(G1)
    except numpy.linalg.LinAlgError:
        print("Uninversable Matrix ..")
        return Failure*1000,
    G2 = numpy.matmul(Ginv,Gt)
    Wp = numpy.matmul(G2,Yp)
    ybar = numpy.matmul(G,Wp)
    Yfailure = []
    for k in range(len(Yp)):
        Yfailure.append (ybar[k] - Yp[k])
    Yft = numpy.matrix.transpose(numpy.array(Yfailure))
    Failure = numpy.matmul(Yft,Yfailure)
    print("Failure :"+ str(Failure))
    print(Wp)
    print(pop[0])
    return ybar,Yp
def plot_regression(ybar,Yp):
    Z = []
    T = []
    for i in range(len(ybar)):
        T.append(i)
    plt.plot(T,Yp,'r.',color='blue')
    plt.plot(T,ybar,'r.',color='red')

    plt.show()


def plot_classification(F,T,pop):
    X1 = []
    Y1 = []
    X2 = []
    Y2 = []
    S = []
    Radius = []

    for i in range(n):
        S.append(0)
    for i in range(len(F)):
        X1.append(F[i][0])
        Y1.append(F[i][1])
    plt.plot(X1,Y1,'r.',color='blue' )
    for i in range(len(T)):
        X2.append(T[i][0])
        Y2.append(T[i][1])
    plt.plot(X2,Y2,'r.',color='green' )
    R1 = []
    R2 = []
    Rx = 0
    Ry = 0
    Rr = 0
    for j in range(m):
        for s in range(n):
            S[s] = pop[0][j*(n+1)+s]
            if (s== 0):
                Rx = S[s]
                R1.append(S[s])
            if (s==1):
                Ry = S[s]
                R2.append(S[s])
        Rr = 1/math.sqrt(abs(pop[0][j*(n+1)+n]))
        circle = plt.Circle((Rx,Ry), Rr,facecolor='none',edgecolor='black')
        ax = plt.gca()
        ax.add_patch(circle)
        plt.axis('scaled')
    plt.plot(R1,R2,'ro',color='red' )
    plt.show()
def RBF():

    pd = pandas.read_csv(train_file,nrows= L,header=None)
    min_max_scaler = preprocessing.MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(pd)
    mode = input("Please Enter a number\n1:Regression\n2:Classification\n")
    if (mode == '1'):
        for i in range(L):
            T = []
            for j in range(n):
                T .append(np_scaled[i][j])
            X.append(T)
            Y.append(np_scaled[i][n])
        start = time.time()
        pop, logbook,c = ES()
        print("Algorithm time :" + str(time.time() - start))
        plot_fitness(logbook)
        y1,y2 = test_regression(pop)
        plot_regression(y1,y2)
    if (mode == '2'):
        Yc = []
        for i in range(L):
            T = []
            for j in range(n):
                T .append(np_scaled[i][j])
            X.append(T)
            Yc.append(np_scaled[i][n])
        classes = []
        for x1 in range(L):
            t = 0
            for x2 in range(len(classes)):
                if (classes[x2] == Yc[x1]):
                    t = 1
            if (t == 0):
                classes.append(Yc[x1])
        c = len(classes)
        print("Classifying into "+str(c)+ " Classes")
        for x2 in range(L):
            temp = []
            for x1 in range(len(classes)):
                if (Yc[x2] == classes[x1]):
                    temp.append(1)
                else:
                    temp.append(-1)
            Y.append(temp)
        start = time.time()
        pop, logbook,k = ES()
        print("Algorithm time :" + str(time.time() - start))
        plot_fitness(logbook)
        F,T = test_classification(pop,classes)
        plot_classification(F,T,pop)
RBF()