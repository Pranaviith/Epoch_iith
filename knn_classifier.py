import numpy as np

data = [[150,7,1,'Apple'],[120,6.5,0,'Banana'],[180,7.5,2,'Orange'],[155,7.2,1,'Apple'],
        [110,6,0,'Banana'],[190,7.8,2,'Orange'],[145,7.1,1,'Apple'],[115,6.3,0,'Banana']]

def prep(data):
    m={'Apple':0,'Banana':1,'Orange':2}
    return np.array([r[:3] for r in data],float),np.array([m[r[3]] for r in data]),{v:k for k,v in m.items()}

def norm(X):
    return (X-X.min(0))/(X.max(0)-X.min(0)+1e-10)

def dist(p1,p2):
    return np.sqrt(((p1-p2)**2).sum())

class KNN:
    def __init__(self,k=3):
        self.k=k
    def fit(self,X,y):
        self.X,self.y=X,y
    def predict(self,X):
        return np.array([np.bincount(self.y[np.argsort([dist(x,t) for t in self.X])[:self.k]]).argmax() for x in X])

def eval(X,y,test,k=[1,3,5],nrm=False):
    if nrm:X,test=norm(X),norm(test)
    e=['Banana','Apple','Orange'];m={0:'Apple',1:'Banana',2:'Orange'}
    for v in k:
        print(f"\nk={v}:")
        knn=KNN(v).fit(X,y)
        p=knn.predict(test)
        for i,q in enumerate(p):print(f"S{i+1}:P={m[q]},E={e[i]}")
        print(f"A:{(p==[1,0,2]).mean():.2f}")

def split(X,y,t=0.25):
    i=np.random.permutation(len(X))
    n=int(t*len(X))
    return X[i[n:]],y[i[n:]],X[i[:n]],y[i[:n]]

if __name__=="__main__":
    X,y,m=prep(data)
    test=np.array([[118,6.2,0],[160,7.3,1],[185,7.7,2]])
    print("No norm:");eval(X,y,test)
    print("\nNorm:");eval(X,y,test,True)
    X1,y1,X2,y2=split(X,y)
    print(f"\nSplit A:{KNN(3).fit(X1,y1).predict(X2)==y2.mean():.2f}")