"""Binary sequential Rosenblatt perceptron, adapted conceptually from nnci-course-python.

Outputs hard class decisions, NOT calibrated AKI probabilities.
"""
from dataclasses import dataclass
from math import isfinite
from random import Random

@dataclass
class Perceptron:
    weights: list[float]
    bias: float=0.0
    threshold: float=0.0

    def score(self,x):
        if len(x)!=len(self.weights) or not all(isfinite(float(v)) for v in x):
            raise ValueError('Invalid feature vector')
        return sum(w*float(v) for w,v in zip(self.weights,x))+self.bias

    def predict(self,x):
        return 1 if self.score(x)>=0 else -1


def train(X,y,*,epochs=100,learning_rate=None,threshold=0.0,seed=2026,shuffle=True):
    if not X or len(X)!=len(y) or not X[0]:
        raise ValueError('Training requires nonempty aligned examples')
    n=len(X[0]); rate=(1.0/n) if learning_rate is None else float(learning_rate)
    if epochs<1 or not isfinite(rate) or rate<=0 or not isfinite(threshold) or threshold<0:
        raise ValueError('Invalid training hyperparameters')
    for x,label in zip(X,y):
        if len(x)!=n or label not in (-1,1) or any(not isfinite(float(v)) for v in x):
            raise ValueError('Invalid training data')
    model=Perceptron([0.0]*n,0.0,threshold)
    rng=Random(seed); updates=0; converged=False; history=[]
    for epoch in range(1,epochs+1):
        order=list(range(len(y)))
        if shuffle: rng.shuffle(order)
        for i in order:
            margin=y[i]*model.score(X[i])
            if margin<=threshold:
                model.weights=[w+rate*y[i]*float(v) for w,v in zip(model.weights,X[i])]
                model.bias+=rate*y[i]
                updates+=1
        margins=[label*model.score(x) for x,label in zip(X,y)]
        violations=sum(m<=threshold for m in margins)
        history.append({'epoch':epoch,'updates':updates,'updates_epoch':updates-(history[-1]['updates'] if history else 0),
                        'violations':violations,'training_accuracy':sum(m>0 for m in margins)/len(margins),
                        'mean_signed_margin':sum(margins)/len(margins),
                        'perceptron_loss':sum(max(0.0,-m) for m in margins)/len(margins)})
        # Strict margin test, matching Rosenblatt convergence semantics.
        if violations==0:
            converged=True;break
    return model,{'epochs_run':epoch,'updates':updates,'converged':converged,
                  'history':history,'max_epochs':epochs,'learning_rate':rate,'threshold':threshold,'seed':seed}
