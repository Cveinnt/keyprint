"""Research kernel: integer strength/bound checks and binary64 proposal arithmetic.

No Fraction operations in the hot path. Exact cross-products check the actual
normalized categorical distribution. This is outside the SDK and unqualified
for model inference, source protection, detector power or production serving.
"""
import math

from keyprint._engine.research.keyprint_exact_categorical_v2 import integer_distribution

MARGIN_DENOMINATOR=2**40


def distribution(values):
    values=tuple(values)
    if not 1<=len(values)<=4096: raise ValueError('Require 1 to 4096 weights')
    return integer_distribution(values)


class PacedKernel:
    def __init__(self,base):
        self.base=tuple(base)
        dist=distribution(self.base)
        self.weights,self.total=dist.weights,dist.total
        if any(0<b and b*2**900<self.total for b in self.weights):
            raise ValueError('Base probability below admitted research range')

    def _strength(self,current):
        """Integer comparisons select a rational strength without observing labels."""
        if len(current.weights)!=len(self.weights): raise ValueError('Shape changed')
        B,Q,G=self.total,current.total,MARGIN_DENOMINATOR
        numerator,denominator=1,1
        freeze=False
        for b,q in zip(self.weights,current.weights):
            if (b>0)!=(q>0): raise ValueError('Support changed')
            low,high=2*B*q-b*Q,2*b*Q-B*q
            if low<0 or high<0: raise ValueError('Current probability outside bounds')
            if 0<q<Q:
                lo_room,hi_room=G*low-2*b*Q,G*high-b*Q
                if min(lo_room,hi_room)<=0:
                    freeze=True
                    continue
                radius=B*G*q*(Q-q)
                for n,d in ((Q*lo_room,2*radius),(Q*hi_room,radius),
                            (b*(G-8)*Q*Q,8*radius)):
                    if n*denominator<numerator*d: numerator,denominator=n,d
        if freeze: return 0.,True
        alpha=numerator/denominator
        a,b=alpha.as_integer_ratio()
        if a*denominator>numerator*b: alpha=math.nextafter(alpha,0.)
        return alpha,False

    def _certify(self,before,after):
        B,Q,R=self.total,before.total,after.total
        if len(after.weights)!=len(self.weights): raise ArithmeticError('Output shape changed')
        for b,q,r in zip(self.weights,before.weights,after.weights):
            if (b>0)!=(r>0): raise ArithmeticError('Output support changed')
            if (2*B*r<b*R or B*r>2*b*R
                    or 8*B*abs(r*Q-q*R)>b*R*Q):
                raise ArithmeticError('Rounded categorical probabilities violate budget')

    def step(self,current,bits):
        current=tuple(current); before=distribution(current)
        alpha,freeze=self._strength(before)
        bits=tuple(bits)
        if len(bits)!=len(current) or any(type(g) is not int or g not in (0,1) for g in bits):
            raise ValueError('Require binary labels matching weights')
        labels={g for q,g in zip(before.weights,bits) if q>0}
        if alpha==0 or len(labels)<2:
            output=current
        else:
            Q=before.total
            mean=sum(q for q,g in zip(before.weights,bits) if g)/Q
            output=tuple((q/Q)*(1+alpha*(g-mean)) for q,g in zip(before.weights,bits))
        after=distribution(output)
        self._certify(before,after)
        return output,{'strength':alpha,'near_boundary_freeze':freeze}
