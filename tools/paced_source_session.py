"""Separate research session with exact partition mass and explicit integer draws.

Not a drop-in float Prepared object. Never round the returned integer distribution
back to floats for sampling: that would invalidate exact excluded-mass guarantees.
Source alignment/lifecycle reuse the frozen research implementation; numeric
policy, PRF profile, prepared representation and draw contract are new.
"""
from dataclasses import dataclass
from functools import reduce
import math

import numpy as np

from keyprint._engine.legacy._impl.research.grouped_canonical_prototype import Config,Profile
from keyprint._engine.legacy._impl.research.token_source_policy import TokenSourceSession
from keyprint._engine.legacy._impl.research.byte_trie_source_policy import SourceRequest,MATCH_BYTES,STARTUP_BYTES,digest_spec
from keyprint._engine.research.keyprint_exact_categorical_v2 import (
    IntegerDistribution,BitDraw,Sample,integer_distribution,index_at_integer_point)
from paced_integer_kernel import PacedKernel


def policy_spec():
    return {'version':'keyprint-paced-source-integer-v1',
        'kernel':'reserved-base-relative-margin-v1','ratio_bounds':['1/2','2'],
        'per_layer_base_fraction':'1/8','rounding_margin':'2^-40',
        'partition':'exact_integer_product_preserves_excluded_normalized_mass',
        'draw':'integer_rejection_gcd_reduced_no_float_conversion',
        'source_match_bytes':MATCH_BYTES,'startup_bytes':STARTUP_BYTES,
        'source_alignment':'all_overlapping_canonical_suffix_matches',
        'canonicalization':'ASCII_whitespace_only_no_Unicode_normalization',
        'repeated_context':'identity','single_distinct_label':'identity',
        'purpose':'explicit_source_request','detector':'uncalibrated_raw_events_only'}


@dataclass(frozen=True,init=False)
class PacedProfile(Profile):
    eos_ids:frozenset[int]
    def __init__(self,token_bytes,*,tokenizer_identity,eos_ids,config=Config()):
        pieces=tuple(token_bytes);eos=frozenset(eos_ids)
        if (not eos or any(type(i) is not int or not 0<=i<len(pieces) or pieces[i] is not None for i in eos)):
            raise ValueError('Explicit EOS IDs must map to excluded None pieces')
        identity=tokenizer_identity+':'+digest_spec({'policy':policy_spec(),'eos_ids':sorted(eos)})
        super().__init__(pieces,tokenizer_identity=identity,config=config)
        object.__setattr__(self,'eos_ids',eos)


@dataclass(frozen=True,eq=False)
class IntegerPrepared:
    index:int
    token_ids:tuple[int,...]
    distribution:IntegerDistribution


class SamplingFailure(RuntimeError):
    def __init__(self,reason,transcript,calls):
        super().__init__(reason)
        self.transcript=tuple(transcript)
        self.callback_calls=calls


def partition(base,token_ids,profile,key,context,protected):
    """Return sparse global integer weights; excluded probabilities are exact."""
    initial=integer_distribution(base)
    active=[j for j,i in enumerate(token_ids) if i not in protected and profile.classes[i] is not None]
    labels=dict.fromkeys(profile.classes[token_ids[j]] for j in active)
    if len(labels)<2:
        return initial,{'executed_layers':0,'boundary_freezes':0,'partitioned':False}
    values=tuple(base[j] for j in active)
    kernel=PacedKernel(values);current=values;freezes=0
    for label in labels: labels[label]=profile.bits(key,context,label)
    for layer in range(profile.config.layers):
        bits=[labels[profile.classes[token_ids[j]]][layer] for j in active]
        current,receipt=kernel.step(current,bits)
        freezes+=receipt['near_boundary_freeze']
    conditional=integer_distribution(current)
    mass=sum(initial.weights[j] for j in active)
    output=[b*conditional.total for b in initial.weights]
    for j,r in zip(active,conditional.weights): output[j]=mass*r
    divisor=reduce(math.gcd,output)
    result=IntegerDistribution(tuple(r//divisor for r in output),initial.total*conditional.total//divisor)
    if sum(result.weights)!=result.total:
        raise ArithmeticError('Partition mass changed')
    active_set=set(active)
    for j,(b,r) in enumerate(zip(initial.weights,result.weights)):
        if r<=0 or 2*initial.total*r<b*result.total or initial.total*r>2*b*result.total:
            raise ArithmeticError('Whole-distribution support or bound changed')
        if j not in active_set and r*initial.total!=b*result.total:
            raise ArithmeticError('Excluded normalized probability changed')
    return result,{'executed_layers':profile.config.layers,'boundary_freezes':freezes,'partitioned':True}


class PacedSourceSession(TokenSourceSession):
    def __init__(self,profile,key,*,condition,request=SourceRequest()):
        if type(profile) is not PacedProfile:
            raise TypeError('Explicit paced research profile required; old calibration cannot transfer')
        super().__init__(profile,key,condition=condition,request=request)
        self._drawn=None
        self._paced_counts={'executed_layers':0,'boundary_freezes':0,'integer_partitions':0}

    def prepare(self,probabilities):
        self._ready()
        if self._pending is not None: raise RuntimeError('Commit or close pending step first')
        if self._steps>=self.profile.config.max_steps: raise RuntimeError('Response step cap reached')
        if not isinstance(probabilities,np.ndarray) or probabilities.dtype!=np.float64:
            raise TypeError('NumPy float64 probabilities required')
        q=probabilities.copy()
        if (q.shape!=(len(self.profile.classes),) or not np.isfinite(q).all() or (q<0).any()):
            raise ValueError('Invalid base probability vector')
        ids=tuple(map(int,np.flatnonzero(q>0)));base=tuple(float(q[i]) for i in ids)
        if abs(math.fsum(base)-1)>1e-12: raise ValueError('Invalid base probability mass')
        try:
            mode,protected,occurrences=self._decision(q)
            eligible=self._condition=='marked' and mode!='startup_ordinary' and self._context not in self._used
            if eligible:
                dist,numeric=partition(base,ids,self.profile,self._key,self._context,frozenset(protected))
            else:
                dist=integer_distribution(base)
                numeric={'executed_layers':0,'boundary_freezes':0,'partitioned':False}
        except Exception:
            self.close()  # No alternate distribution or retried numerical proposal.
            raise
        self._last_decision={'mode':mode,'protected_token_ids':protected,
            'alignment_occurrences':occurrences,'integer_sampling':True,**numeric}
        self._policy_counts[mode]+=1
        self._protected_candidates+=len(protected);self._alignment_occurrences+=occurrences
        self._paced_counts['executed_layers']+=numeric['executed_layers']
        self._paced_counts['boundary_freezes']+=numeric['boundary_freezes']
        self._paced_counts['integer_partitions']+=numeric['partitioned']
        self._pending=IntegerPrepared(self._steps,ids,dist)
        self._support=q>0;self._drawn=None
        return self._pending

    def draw(self,prepared,random_bits,*,max_draws=1024):
        self._ready()
        if prepared is not self._pending or prepared is None: raise ValueError('Foreign or stale prepared step')
        if self._drawn is not None: raise RuntimeError('Prepared step already drawn')
        if type(max_draws) is not int or not 1<=max_draws<=1024: raise ValueError('Invalid draw cap')
        if not callable(random_bits): raise TypeError('Explicit callable random-bit source required')
        transcript=[];calls=0
        try:
            d=prepared.distribution;count=(d.total-1).bit_length()
            if count==0:
                self._drawn=Sample(prepared.token_ids[0],0,d.total,())
                return self._drawn
            for _ in range(max_draws):
                calls+=1;point=random_bits(count)
                if type(point) is not int or not 0<=point<2**count:
                    raise ValueError('Invalid random-bit callback result')
                accepted=point<d.total
                transcript.append(BitDraw(count,point,accepted))
                if accepted:
                    index=index_at_integer_point(d,point)
                    self._drawn=Sample(prepared.token_ids[index],point,d.total,tuple(transcript))
                    return self._drawn
            raise RuntimeError('Integer draw cap exhausted')
        except BaseException as error:
            self.close()
            if isinstance(error,Exception):
                raise SamplingFailure(type(error).__name__,transcript,calls) from error
            raise

    def commit(self,prepared,token_id):
        self._ready()
        if self._drawn is None or token_id!=self._drawn.token_index:
            raise ValueError('Commit must match this session\'s recorded draw')
        event=super().commit(prepared,token_id)
        self._drawn=None
        if token_id in self.profile.eos_ids: self.close()
        return event

    def source_receipt(self):
        receipt=super().source_receipt()
        receipt['inherited_source_policy_sha256']=receipt.pop('source_policy_sha256')
        receipt.update(source_policy_sha256=digest_spec(policy_spec()),profile_sha256=self.profile.digest,
            numeric_counters=dict(self._paced_counts),integer_sampling=True,
            detector_calibrated=False,quality_acceptance=False)
        return receipt
